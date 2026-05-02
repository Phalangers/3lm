#!/usr/bin/env node

import { spawn, execSync } from "node:child_process";
import { createInterface } from "node:readline";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync, mkdirSync, renameSync, unlinkSync, createWriteStream } from "node:fs";
import { arch, platform } from "node:os";

const __dirname = dirname(fileURLToPath(import.meta.url));

// --- platform detection ---

const LLAMA_VERSION = "b8918";

function detectPlatform() {
  const os = platform();
  const cpu = arch();
  if (os === "android" && cpu === "arm64") return "android-arm64";
  if (os === "linux" && cpu === "arm64") return "linux-arm64";
  if (os === "linux" && cpu === "x64") return "linux-x64";
  if (os === "darwin" && cpu === "arm64") return "macos-arm64";
  if (os === "darwin" && cpu === "x64") return "macos-x64";
  return null;
}

const PLATFORM_ASSETS = {
  "linux-arm64":   `llama-${LLAMA_VERSION}-bin-ubuntu-arm64.tar.gz`,
  "linux-x64":     `llama-${LLAMA_VERSION}-bin-ubuntu-x64.tar.gz`,
  "android-arm64": `llama-${LLAMA_VERSION}-bin-android-arm64.tar.gz`,
  "macos-arm64":   `llama-${LLAMA_VERSION}-bin-macos-arm64.tar.gz`,
  "macos-x64":     `llama-${LLAMA_VERSION}-bin-macos-x64.tar.gz`,
};

const plat = detectPlatform();
const LLAMA_DIR = join(__dirname, "bin", plat || "unknown");
const LLAMA_SERVER = join(LLAMA_DIR, "llama-server");

const MODELS = {
  qwen: {
    name: "Qwen3 0.6B",
    file: "qwen3-0.6b-q8_0.gguf",
    url: "https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
    port: 8787,
    noThinkTag: "/no_think",
  },
  gemma: {
    name: "Gemma 4 E2B",
    file: "gemma-4-E2B-it-Q4_K_M.gguf",
    url: "https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF/resolve/main/google_gemma-4-E2B-it-Q4_K_M.gguf",
    port: 8788,
    extraArgs: (thinking) => ["--flash-attn", "on", "--reasoning", thinking ? "on" : "off"],
  },
};

// --- helpers ---

let quiet = false;

function fatal(msg) {
  console.error(`\x1b[31merror:\x1b[0m ${msg}`);
  process.exit(1);
}

function log(msg) {
  if (!quiet) console.log(msg);
}

async function downloadToFile(url, label, destPath) {
  log(`Downloading ${label}...`);
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok) fatal(`download failed: ${res.status} ${res.statusText}`);

  const total = parseInt(res.headers.get("content-length") || "0");
  let downloaded = 0;
  const fileStream = createWriteStream(destPath);
  const reader = res.body.getReader();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    fileStream.write(value);
    downloaded += value.length;
    const mb = (downloaded / 1e6).toFixed(0);
    const totalMb = total ? ` / ${(total / 1e6).toFixed(0)} MB` : "";
    const pct = total ? ` (${((downloaded / total) * 100).toFixed(0)}%)` : "";
    if (!quiet) process.stdout.write(`\r  ${mb}${totalMb}${pct}  `);
  }
  if (!quiet) console.log();
  await new Promise((resolve, reject) => {
    fileStream.on("finish", resolve);
    fileStream.on("error", reject);
    fileStream.end();
  });
}

async function downloadModel(model) {
  const modelsDir = join(__dirname, "models");
  mkdirSync(modelsDir, { recursive: true });

  const dest = join(modelsDir, model.file);
  const tmp = dest + ".tmp";

  await downloadToFile(model.url, model.name, tmp);
  renameSync(tmp, dest);
  log(`  Saved to ${dest}`);
}

async function ensureBinaries() {
  if (existsSync(LLAMA_SERVER)) return;
  if (!plat) fatal(`unsupported platform: ${platform()}/${arch()}`);

  const asset = PLATFORM_ASSETS[plat];
  if (!asset) fatal(`no prebuilt binary for ${plat}`);

  const url = `https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_VERSION}/${asset}`;
  const tmp = join(__dirname, "bin", `${plat}.tmp.tar.gz`);
  mkdirSync(join(__dirname, "bin"), { recursive: true });

  await downloadToFile(url, `llama.cpp for ${plat}`, tmp);

  mkdirSync(LLAMA_DIR, { recursive: true });
  execSync(`tar xzf "${tmp}" -C "${LLAMA_DIR}" --strip-components=1`);
  unlinkSync(tmp);
  log(`  Installed to ${LLAMA_DIR}`);
}

async function serverAlive(baseUrl) {
  try {
    const res = await fetch(`${baseUrl}/health`);
    const body = await res.json();
    return body.status === "ok";
  } catch {
    return false;
  }
}

async function waitForServer(baseUrl, maxWait = 120_000) {
  const start = Date.now();
  while (Date.now() - start < maxWait) {
    if (await serverAlive(baseUrl)) return true;
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

async function ensureServer(baseUrl, model, opts) {
  if (await serverAlive(baseUrl)) return;

  if (opts.proc) {
    opts.proc.kill("SIGKILL");
    opts.proc = null;
  }

  await ensureBinaries();

  log(`Starting ${model.name}...`);

  const serverArgs = [
    "-m", opts.modelPath,
    "-c", String(opts.ctxSize),
    "-t", String(opts.threads),
    "--port", String(model.port),
    "--no-webui",
    "-np", "1",
    ...(model.extraArgs ? model.extraArgs(opts.showThinking) : []),
  ];

  const proc = spawn(LLAMA_SERVER, serverArgs, {
    env: { ...process.env, LD_LIBRARY_PATH: `${LLAMA_DIR}:${process.env.LD_LIBRARY_PATH || ""}` },
    stdio: ["ignore", "pipe", "pipe"],
  });

  let serverLog = "";
  proc.stderr.on("data", (d) => { serverLog += d.toString(); });
  proc.stdout.on("data", (d) => { serverLog += d.toString(); });

  proc.on("error", (err) => fatal(`failed to start llama-server: ${err.message}`));
  proc.on("exit", (code) => {
    if (code !== null && code !== 0 && !proc.killed) {
      console.error(serverLog.slice(-1000));
      console.error(`\x1b[31mllama-server exited with code ${code}\x1b[0m`);
    }
    opts.proc = null;
  });

  opts.proc = proc;

  if (!await waitForServer(baseUrl)) {
    console.error(serverLog.slice(-1000));
    fatal("llama-server failed to start within timeout");
  }
}

async function chat(baseUrl, messages, modelLabel) {
  const res = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, stream: true }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`server error ${res.status}: ${text}`);
  }

  let full = "";
  let inThinking = false;
  let startedContent = false;
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    const lines = buf.split("\n");
    buf = lines.pop();

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6).trim();
      if (data === "[DONE]") continue;
      try {
        const json = JSON.parse(data);
        const delta = json.choices?.[0]?.delta;

        const reasonToken = delta?.reasoning_content;
        if (reasonToken && !quiet) {
          if (!inThinking) {
            inThinking = true;
            process.stdout.write("\x1b[2m(thinking) ");
          }
          process.stdout.write(reasonToken);
        }

        const token = delta?.content;
        if (token) {
          if (inThinking && !startedContent && !quiet) {
            inThinking = false;
            startedContent = true;
            process.stdout.write(`\x1b[0m\n\x1b[33m${modelLabel}>\x1b[0m `);
          }
          process.stdout.write(token);
          full += token;
        }
      } catch {}
    }
  }

  if (inThinking && !quiet) process.stdout.write("\x1b[0m");
  process.stdout.write("\n");
  return full;
}

// --- main ---

// handle "download" subcommand
if (process.argv[2] === "download") {
  const which = process.argv[3];
  if (which === "bin" || which === "binaries") {
    await ensureBinaries();
    process.exit(0);
  }
  const toDownload = which
    ? { [which]: MODELS[which] || fatal(`unknown model "${which}". Available: ${Object.keys(MODELS).join(", ")}, bin`) }
    : { qwen: MODELS.qwen };
  for (const [key, m] of Object.entries(toDownload)) {
    const dest = join(__dirname, "models", m.file);
    if (existsSync(dest)) {
      console.log(`${m.name} already downloaded.`);
    } else {
      await downloadModel(m);
    }
  }
  process.exit(0);
}

const args = process.argv.slice(2);
let modelKey = "qwen";
let ctxSize = 2048;
let threads = 8;
let showThinking = false;
let promptArg = null;

for (let i = 0; i < args.length; i++) {
  if ((args[i] === "--model" || args[i] === "-m") && args[i + 1]) modelKey = args[++i];
  else if (args[i] === "--ctx" && args[i + 1]) ctxSize = parseInt(args[++i]);
  else if (args[i] === "--threads" && args[i + 1]) threads = parseInt(args[++i]);
  else if (args[i] === "--think") showThinking = true;
  else if ((args[i] === "--prompt" || args[i] === "-p") && args[i + 1]) promptArg = args[++i];
  else if (args[i] === "--help" || args[i] === "-h") {
    console.log(`Usage: node llm.mjs [options]
  node llm.mjs download [qwen|gemma|bin]

Options:
  -m, --model   Model to use: ${Object.keys(MODELS).join(", ")} (default: qwen)
  -p, --prompt  Send a single prompt and exit (non-interactive)
  --ctx N       Context size (default: 2048)
  --threads N   CPU threads (default: 8)
  --think       Enable model thinking/reasoning
  -h, --help    Show this help`);
    process.exit(0);
  }
}

const model = MODELS[modelKey];
if (!model) fatal(`unknown model "${modelKey}". Available: ${Object.keys(MODELS).join(", ")}`);

const modelPath = join(__dirname, "models", model.file);
if (!existsSync(modelPath)) {
  await downloadModel(model);
}

const baseUrl = `http://127.0.0.1:${model.port}`;
const opts = { proc: null, modelPath, ctxSize, threads, showThinking };

await ensureServer(baseUrl, model, opts);

function cleanup() {
  if (opts.proc) opts.proc.kill("SIGTERM");
}
process.on("SIGINT", () => { cleanup(); process.exit(0); });
process.on("SIGTERM", () => { cleanup(); process.exit(0); });
process.on("exit", cleanup);

const label = modelKey;

if (promptArg) {
  quiet = true;
  const content = (!showThinking && model.noThinkTag)
    ? `${model.noThinkTag}\n${promptArg}`
    : promptArg;
  try {
    await chat(baseUrl, [{ role: "user", content }], label);
  } catch (err) {
    console.error(`\x1b[31merror:\x1b[0m ${err.cause?.code || err.message}`);
  }
  cleanup();
  process.exit(0);
}

console.log(`\x1b[32m${model.name} ready!\x1b[0m (ctx=${ctxSize}, threads=${threads})`);
console.log('Type your message. "/clear" to reset, "/quit" to exit.\n');

const rl = createInterface({ input: process.stdin, output: process.stdout });
const prompt = () => new Promise((resolve) => rl.question("\x1b[36myou>\x1b[0m ", resolve));

let messages = [];

while (true) {
  const input = await prompt();
  const trimmed = input.trim();

  if (!trimmed) continue;
  if (trimmed === "/quit" || trimmed === "/exit") break;
  if (trimmed === "/clear") {
    messages = [];
    console.log("(conversation cleared)\n");
    continue;
  }

  const content = (!showThinking && model.noThinkTag)
    ? `${model.noThinkTag}\n${trimmed}`
    : trimmed;
  messages.push({ role: "user", content });

  process.stdout.write(`\x1b[33m${label}>\x1b[0m `);
  try {
    await ensureServer(baseUrl, model, opts);
    const reply = await chat(baseUrl, messages, label);
    messages.push({ role: "assistant", content: reply });
  } catch (err) {
    console.error(`\x1b[31merror:\x1b[0m ${err.cause?.code || err.message}`);
    messages.pop();
  }
  console.log();
}

cleanup();
rl.close();
process.exit(0);
