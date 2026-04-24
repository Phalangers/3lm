#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";

const __dirname = dirname(fileURLToPath(import.meta.url));

const LLAMA_DIR = join(__dirname, "bin", "llama-b8902");
const LLAMA_SERVER = join(LLAMA_DIR, "llama-server");

const MODELS = {
  qwen: {
    name: "Qwen3 0.6B",
    file: "qwen3-0.6b-q8_0.gguf",
    port: 8787,
    noThinkTag: "/no_think",
  },
  gemma: {
    name: "Gemma 4 E2B",
    file: "gemma-4-E2B-it-Q4_K_M.gguf",
    port: 8788,
    extraArgs: (thinking) => ["--flash-attn", "on", "--reasoning", thinking ? "on" : "off"],
  },
};

// --- helpers ---

function fatal(msg) {
  console.error(`\x1b[31merror:\x1b[0m ${msg}`);
  process.exit(1);
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

  // kill stale process if we own one
  if (opts.proc) {
    opts.proc.kill("SIGKILL");
    opts.proc = null;
  }

  console.log(`Starting ${model.name}...`);

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
        if (reasonToken) {
          if (!inThinking) {
            inThinking = true;
            process.stdout.write("\x1b[2m(thinking) ");
          }
          process.stdout.write(reasonToken);
        }

        const token = delta?.content;
        if (token) {
          if (inThinking && !startedContent) {
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

  if (inThinking) process.stdout.write("\x1b[0m");
  process.stdout.write("\n");
  return full;
}

// --- main ---

const args = process.argv.slice(2);
let modelKey = "qwen";
let ctxSize = 2048;
let threads = 8;
let showThinking = false;

for (let i = 0; i < args.length; i++) {
  if ((args[i] === "--model" || args[i] === "-m") && args[i + 1]) modelKey = args[++i];
  if (args[i] === "--ctx" && args[i + 1]) ctxSize = parseInt(args[++i]);
  if (args[i] === "--threads" && args[i + 1]) threads = parseInt(args[++i]);
  if (args[i] === "--think") showThinking = true;
  if (args[i] === "--help" || args[i] === "-h") {
    console.log(`Usage: node llm.mjs [options]
Options:
  -m, --model   Model to use: ${Object.keys(MODELS).join(", ")} (default: qwen)
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
if (!existsSync(modelPath)) fatal(`model not found: ${modelPath}`);

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
