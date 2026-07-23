/**
 * Test harness for the auto-failover extension.
 * Runs with plain Node (24+):  node test/run.mjs
 * Mocks ExtensionAPI/ExtensionContext and drives the event handlers.
 */
import { strict as assert } from "node:assert";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const TEST_DIR = "/tmp/pi-auto-failover-test";
rmSync(TEST_DIR, { recursive: true, force: true });
mkdirSync(TEST_DIR, { recursive: true });
process.env.PI_CODING_AGENT_DIR = TEST_DIR;

const { default: extension, suggestFilterTokens, pickModelWithSearch } = await import("../index.ts");

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const usage = () => ({
	input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
});

const mkModel = (provider, id) => ({
	provider, id, name: id, api: "anthropic-messages", baseUrl: "https://example.com",
	reasoning: true, input: ["text"], cost: { input: 1, output: 1, cacheRead: 1, cacheWrite: 1 },
	contextWindow: 200000, maxTokens: 8192,
});

const ANTHROPIC = mkModel("anthropic", "claude-opus-4-5");
const OPENROUTER = mkModel("openrouter", "anthropic/claude-opus-4.5");
const BEDROCK = mkModel("amazon-bedrock", "anthropic.claude-opus-4-5-20251101-v1:0");
const ALL_MODELS = [ANTHROPIC, OPENROUTER, BEDROCK];

const CHAIN = [ANTHROPIC, OPENROUTER, BEDROCK].map((m) => `${m.provider}/${m.id}`);

const errMsg = (model, errorMessage, stopReason = "error") => ({
	role: "assistant", content: [], api: model.api, provider: model.provider, model: model.id,
	usage: usage(), stopReason, errorMessage, timestamp: Date.now(),
});

function makePi() {
	const handlers = {};
	const calls = { setModel: [], sendMessage: [] };
	const pi = {
		handlers, calls, command: undefined, setModelResult: true,
		on(ev, h) { (handlers[ev] ??= []).push(h); },
		registerCommand(name, def) { this.command = { name, ...def }; },
		async setModel(m) { calls.setModel.push(m); return this.setModelResult; },
		sendMessage(msg, opts) { calls.sendMessage.push({ msg, opts }); },
	};
	return pi;
}

function makeCtx({ model, idle = true, hasAuth = () => true }) {
	const notifications = [];
	const statuses = {};
	const ctx = {
		hasUI: false, mode: "rpc", cwd: "/tmp", model, signal: undefined,
		isIdle: () => idle,
		modelRegistry: {
			find: (p, id) => ALL_MODELS.find((m) => m.provider === p && m.id === id),
			hasConfiguredAuth: hasAuth,
			getAll: () => ALL_MODELS,
			getAvailable: () => ALL_MODELS,
			getProviderDisplayName: (p) => p,
		},
		ui: {
			notify: (msg, type) => notifications.push({ msg, type }),
			setStatus: (k, t) => { statuses[k] = t; },
			select: async () => undefined,
			confirm: async () => false,
			input: async () => undefined,
		},
	};
	return { ctx, notifications, statuses };
}

const emit = async (pi, ev, event, ctx) => {
	for (const h of pi.handlers[ev] ?? []) await h(event, ctx);
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Simulate pi emitting model_select after a successful setModel. */
const completeSwitch = async (pi, ctx, from, to) => {
	ctx.model = to;
	await emit(pi, "model_select", { type: "model_select", model: to, previousModel: from, source: "set" }, ctx);
};

function writeConfig(cfg) {
	writeFileSync(join(TEST_DIR, "auto-failover.json"), JSON.stringify(cfg, null, 2));
}

function freshExtension(cfg) {
	writeConfig(cfg);
	const pi = makePi();
	extension(pi);
	return pi;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test("transient error after retries → fails over to backup and re-triggers turn", async () => {
	const pi = freshExtension({ enabled: true, cooldownSeconds: 300, chains: [CHAIN] });
	const { ctx, notifications } = makeCtx({ model: ANTHROPIC });

	await emit(pi, "message_end", { type: "message_end", message: errMsg(ANTHROPIC, "429 rate limit exceeded") }, ctx);

	assert.equal(pi.calls.setModel.length, 1, "one setModel call");
	assert.equal(pi.calls.setModel[0].provider, "openrouter");
	assert.equal(pi.calls.setModel[0].id, "anthropic/claude-opus-4.5");
	assert.equal(pi.calls.sendMessage.length, 1, "retry turn triggered");
	assert.equal(pi.calls.sendMessage[0].opts.triggerTurn, true);
	assert.equal(pi.calls.sendMessage[0].opts.deliverAs, "followUp");
	assert.equal(pi.calls.sendMessage[0].msg.display, false);
	assert.ok(notifications.some((n) => n.msg.includes("retrying on openrouter")), "notify mentions backup");

	// Programmatic switch must not clear state.
	await completeSwitch(pi, ctx, ANTHROPIC, OPENROUTER);
	assert.ok(ctx.ui && pi.handlers, "sanity");
});

test("second failure on backup, same prompt → chain exhausted, no further switch", async () => {
	const pi = freshExtension({ enabled: true, cooldownSeconds: 300, chains: [CHAIN] });
	const { ctx, notifications } = makeCtx({ model: ANTHROPIC });

	await emit(pi, "message_end", { type: "message_end", message: errMsg(ANTHROPIC, "overloaded") }, ctx);
	await completeSwitch(pi, ctx, ANTHROPIC, OPENROUTER);
	await emit(pi, "message_end", { type: "message_end", message: errMsg(OPENROUTER, "503 service unavailable") }, ctx);
	await completeSwitch(pi, ctx, OPENROUTER, BEDROCK);
	await emit(pi, "message_end", { type: "message_end", message: errMsg(BEDROCK, "timeout") }, ctx);

	assert.equal(pi.calls.setModel.length, 2, "two hops only (anthropic→openrouter→bedrock)");
	assert.equal(pi.calls.sendMessage.length, 2);
	assert.ok(
		notifications.some((n) => n.msg.includes("no usable backup") && n.type === "error"),
		"exhaustion notified",
	);
});

test("new user message resets the per-prompt loop guard (may fail back toward primary)", async () => {
	const pi = freshExtension({ enabled: true, cooldownSeconds: 300, chains: [CHAIN] });
	const { ctx } = makeCtx({ model: ANTHROPIC });

	await emit(pi, "message_end", { type: "message_end", message: errMsg(ANTHROPIC, "429") }, ctx);
	await completeSwitch(pi, ctx, ANTHROPIC, OPENROUTER);
	assert.equal(pi.calls.setModel.length, 1);

	// User sends a real message while failed over.
	await emit(pi, "message_end", { type: "message_end", message: { role: "user", content: "next task", timestamp: Date.now() } }, ctx);

	// Backup fails again on the new prompt → may hop to the first untried provider (anthropic).
	await emit(pi, "message_end", { type: "message_end", message: errMsg(OPENROUTER, "429") }, ctx);
	assert.equal(pi.calls.setModel.length, 2, "one more hop allowed for the new prompt");
	assert.equal(pi.calls.setModel[1].provider, "anthropic");
});

test("non-transient errors (auth/quota) do not fail over", async () => {
	const pi = freshExtension({ enabled: true, cooldownSeconds: 300, chains: [CHAIN] });
	const { ctx } = makeCtx({ model: ANTHROPIC });

	await emit(pi, "message_end", { type: "message_end", message: errMsg(ANTHROPIC, "401 unauthorized: invalid api key") }, ctx);
	await emit(pi, "message_end", { type: "message_end", message: errMsg(ANTHROPIC, "429 insufficient_quota: billing hard limit") }, ctx);

	assert.equal(pi.calls.setModel.length, 0);
	assert.equal(pi.calls.sendMessage.length, 0);
});

test("context overflow is left to compaction recovery", async () => {
	const pi = freshExtension({ enabled: true, cooldownSeconds: 300, chains: [CHAIN] });
	const { ctx } = makeCtx({ model: ANTHROPIC });

	await emit(pi, "message_end", { type: "message_end", message: errMsg(ANTHROPIC, "prompt is too long: 250000 tokens > 200000 maximum") }, ctx);

	assert.equal(pi.calls.setModel.length, 0);
});

test("aborted requests never fail over", async () => {
	const pi = freshExtension({ enabled: true, cooldownSeconds: 300, chains: [CHAIN] });
	const { ctx } = makeCtx({ model: ANTHROPIC });

	await emit(pi, "message_end", { type: "message_end", message: errMsg(ANTHROPIC, undefined, "aborted") }, ctx);

	assert.equal(pi.calls.setModel.length, 0);
});

test("model without a configured chain is ignored", async () => {
	const pi = freshExtension({ enabled: true, cooldownSeconds: 300, chains: [CHAIN] });
	const GPT = mkModel("openai", "gpt-5");
	const { ctx } = makeCtx({ model: GPT });

	await emit(pi, "message_end", { type: "message_end", message: errMsg(GPT, "429") }, ctx);

	assert.equal(pi.calls.setModel.length, 0);
});

test("backup without API key is skipped", async () => {
	const pi = freshExtension({ enabled: true, cooldownSeconds: 300, chains: [CHAIN] });
	const { ctx, notifications } = makeCtx({
		model: ANTHROPIC,
		hasAuth: (m) => m.provider !== "openrouter", // openrouter has no key
	});

	await emit(pi, "message_end", { type: "message_end", message: errMsg(ANTHROPIC, "429") }, ctx);

	assert.equal(pi.calls.setModel.length, 1);
	assert.equal(pi.calls.setModel[0].provider, "amazon-bedrock", "skips keyless openrouter");
	assert.ok(notifications.some((n) => n.msg.includes("retrying on amazon-bedrock")));
});

test("auto-return to primary after cooldown when idle", async () => {
	const pi = freshExtension({ enabled: true, cooldownSeconds: 1, chains: [CHAIN] });
	const { ctx, notifications } = makeCtx({ model: ANTHROPIC });

	await emit(pi, "message_end", { type: "message_end", message: errMsg(ANTHROPIC, "429") }, ctx);
	await completeSwitch(pi, ctx, ANTHROPIC, OPENROUTER);
	assert.equal(pi.calls.setModel.length, 1);

	await sleep(1200); // cooldown elapses while idle

	assert.equal(pi.calls.setModel.length, 2, "auto-return fired");
	assert.equal(pi.calls.setModel[1].provider, "anthropic");
	assert.ok(notifications.some((n) => n.msg.includes("returned to primary anthropic")));
	// Status cleared after return (resetState → setStatus undefined).
});

test("cooldown expiring mid-turn waits for agent_settled", async () => {
	const pi = freshExtension({ enabled: true, cooldownSeconds: 1, chains: [CHAIN] });
	const busyCtx = makeCtx({ model: ANTHROPIC, idle: false });
	const notifications = busyCtx.notifications;

	await emit(pi, "message_end", { type: "message_end", message: errMsg(ANTHROPIC, "429") }, busyCtx.ctx);
	busyCtx.ctx.model = OPENROUTER;

	await sleep(1200); // cooldown elapses while busy
	assert.equal(pi.calls.setModel.length, 1, "no return while busy");

	busyCtx.ctx.isIdle = () => true;
	await emit(pi, "agent_settled", { type: "agent_settled" }, busyCtx.ctx);
	assert.equal(pi.calls.setModel.length, 2, "returned once settled");
	assert.equal(pi.calls.setModel[1].provider, "anthropic");
	assert.ok(notifications.some((n) => n.msg.includes("returned to primary")));
});

test("manual model change cancels failover state and auto-return", async () => {
	const pi = freshExtension({ enabled: true, cooldownSeconds: 1, chains: [CHAIN] });
	const { ctx, statuses } = makeCtx({ model: ANTHROPIC });

	await emit(pi, "message_end", { type: "message_end", message: errMsg(ANTHROPIC, "429") }, ctx);
	await completeSwitch(pi, ctx, ANTHROPIC, OPENROUTER);

	// User manually picks an unrelated model.
	const GPT = mkModel("openai", "gpt-5");
	await emit(pi, "model_select", { type: "model_select", model: GPT, previousModel: OPENROUTER, source: "set" }, ctx);

	assert.equal(statuses["auto-failover"], undefined, "status cleared");
	await sleep(1200);
	assert.equal(pi.calls.setModel.length, 1, "no auto-return after user takeover");
});

test("/failover command: status, cooldown, disable/enable, reset", async () => {
	const pi = freshExtension({ enabled: true, cooldownSeconds: 300, chains: [CHAIN] });
	const { ctx, notifications } = makeCtx({ model: ANTHROPIC });

	await pi.command.handler("status", ctx);
	assert.ok(notifications.some((n) => n.msg.includes("enabled: true")), "status prints config");
	assert.ok(notifications.some((n) => n.msg.includes(CHAIN.join("  →  "))), "status prints chain");

	await pi.command.handler("cooldown 60", ctx);
	assert.ok(notifications.some((n) => n.msg.includes("cooldown set to 60s")));

	// Fail over, then reset.
	await emit(pi, "message_end", { type: "message_end", message: errMsg(ANTHROPIC, "429") }, ctx);
	await completeSwitch(pi, ctx, ANTHROPIC, OPENROUTER);
	assert.equal(pi.calls.setModel.length, 1);

	await pi.command.handler("reset", ctx);
	assert.equal(pi.calls.setModel.length, 2, "reset returns to primary");
	assert.equal(pi.calls.setModel[1].provider, "anthropic");

	// Disable: further failures ignored.
	await pi.command.handler("disable", ctx);
	await emit(pi, "message_end", { type: "message_end", message: errMsg(ANTHROPIC, "429") }, ctx);
	assert.equal(pi.calls.setModel.length, 2, "disabled → no failover");

	await pi.command.handler("enable", ctx);
	assert.ok(notifications.some((n) => n.msg === "auto-failover: enabled"));
});

test("setModel failure (no key at switch time) keeps things sane", async () => {
	const pi = freshExtension({ enabled: true, cooldownSeconds: 300, chains: [CHAIN] });
	pi.setModelResult = false;
	const { ctx, notifications } = makeCtx({ model: ANTHROPIC });

	await emit(pi, "message_end", { type: "message_end", message: errMsg(ANTHROPIC, "429") }, ctx);

	assert.equal(pi.calls.setModel.length, 1);
	assert.equal(pi.calls.sendMessage.length, 0, "no retry triggered when switch fails");
	assert.ok(notifications.some((n) => n.msg.includes("failed to switch")));
});

/** Drive the searchable model picker headlessly. Returns [promise, component]. */
function drivePicker(models, prefill = "") {
	const theme = { fg: (_c, s) => s, bold: (s) => s };
	const tui = { requestRender: () => {} };
	let component;
	const ctx = {
		hasUI: true, mode: "tui",
		ui: {
			custom: (factory) =>
				new Promise((resolve) => {
					component = factory(tui, theme, {}, resolve);
				}),
		},
	};
	const promise = pickModelWithSearch(ctx, "test", models, prefill);
	const type = (text) => { for (const ch of text) component.handleInput(ch); };
	const render = () => component.render(100).join("\n");
	return { promise, component, type, render };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

test("model picker: type-to-filter narrows huge lists, enter selects", async () => {
	const models = [];
	for (let i = 0; i < 300; i++) models.push(mkModel("openrouter", `vendor${i}/model-${i}`));
	models.push(mkModel("openrouter", "anthropic/claude-opus-4.5"));
	const { promise, component, type, render } = drivePicker(models);
	await tick();

	type("claude opus");
	const out = render();
	assert.ok(out.includes("anthropic/claude-opus-4.5"), "fuzzy match shown");
	assert.ok(!out.includes("vendor17/"), "irrelevant models filtered out");
	assert.ok(out.includes("1/301"), "match counter reflects filtering");

	component.handleInput("\r"); // enter selects the single match
	assert.equal(await promise, "anthropic/claude-opus-4.5");
});

test("model picker: prefill is applied when it matches, cleared when it doesn't", async () => {
	const models = [mkModel("openrouter", "anthropic/claude-opus-4.5"), mkModel("openrouter", "openai/gpt-5")];
	const good = drivePicker(models, "claude opus 4 5");
	await tick();
	assert.ok(good.render().includes("> claude opus 4 5"), "prefill visible in filter line");
	assert.ok(good.render().includes("1/2"), "prefill narrows the list");
	good.component.handleInput("\x1b"); // esc
	assert.equal(await good.promise, undefined);

	const bad = drivePicker(models, "nomatchtokenshere");
	await tick();
	assert.ok(bad.render().includes("> █") || bad.render().includes(">  "), "useless prefill cleared");
	assert.ok(bad.render().includes("2/2"), "full list shown instead");
	bad.component.handleInput("\x1b");
	await bad.promise;
});

test("model picker: empty matches → enter accepts typed text as manual ID", async () => {
	const models = [mkModel("openrouter", "anthropic/claude-opus-4.5")];
	const { promise, component, type } = drivePicker(models);
	await tick();
	type("my-custom-model");
	component.handleInput("\r");
	assert.equal(await promise, "my-custom-model");
});

test("model picker: backspace edits the filter", async () => {
	const models = [mkModel("openrouter", "a/b"), mkModel("openrouter", "c/d")];
	const { promise, component, type, render } = drivePicker(models);
	await tick();
	type("zz");
	assert.ok(render().includes("0/2"));
	component.handleInput("\x7f");
	component.handleInput("\x7f");
	assert.ok(render().includes("2/2"), "filter cleared restores full list");
	component.handleInput("\x1b");
	assert.equal(await promise, undefined, "esc cancels");
});

test("suggestFilterTokens derives cross-provider filter tokens", async () => {
	assert.equal(suggestFilterTokens("claude-opus-4-5"), "claude opus 4 5");
	assert.equal(suggestFilterTokens("anthropic/claude-opus-4.5"), "claude opus 4 5");
	assert.equal(suggestFilterTokens("anthropic.claude-opus-4-5-20251101-v1:0"), "claude opus 4 5");
	assert.equal(suggestFilterTokens("openai/gpt-5.2-codex"), "gpt 5 2 codex");
	assert.equal(suggestFilterTokens("gemini-2.5-pro"), "gemini 2 5 pro");
});

// ---------------------------------------------------------------------------

let failed = 0;
for (const [name, fn] of tests) {
	try {
		await fn();
		console.log(`  ok   ${name}`);
	} catch (err) {
		failed++;
		console.error(`  FAIL ${name}`);
		console.error(err);
	}
}
console.log(failed === 0 ? `\n${tests.length} tests passed` : `\n${failed}/${tests.length} tests FAILED`);
process.exit(failed === 0 ? 0 : 1);
