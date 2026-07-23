/**
 * Auto-Failover Extension
 *
 * Automatically fails over to a backup provider when the current provider
 * fails with a transient error (rate limit, overloaded, 5xx, network/timeout)
 * and pi's built-in auto-retry has been exhausted. The failed request is
 * seamlessly retried on the backup provider; after a cooldown the extension
 * switches back to the primary provider.
 *
 * Model IDs are namespaced per provider (e.g. "claude-opus-4-5" on anthropic
 * vs "anthropic/claude-opus-4.5" on openrouter), so failover chains are
 * ordered lists of "provider/modelId" refs. Entry 0 is the primary.
 *
 * Config file: ~/.pi/agent/auto-failover.json
 * ```json
 * {
 *   "enabled": true,
 *   "cooldownSeconds": 300,
 *   "chains": [
 *     [
 *       "anthropic/claude-opus-4-5",
 *       "openrouter/anthropic/claude-opus-4.5",
 *       "amazon-bedrock/anthropic.claude-opus-4-5-20251101-v1:0"
 *     ]
 *   ]
 * }
 * ```
 *
 * Commands:
 * - `/failover`                 — status overview
 * - `/failover add`             — interactive wizard to create a chain
 * - `/failover remove [n]`      — remove chain n (interactive picker if omitted)
 * - `/failover enable|disable`  — toggle failover globally
 * - `/failover cooldown [sec]`  — show/set auto-return cooldown
 * - `/failover reset`           — immediately return to the primary provider
 *
 * Config is (re)loaded on session start, so `/reload` picks up file edits.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
	isContextOverflow,
	isRetryableAssistantError,
	type Api,
	type Model,
} from "@earendil-works/pi-ai";
import {
	DynamicBorder,
	getAgentDir,
	type ExtensionAPI,
	type ExtensionCommandContext,
	type ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import {
	Container,
	decodeKittyPrintable,
	fuzzyFilter,
	matchesKey,
	SelectList,
	Text,
	type SelectItem,
	type SelectListTheme,
} from "@earendil-works/pi-tui";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

interface ModelRef {
	provider: string;
	modelId: string;
}

interface FailoverConfig {
	enabled: boolean;
	cooldownSeconds: number;
	/** Ordered "provider/modelId" refs; entry 0 is the primary. */
	chains: string[][];
}

const DEFAULT_COOLDOWN_SECONDS = 300;

function configPath(): string {
	return join(getAgentDir(), "auto-failover.json");
}

function parseRef(ref: string): ModelRef | undefined {
	const i = ref.indexOf("/");
	if (i <= 0 || i === ref.length - 1) return undefined;
	return { provider: ref.slice(0, i), modelId: ref.slice(i + 1) };
}

function refKey(ref: ModelRef): string {
	return `${ref.provider}/${ref.modelId}`;
}

function loadConfig(): FailoverConfig {
	const fallback: FailoverConfig = { enabled: true, cooldownSeconds: DEFAULT_COOLDOWN_SECONDS, chains: [] };
	try {
		if (!existsSync(configPath())) return fallback;
		const raw = JSON.parse(readFileSync(configPath(), "utf8")) as Partial<FailoverConfig>;
		const chains = Array.isArray(raw.chains)
			? raw.chains
					.filter((c): c is string[] => Array.isArray(c))
					.map((c) => c.filter((r): r is string => typeof r === "string" && parseRef(r) !== undefined))
					.filter((c) => c.length >= 2)
			: [];
		return {
			enabled: raw.enabled !== false,
			cooldownSeconds:
				typeof raw.cooldownSeconds === "number" && raw.cooldownSeconds >= 0
					? raw.cooldownSeconds
					: DEFAULT_COOLDOWN_SECONDS,
			chains,
		};
	} catch {
		return fallback;
	}
}

function saveConfig(config: FailoverConfig): void {
	mkdirSync(dirname(configPath()), { recursive: true });
	writeFileSync(configPath(), JSON.stringify(config, null, 2) + "\n", "utf8");
}

// ---------------------------------------------------------------------------
// Runtime state
// ---------------------------------------------------------------------------

interface ActiveFailover {
	/** The chain currently in effect (parsed refs; entry 0 = primary). */
	chain: ModelRef[];
	/** refKeys already tried for the current user prompt (loop guard). */
	tried: string[];
	/** Timestamp of the last failover hop (cooldown starts here). */
	failedAt: number;
	/** Cooldown expired while busy; return to primary once idle. */
	pendingReturn: boolean;
	returnTimer?: ReturnType<typeof setTimeout>;
	statusInterval?: ReturnType<typeof setInterval>;
}

/**
 * Derive filter tokens for the model picker from a chain entry's model ID.
 * Strips vendor namespaces ("anthropic/", "anthropic."), date stamps
 * ("20251101") and version suffixes ("-v1:0") so the remaining tokens match
 * the same model on other providers: "anthropic.claude-opus-4-5-20251101-v1:0"
 * → "claude opus 4 5".
 */
export function suggestFilterTokens(modelId: string): string {
	let s = modelId.includes("/") ? modelId.slice(modelId.lastIndexOf("/") + 1) : modelId;
	s = s.replace(/^(anthropic|openai|google|meta|mistral|deepseek|qwen|xai|amazon|cohere|ai21|nvidia|microsoft)[.:]/i, "");
	s = s.replace(/\d{8}/g, " "); // date stamps like 20251101
	s = s.replace(/[-._:]v\d+(:\d+)?\b/gi, " "); // version suffixes like -v1:0
	s = s.replace(/[^a-z0-9]+/gi, " ").trim().replace(/\s+/g, " ");
	return s;
}

/** True for input that is ordinary text (incl. kitty-protocol printable keys), not control keys. */
function isPrintableInput(data: string): boolean {
	if (data.length === 0) return false;
	if (data.startsWith("\x1b")) return decodeKittyPrintable(data) !== undefined;
	for (const ch of data) {
		if (ch < " ") return false; // control characters (incl. \t, \r, ctrl+*)
	}
	return true;
}

/**
 * Searchable model picker for huge provider catalogs (openrouter has hundreds
 * of models; the plain select dialog does not scroll-follow). Type to filter
 * (fuzzy, multi-token), arrows to navigate, enter to select. If nothing
 * matches, enter accepts the typed text as a manual model ID.
 */
export async function pickModelWithSearch(
	ctx: ExtensionCommandContext,
	provider: string,
	providerModels: Model<Api>[],
	prefill: string,
): Promise<string | undefined> {
	const allItems: SelectItem[] = providerModels.map((m) => ({
		value: m.id,
		label: m.id,
		description: `${m.name} · ~${Math.round(m.contextWindow / 1000)}k ctx`,
	}));
	const itemText = (it: SelectItem) => `${it.value} ${it.description ?? ""}`;
	const filtered = (q: string) => (q.trim() ? fuzzyFilter(allItems, q.trim(), itemText) : allItems);

	// Only keep the prefill if it actually narrows to results on this provider.
	let initial = prefill.trim();
	if (initial && filtered(initial).length === 0) initial = "";

	return ctx.ui.custom<string | undefined>((tui, theme, _kb, done) => {
		let filter = initial;
		let finished = false;
		const finish = (value: string | undefined) => {
			if (!finished) {
				finished = true;
				done(value);
			}
		};

		const listTheme: SelectListTheme = {
			selectedPrefix: (t) => theme.fg("accent", t),
			selectedText: (t) => theme.fg("accent", t),
			description: (t) => theme.fg("muted", t),
			scrollInfo: (t) => theme.fg("dim", t),
			noMatch: (t) => theme.fg("warning", t),
		};

		const buildList = () => {
			const l = new SelectList(filtered(filter), 12, listTheme);
			l.onSelect = (item) => finish(item.value);
			l.onCancel = () => finish(undefined);
			return l;
		};

		const container = new Container();
		container.addChild(new DynamicBorder((s: string) => theme.fg("accent", s)));
		container.addChild(new Text(theme.fg("accent", theme.bold(`Model on ${provider} (${allItems.length} available)`))));
		const filterLine = new Text("");
		container.addChild(filterLine);
		let list = buildList();
		container.addChild(list);
		const footer = new Text("");
		container.addChild(footer);
		container.addChild(new DynamicBorder((s: string) => theme.fg("accent", s)));

		const refresh = () => {
			const count = filtered(filter).length;
			filterLine.setText(`> ${filter}█  ${theme.fg("dim", `${count}/${allItems.length}`)}`);
			footer.setText(
				theme.fg(
					"dim",
					"type to filter · ↑↓ navigate · enter select · esc cancel" +
						(count === 0 && filter.trim() ? " · enter uses typed text as model ID" : ""),
				),
			);
		};
		refresh();

		const rebuild = () => {
			const idx = container.children.indexOf(list);
			const next = buildList();
			container.children.splice(idx, 1, next);
			list = next;
		};

		return {
			render(width: number) {
				return container.render(width);
			},
			invalidate() {
				container.invalidate();
			},
			handleInput(data: string) {
				if (finished) return;
				if (matchesKey(data, "backspace")) {
					if (filter.length > 0) {
						filter = Array.from(filter).slice(0, -1).join("");
						rebuild();
						refresh();
					}
				} else if (isPrintableInput(data)) {
					filter += decodeKittyPrintable(data) ?? data;
					rebuild();
					refresh();
				} else {
					// Navigation / enter / escape (respects pi keybindings).
					list.handleInput(data);
					if (!finished && matchesKey(data, "enter") && list.getSelectedItem() === null && filter.trim()) {
						finish(filter.trim()); // manual model ID
					}
				}
				tui.requestRender();
			},
		};
	});
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI) {
	let config = loadConfig();
	let active: ActiveFailover | undefined;
	/** Set before programmatic setModel calls so model_select can ignore them. */
	let programmaticModel: Model<Api> | undefined;
	/** Latest context, used by timer callbacks. */
	let lastCtx: ExtensionContext | undefined;

	// ----- helpers -----------------------------------------------------------

	function clearTimers(state: ActiveFailover | undefined): void {
		if (!state) return;
		if (state.returnTimer) clearTimeout(state.returnTimer);
		if (state.statusInterval) clearInterval(state.statusInterval);
		state.returnTimer = undefined;
		state.statusInterval = undefined;
	}

	function resetState(): void {
		clearTimers(active);
		active = undefined;
		programmaticModel = undefined;
		if (lastCtx) lastCtx.ui.setStatus("auto-failover", undefined);
	}

	function findChain(ref: ModelRef): ModelRef[] | undefined {
		const key = refKey(ref);
		for (const chain of config.chains) {
			const refs = chain.map(parseRef).filter((r): r is ModelRef => r !== undefined);
			if (refs.some((r) => refKey(r) === key)) return refs;
		}
		return undefined;
	}

	function summarizeError(errorMessage: string | undefined): string {
		if (!errorMessage) return "unknown error";
		const firstLine = errorMessage.split("\n")[0].trim();
		return firstLine.length > 80 ? firstLine.slice(0, 77) + "…" : firstLine;
	}

	function formatRemaining(ms: number): string {
		const total = Math.max(0, Math.ceil(ms / 1000));
		return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
	}

	function updateStatus(ctx: ExtensionContext): void {
		if (!config.enabled) {
			ctx.ui.setStatus("auto-failover", "auto-failover: disabled");
			return;
		}
		if (!active) {
			ctx.ui.setStatus("auto-failover", undefined);
			return;
		}
		const currentProvider = ctx.model?.provider ?? "?";
		if (active.pendingReturn) {
			ctx.ui.setStatus(
				"auto-failover",
				`⚡failover: ${currentProvider} (returning to ${active.chain[0].provider} when idle)`,
			);
			return;
		}
		const remainingMs = config.cooldownSeconds * 1000 - (Date.now() - active.failedAt);
		ctx.ui.setStatus(
			"auto-failover",
			`⚡failover: ${currentProvider} (primary ${active.chain[0].provider} in ${formatRemaining(remainingMs)})`,
		);
	}

	// ----- programmatic model switching -------------------------------------

	async function switchModel(ctx: ExtensionContext, model: Model<Api>): Promise<boolean> {
		programmaticModel = model;
		const ok = await pi.setModel(model);
		if (!ok) programmaticModel = undefined;
		return ok;
	}

	// ----- cooldown / auto-return -------------------------------------------

	function scheduleReturn(ctx: ExtensionContext): void {
		if (!active) return;
		clearTimers(active);
		if (config.cooldownSeconds <= 0) return; // 0 = never auto-return
		active.returnTimer = setTimeout(() => {
			void onCooldownExpired();
		}, config.cooldownSeconds * 1000);
		active.returnTimer.unref?.();
		active.statusInterval = setInterval(() => {
			if (lastCtx) updateStatus(lastCtx);
		}, 15000);
		active.statusInterval.unref?.();
	}

	async function onCooldownExpired(): Promise<void> {
		const ctx = lastCtx;
		if (!active || !ctx) return;
		if (!ctx.isIdle()) {
			active.pendingReturn = true;
			updateStatus(ctx);
			return;
		}
		await returnToPrimary(ctx, "cooldown elapsed");
	}

	async function returnToPrimary(ctx: ExtensionContext, reason: string): Promise<void> {
		if (!active) return;
		const primary = active.chain[0];
		const currentKey = ctx.model ? refKey({ provider: ctx.model.provider, modelId: ctx.model.id }) : undefined;
		resetState();
		if (currentKey === refKey(primary)) return; // already on primary
		const model = ctx.modelRegistry.find(primary.provider, primary.modelId);
		if (!model || !ctx.modelRegistry.hasConfiguredAuth(model)) {
			ctx.ui.notify(
				`auto-failover: cannot return to primary ${primary.provider} (model or API key unavailable)`,
				"warning",
			);
			return;
		}
		if (await switchModel(ctx, model)) {
			ctx.ui.notify(`auto-failover: returned to primary ${primary.provider} (${reason})`, "info");
		} else {
			ctx.ui.notify(`auto-failover: failed to return to primary ${primary.provider} (no API key)`, "warning");
		}
	}

	// ----- core: failover on exhausted retries -------------------------------

	pi.on("message_end", async (event, ctx) => {
		lastCtx = ctx;

		// A genuine user message resets the per-prompt loop guard. (Our own
		// failover retry is a custom message, not a user message.)
		if (event.message.role === "user") {
			if (active && ctx.model) active.tried = [refKey({ provider: ctx.model.provider, modelId: ctx.model.id })];
			return;
		}

		if (!config.enabled) return;
		const msg = event.message;
		if (msg.role !== "assistant") return;
		if (msg.stopReason !== "error") return;
		if (isContextOverflow(msg)) return; // owned by pi's compaction recovery
		if (!isRetryableAssistantError(msg)) return; // auth/quota/billing: failing over won't help

		const failedRef: ModelRef = { provider: msg.provider, modelId: msg.model };
		const chain = findChain(failedRef);
		if (!chain || chain.length < 2) return;

		// Initialize state on first failure for this chain.
		if (!active || refKey(active.chain[0]) !== refKey(chain[0])) {
			clearTimers(active);
			active = { chain, tried: [], failedAt: Date.now(), pendingReturn: false };
		}
		const failedKey = refKey(failedRef);
		if (!active.tried.includes(failedKey)) active.tried.push(failedKey);

		// Pick the first untried chain entry with a resolvable model and auth.
		const skipped: string[] = [];
		let target: Model<Api> | undefined;
		let targetRef: ModelRef | undefined;
		for (const ref of chain) {
			const key = refKey(ref);
			if (active.tried.includes(key)) continue;
			active.tried.push(key); // mark regardless: never retry a hop within one prompt
			const model = ctx.modelRegistry.find(ref.provider, ref.modelId);
			if (!model) {
				skipped.push(`${ref.provider} (model not found)`);
				continue;
			}
			if (!ctx.modelRegistry.hasConfiguredAuth(model)) {
				skipped.push(`${ref.provider} (no API key)`);
				continue;
			}
			target = model;
			targetRef = ref;
			break;
		}

		if (!target || !targetRef) {
			ctx.ui.notify(
				`auto-failover: no usable backup left for ${failedRef.modelId}` +
					(skipped.length ? ` — skipped: ${skipped.join(", ")}` : " — all providers tried"),
				"error",
			);
			updateStatus(ctx);
			return;
		}

		if (!(await switchModel(ctx, target))) {
			ctx.ui.notify(`auto-failover: failed to switch to ${targetRef.provider} (no API key)`, "error");
			updateStatus(ctx);
			return;
		}

		active.failedAt = Date.now();
		active.pendingReturn = false;
		scheduleReturn(ctx);

		const brief = summarizeError(msg.errorMessage);
		ctx.ui.notify(`auto-failover: ${failedRef.provider} failed (${brief}) → retrying on ${targetRef.provider}`, "warning");
		updateStatus(ctx);

		// Seamless retry: hidden custom message re-triggers the turn once the
		// current (failed) run finishes. The transcript stays clean.
		pi.sendMessage(
			{
				customType: "auto-failover",
				content:
					`[auto-failover] The previous model request failed with a transient error on provider ` +
					`"${failedRef.provider}" (${brief}). It is now being retried on provider "${targetRef.provider}". ` +
					`Continue the task seamlessly; do not apologize for or comment on the failover.`,
				display: false,
				details: { from: failedRef, to: targetRef, error: msg.errorMessage },
			},
			{ triggerTurn: true, deliverAs: "followUp" },
		);
	});

	// ----- auto-return when cooldown expires mid-turn ------------------------

	pi.on("agent_settled", async (_event, ctx) => {
		lastCtx = ctx;
		if (active?.pendingReturn) {
			await returnToPrimary(ctx, "cooldown elapsed");
		}
		updateStatus(ctx);
	});

	// ----- don't fight the user ----------------------------------------------

	pi.on("model_select", (event, ctx) => {
		lastCtx = ctx;
		if (
			programmaticModel &&
			event.model.provider === programmaticModel.provider &&
			event.model.id === programmaticModel.id
		) {
			programmaticModel = undefined; // our own switch
			return;
		}
		// User (or session restore) changed the model: stop managing failover.
		if (active) resetState();
	});

	// ----- session lifecycle ---------------------------------------------------

	pi.on("session_start", (_event, ctx) => {
		lastCtx = ctx;
		resetState();
		config = loadConfig(); // picks up manual file edits and /reload
		updateStatus(ctx);
	});

	pi.on("session_shutdown", () => {
		clearTimers(active);
	});

	// ----- /failover command ----------------------------------------------------

	function statusLines(ctx: ExtensionCommandContext): string[] {
		const lines: string[] = [];
		lines.push(`enabled: ${config.enabled}   cooldown: ${config.cooldownSeconds}s   config: ${configPath()}`);
		if (active && ctx.model) {
			const remainingMs = config.cooldownSeconds * 1000 - (Date.now() - active.failedAt);
			lines.push(
				`state: FAILED OVER — on ${ctx.model.provider}, primary ${active.chain[0].provider} ` +
					(active.pendingReturn ? "(return pending idle)" : `(return in ${formatRemaining(remainingMs)})`),
			);
		} else {
			lines.push("state: on primary (no active failover)");
		}
		if (config.chains.length === 0) {
			lines.push("no chains configured — run /failover add");
		} else {
			config.chains.forEach((chain, i) => {
				lines.push(`chain ${i + 1}: ${chain.join("  →  ")}`);
			});
		}
		return lines;
	}

	async function wizardAdd(ctx: ExtensionCommandContext): Promise<void> {
		if (!ctx.hasUI) {
			ctx.ui.notify(`auto-failover: no interactive UI — edit ${configPath()} manually`, "warning");
			return;
		}
		const models = ctx.modelRegistry.getAll();
		if (models.length === 0) {
			ctx.ui.notify("auto-failover: no models in registry", "error");
			return;
		}
		const availableProviders = new Set(ctx.modelRegistry.getAvailable().map((m) => m.provider));

		const pickRef = async (title: string, exclude: Set<string>, prefill = ""): Promise<ModelRef | undefined> => {
			const providers = [...new Set(models.map((m) => m.provider))].sort();
			const providerChoice = await ctx.ui.select(
				title,
				providers.map((p) => (availableProviders.has(p) ? p : `${p}  (no auth configured)`)),
			);
			if (!providerChoice) return undefined;
			const provider = providerChoice.replace(/ {2}\(no auth configured\)$/, "");
			if (exclude.has(provider)) {
				ctx.ui.notify(`auto-failover: ${provider} is already in this chain`, "warning");
				return undefined;
			}
			const providerModels = models
				.filter((m) => m.provider === provider)
				.sort((a, b) => a.id.localeCompare(b.id));
			let modelId: string | undefined;
			if (ctx.mode === "tui") {
				// Searchable picker — provider catalogs can be huge (openrouter).
				modelId = await pickModelWithSearch(ctx, provider, providerModels, prefill);
			} else {
				const input = await ctx.ui.input(
					`Model ID on ${provider} (${providerModels.length} available)`,
					prefill || "e.g. claude-opus-4-5",
				);
				modelId = input?.trim() || undefined;
			}
			if (!modelId) return undefined;
			return { provider, modelId };
		};

		const chain: ModelRef[] = [];
		const usedProviders = new Set<string>();
		const primary = await pickRef("Failover chain — PRIMARY provider/model (step 1)", usedProviders);
		if (!primary) return;
		chain.push(primary);
		usedProviders.add(primary.provider);

		while (chain.length < 8) {
			const next = await pickRef(
				`Failover chain — backup #${chain.length} (esc to finish)`,
				usedProviders,
				suggestFilterTokens(chain[chain.length - 1].modelId),
			);
			if (!next) break;
			chain.push(next);
			usedProviders.add(next.provider);
			if (!(await ctx.ui.confirm("Failover chain", "Add another backup provider?"))) break;
		}

		if (chain.length < 2) {
			ctx.ui.notify("auto-failover: a chain needs at least a primary and one backup — not saved", "warning");
			return;
		}

		// Remove any existing chain that starts with the same primary.
		const primaryKey = refKey(chain[0]);
		config.chains = config.chains.filter((c) => c[0] !== primaryKey);
		config.chains.push(chain.map(refKey));
		saveConfig(config);
		ctx.ui.notify(`auto-failover: chain saved — ${chain.map(refKey).join(" → ")}`, "info");
	}

	async function removeChain(ctx: ExtensionCommandContext, arg: string | undefined): Promise<void> {
		if (config.chains.length === 0) {
			ctx.ui.notify("auto-failover: no chains configured", "info");
			return;
		}
		let index: number | undefined;
		if (arg) {
			const n = Number.parseInt(arg, 10);
			if (Number.isNaN(n) || n < 1 || n > config.chains.length) {
				ctx.ui.notify(`auto-failover: invalid chain number "${arg}" (1–${config.chains.length})`, "error");
				return;
			}
			index = n - 1;
		} else if (ctx.hasUI) {
			const choice = await ctx.ui.select(
				"Remove which failover chain?",
				config.chains.map((c, i) => `chain ${i + 1}: ${c.join(" → ")}`),
			);
			if (!choice) return;
			index = Number.parseInt(choice.split(":")[0].replace("chain ", ""), 10) - 1;
		} else {
			ctx.ui.notify("auto-failover: usage /failover remove <n>", "warning");
			return;
		}
		const [removed] = config.chains.splice(index, 1);
		saveConfig(config);
		ctx.ui.notify(`auto-failover: removed chain — ${removed.join(" → ")}`, "info");
	}

	pi.registerCommand("failover", {
		description: "Manage auto-failover provider chains",
		getArgumentCompletions: (prefix) => {
			const subs = ["status", "add", "remove", "enable", "disable", "cooldown", "reset", "help"];
			return subs
				.filter((s) => s.startsWith(prefix))
				.map((s) => ({ value: s, label: s }));
		},
		handler: async (args, ctx) => {
			lastCtx = ctx as unknown as ExtensionContext;
			const [sub, ...rest] = args.trim().split(/\s+/).filter(Boolean);
			switch (sub ?? "status") {
				case "status": {
					const lines = statusLines(ctx);
					if (ctx.hasUI && lines.length > 3) {
						await ctx.ui.select("auto-failover status (esc to close)", [...lines, "(close)"]);
					} else {
						ctx.ui.notify(lines.join("\n"), "info");
					}
					break;
				}
				case "add":
					await wizardAdd(ctx);
					break;
				case "remove":
					await removeChain(ctx, rest[0]);
					break;
				case "enable":
					config.enabled = true;
					saveConfig(config);
					updateStatusCtx(ctx);
					ctx.ui.notify("auto-failover: enabled", "info");
					break;
				case "disable":
					config.enabled = false;
					resetState();
					updateStatusCtx(ctx);
					ctx.ui.notify("auto-failover: disabled", "info");
					break;
				case "cooldown": {
					const val = rest[0] ? Number.parseInt(rest[0], 10) : Number.NaN;
					if (rest[0] === undefined) {
						ctx.ui.notify(`auto-failover: cooldown is ${config.cooldownSeconds}s (0 = never auto-return)`, "info");
					} else if (Number.isNaN(val) || val < 0) {
						ctx.ui.notify(`auto-failover: invalid cooldown "${rest[0]}"`, "error");
					} else {
						config.cooldownSeconds = val;
						saveConfig(config);
						if (active) scheduleReturnCtx(ctx);
						ctx.ui.notify(
							val === 0
								? "auto-failover: auto-return disabled (use /failover reset to return manually)"
								: `auto-failover: cooldown set to ${val}s`,
							"info",
						);
					}
					break;
				}
				case "reset":
					if (!active) {
						ctx.ui.notify("auto-failover: not currently failed over", "info");
					} else {
						await returnToPrimaryCtx(ctx, "manual reset");
					}
					break;
				case "help":
					ctx.ui.notify(
						"/failover [status] · add · remove [n] · enable · disable · cooldown [sec] · reset",
						"info",
					);
					break;
				default:
					ctx.ui.notify(`auto-failover: unknown subcommand "${sub}" — try /failover help`, "warning");
			}
		},
	});

	// Thin wrappers so command handlers can reuse the ExtensionContext-typed helpers.
	function updateStatusCtx(ctx: ExtensionCommandContext): void {
		updateStatus(ctx as unknown as ExtensionContext);
	}
	function scheduleReturnCtx(ctx: ExtensionCommandContext): void {
		scheduleReturn(ctx as unknown as ExtensionContext);
	}
	async function returnToPrimaryCtx(ctx: ExtensionCommandContext, reason: string): Promise<void> {
		await returnToPrimary(ctx as unknown as ExtensionContext, reason);
	}
}
