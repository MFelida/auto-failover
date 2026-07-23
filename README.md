# pi-auto-failover

Automatic provider failover for [pi](https://github.com/earendil-works/pi).
When the active model's provider fails with a **transient error** (rate limit,
overloaded, 5xx, network/timeout) and pi's built-in auto-retry has been
exhausted, this extension switches to the next provider in your configured
chain and seamlessly retries the request. After a cooldown it switches back to
the primary provider.

## How it works

1. Pi retries transient errors on the same provider with backoff (built-in).
2. If the request still fails, the final assistant message carries
   `stopReason: "error"`. The extension classifies it with pi-ai's own
   `isRetryableAssistantError()` — so failover triggers on exactly the errors
   pi considers transient. Auth failures, quota/billing errors, aborts and
   context-overflow are deliberately excluded (failing over wouldn't help, or
   is owned by pi's compaction recovery).
3. The extension switches the model to the next chain entry
   (`pi.setModel`) and re-triggers the turn with a hidden custom message —
   the conversation continues without any user action.
4. A status bar entry shows the failover state and countdown. After
   `cooldownSeconds` (and once the agent is idle), it switches back to the
   primary provider.

Guardrails:

- Each provider is tried at most once per user prompt (no failover loops).
- Providers without a configured API key are skipped automatically.
- If you change the model manually, the extension steps aside and cancels
  the pending auto-return.
- Model IDs are namespaced per provider, so chains map each provider to its
  own ID for the same underlying model.

## Install

```bash
# Option A: global, all projects
mkdir -p ~/.pi/agent/extensions
ln -s "$(pwd)/index.ts" ~/.pi/agent/extensions/auto-failover.ts

# Option B: single run
pi -e /path/to/auto-failover/index.ts
```

## Configure

Run `/failover add` inside pi — a wizard walks you through picking the
primary provider/model and one or more backups, then persists the chain to
`~/.pi/agent/auto-failover.json`.

The model picker is **searchable** (type to fuzzy-filter, arrows to navigate),
which keeps even the huge openrouter catalog manageable. When adding a backup,
the filter is pre-filled with tokens derived from the previous chain entry
(e.g. `claude opus 4 5`), so the equivalent model usually surfaces
immediately. If nothing matches, pressing enter accepts the typed text as a
manual model ID:

```json
{
  "enabled": true,
  "cooldownSeconds": 300,
  "chains": [
    [
      "anthropic/claude-opus-4-5",
      "openrouter/anthropic/claude-opus-4.5",
      "amazon-bedrock/anthropic.claude-opus-4-5-20251101-v1:0"
    ]
  ]
}
```

- `enabled` — master switch (also `/failover enable|disable`).
- `cooldownSeconds` — delay before auto-returning to the primary.
  `0` disables auto-return (use `/failover reset` manually).
- `chains` — ordered lists of `provider/modelId` refs. Entry 0 is the
  primary. Multiple independent chains are supported.

The file is re-read on session start, so `/reload` (or a new session) picks
up manual edits.

## Commands

| Command | Description |
| --- | --- |
| `/failover` | Status: state, cooldown, configured chains |
| `/failover add` | Interactive chain wizard |
| `/failover remove [n]` | Remove chain `n` (interactive picker if omitted) |
| `/failover enable` / `disable` | Toggle failover globally |
| `/failover cooldown [sec]` | Show or set the auto-return cooldown |
| `/failover reset` | Immediately return to the primary provider |

## Development

```bash
# Type check (requires node_modules symlinks or npm install of the peer deps)
npx -p typescript@5.5 tsc --noEmit

# Test harness (plain Node 24+, no dependencies)
node test/run.mjs

# Smoke-test that pi loads the extension
pi -e ./index.ts --mode json "hi"
```

`node_modules` here contains symlinks to the globally installed
`@earendil-works/pi-coding-agent`, `@earendil-works/pi-ai` and `@types/node`
so `tsc` can resolve types. At runtime pi provides these modules to the
extension automatically.
