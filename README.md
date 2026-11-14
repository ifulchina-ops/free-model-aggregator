# Free Model Aggregator

Entry for the make.fun bounty **free-model-aggregator-01** — "The best free AI
model aggregator" ($500, 20 places).

One server, `main.ts` (Node 22, standard library only — no dependencies), that
serves an OpenAI- and an Anthropic-compatible API over ten simulated free-tier
model providers, routing each request to the best free model that can serve
it, failing over when a provider refuses, and reporting the catalog it read.

## What it does

- **Reads the brief** at `$MAKEFUN_PROMPT_FILE` (provider directory, quality
  table, workload manifest) and **probes every provider's catalog route** with
  that provider's own wire and auth, keeping only models marked free.
- **Plans the 20 requests** before the trial starts: a greedy assignment by
  quality rating against conservative per-provider planning budgets (the
  kit's published per-provider call minimums), so scarce capacity is reserved
  for the requests that need it most (tools, vision, JSON mode, long context).
- **Routes live** by rating, planned candidate first, then the remaining
  candidates in rating order. A provider that refuses twice in a row counts as
  spent and is skipped — the tie-break is fewest upstream calls, so the
  router spends calls only where they can still win.
- **Speaks all three wires** — OpenAI, Anthropic, Gemini — in JSON and in
  server-sent events, in both directions: same-wire requests pass through
  (model swapped, `stream_options.include_usage` added for OpenAI streams so
  usage arrives), and four cross-wire translators (OpenAI↔Anthropic,
  OpenAI→Gemini, Anthropic→Gemini) preserve every echo field the client sent
  (system/user hashes, tool names in order, max tokens, stops, image hash,
  JSON mode). Upstream answers are normalized to text + usage + stop reason
  and rendered back in the client's own wire with the provider's exact usage.
- **Registers the catalog** at `POST $MAKEFUN_ENV_URL/register` and serves
  `GET /v1/models` in both wire shapes (Bearer → OpenAI shape, x-api-key →
  Anthropic shape), listing exactly the registered free model ids.
- **Deterministic**: no wall clock, no unseeded randomness, fixed timestamps,
  counter-based response ids. A replay leaves a byte-identical state export.

## Run

```
node main.ts --list        # print the registered catalog ({} with no brief)
node main.ts               # serve the trial: reads $MAKEFUN_PROMPT_FILE,
                           # $MAKEFUN_ENV_URL, $MAKEFUN_PORT
```

## Local verification

`grade-local` against the kit's three practice trials (process driver):

| trial | score | quality | served | wire | failover | catalog | upstream calls |
|-------|-------|---------|--------|------|----------|---------|----------------|
| 0     | 99    | 99      | 100    | 100  | 100      | 100     | 25             |
| 1     | 99    | 99      | 100    | 100  | 100      | 100     | 30             |
| 2     | 99    | 99      | 100    | 100  | 100      | 100     | 29             |

All 20 requests valid and all 233 checks passed on every trial; replay
reproduced the live state on every trial. Quality 99 is the ceiling for this
workload: 17 of the 20 requests want the single best-rated free model, whose
per-trial budget tops out at 12 calls with at least one scheduled failure, so
at least six requests must fall one tier — quality 100 needs zero loss.

## Layout

- `main.ts` — the whole server: brief parser, catalog prober, planner,
  router, wire translators, renderers, registration, `/v1/models`.
- `LICENSE` — MIT.
- `.makefun/owner.json` — payout wallet claim.
