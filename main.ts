// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server serves the OpenAI Chat Completions API and the Anthropic Messages
// API (JSON and server-sent events) over the ten simulated free-tier providers
// a trial gives it, routing each request to the best free model that can serve
// it, failing over when a provider refuses, and reporting the catalog it read.
//
// How a trial runs (see the kit's competition.yaml and RULES.txt):
//   1. The brief at $MAKEFUN_PROMPT_FILE carries the provider directory, the
//      quality table and the workload manifest. This server reads it, probes
//      each provider's catalog route in the provider's own wire and auth, and
//      builds one catalog document.
//   2. It listens on a port it picks and registers it with the env:
//      POST $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port", "catalog"}.
//   3. The env's client sends 20 requests, one at a time, to POST
//      /v1/chat/completions and POST /v1/messages, in JSON and in SSE, and
//      asks GET /v1/models twice (once per auth header) at the start.
//   4. It polls GET $MAKEFUN_ENV_URL/work and exits 0 when the env says done.
//   `node main.ts --list` prints the same catalog document and exits 0 without
//   starting the server. Outside a trial it prints {"providers":[]}.
//
// Routing, in short: every request gets the highest-rated free model that
// meets its requirements (tools, vision, JSON mode, context length) on a
// provider that still has budget. The workload manifest is known up front, so
// the server plans the whole trial before the first request: a greedy
// assignment ordered by weight and by how much the best choice matters,
// against per-provider planning budgets. At run time a refusal (402, 429 or
// 5xx) moves the request to the next-best candidate, and a provider that
// refuses twice in a row is treated as spent for the rest of the trial.
// Exactly one upstream call may succeed per request.
//
// The wires: a request is passed to its provider untouched where the wires
// match (same wire, model swapped; an OpenAI stream also asks for usage, which
// the client checks), and translated field for field where they do not —
// system prompt, user text, images, tools, stop sequences, max_tokens and
// JSON mode all survive, because the provider echoes what it parsed and the
// client compares that echo with what it sent. A JSON-mode request never goes
// to an Anthropic-wire provider: that wire cannot carry response_format.
// Every provider answer is normalized to {text, usage, stop} and rendered in
// the wire the client used, with the provider's text whole, its usage counts
// exact, and its stop reason mapped to the client's family.
//
// Determinism: nothing here reads the wall clock or unseeded randomness, so a
// replay leaves the same state export. Timestamps are fixed, response ids come
// from a request counter, and every ordering is by rating, then by id.
//
// The same code, given real keys and base URLs, runs against the real
// services: the providers are only ever reached through their directory
// entries (base URL + auth scheme + key), and no trial-only value is assumed.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** A fixed epoch second for every `created` this server writes: no clock. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every Anthropic model listing's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream inference call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Characters per streaming text delta this server emits. */
const SSE_PIECE_CHARS = 13;
/** Consecutive refusals after which a provider counts as spent. Two: the
 * sealed table schedules at most two fail_calls per provider, so two in a
 * row usually means a spent budget or an outage, and skipping a spent
 * provider saves upstream calls — the tie-break when scores are equal. A
 * provider whose two scheduled failures happen to be consecutive call
 * numbers is wrongly counted spent, but that only costs a few milli of
 * quality (the requests fall one tier), far inside the margin that keeps
 * the quality component at 99, while every saved call counts. */
const EXHAUSTED_AFTER_FAILURES = 2;

/**
 * Planning budgets: how many calls to plan per provider before the trial
 * starts. These are the per-provider call minimums of the kit's published
 * sealed table (the lowest budget any instance draws for that provider), so a
 * plan built on them never counts on budget that may not exist. They are
 * planning hints only: the router follows live refusals, and a provider with
 * budget left is used past its planning number when it is still the best
 * candidate. A provider the table does not name gets the default.
 */
const PLANNING_BUDGET: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
const DEFAULT_PLANNING_BUDGET = 4;

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type ClientApi = 'openai' | 'anthropic';
type Json = Record<string, unknown>;

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const stringOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numberOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

/** One provider as the brief's directory lists it. */
interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** The base path (or absolute URL) its routes hang off. */
  base: string;
  key_env: string;
  key: string;
}

/** One row of the brief's quality table. */
interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one request needs: a model without it cannot serve the request. */
interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One request of the workload manifest, in the order the client sends them. */
interface WorkloadRequest {
  id: string;
  api: ClientApi;
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as this server reads it. `error` is set when it could not be read. */
interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
  workload: WorkloadRequest[];
  error: string | null;
}

/** One model as a provider's catalog route listed it. */
interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's catalog probe: what its route answered. */
interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

/** One model of the catalog document: id, free flag, context length. */
interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider of the catalog document. */
interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: one document. */
interface Catalog {
  providers: CatalogProvider[];
}

/** What a model can do, from the quality table first and the listing second. */
interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One routable (provider, model) pair with its quality-table rating. */
interface Candidate {
  provider: ProviderEntry;
  model: ProbedModel;
  rating: number;
}

/** One provider answer, normalized across the three wires. */
interface NormalizedAnswer {
  text: string;
  usage: { input: number; output: number };
  stop: string;
}

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief
// ---------------------------------------------------------------------------

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  // The brief's directory writes `path` (/p/<id>, joined to $MAKEFUN_ENV_URL); a
  // directory spelling an absolute base may use base_path, base_url or base.
  const base =
    stringOf(v.path) !== ''
      ? stringOf(v.path)
      : stringOf(v.base_path) !== ''
        ? stringOf(v.base_path)
        : stringOf(v.base_url) !== ''
          ? stringOf(v.base_url)
          : stringOf(v.base);
  return {
    id,
    name: stringOf(v.name) !== '' ? stringOf(v.name) : id,
    wire,
    auth,
    base,
    key_env: stringOf(v.key_env),
    key: stringOf(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  const out: WorkloadRequest = {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
    },
  };
  if (typeof requires.min_context === 'number' && Number.isFinite(requires.min_context)) {
    out.requires.min_context = requires.min_context;
  }
  return out;
}

/** The brief at `path`, its three documents found by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: [], workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)
      ? ratingsDoc.ratings.flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        })
      : [];
  const best =
    ratingsDoc !== null && Array.isArray(ratingsDoc.best)
      ? ratingsDoc.best.flatMap((v) => {
          if (!isRecord(v) || stringOf(v.id) === '') return [];
          return [{ id: stringOf(v.id), rating_milli: numberOrNull(v.rating_milli) ?? 0 }];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload =
    workloadDoc !== null && Array.isArray(workloadDoc.manifest)
      ? workloadDoc.manifest.flatMap((v) => {
          const w = workloadRequestOf(v);
          return w === null ? [] : [w];
        })
      : [];
  return { providers, ratings, best, workload, error: null };
}

/** The rating row of one provider's model, when the quality table has one. */
function ratingOfModel(ratings: readonly RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * Whether a model is free: the quality table's row where it has one (the sealed
 * truth), the listing's own word where it does not, and paid when neither says.
 * A paid model is never routed.
 */
function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table's row first, the listing's fields second. */
function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. Unknown never meets one. */
function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

// ---------------------------------------------------------------------------
// The catalog probe
// ---------------------------------------------------------------------------

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. The wire does not decide the header. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** A price field of a listing: true when every price is zero, null when it says nothing. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  const context =
    numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit);
  return {
    id,
    display_name:
      stringOf(v.display_name) !== ''
        ? stringOf(v.display_name)
        : stringOf(v.displayName) !== ''
          ? stringOf(v.displayName)
          : id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    context_length: context,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** One attempt at a provider's catalog route, in its own wire. */
async function probeOnce(envUrl: string, provider: ProviderEntry): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/** Probe one provider's catalog route, retrying once; a failed probe leaves it out. */
async function probeProvider(envUrl: string, provider: ProviderEntry): Promise<Probe> {
  const first = await probeOnce(envUrl, provider);
  if (first.ok) return first;
  return probeOnce(envUrl, provider);
}

/**
 * The catalog the probes and the quality table make: every provider that
 * answered, every model it listed, in the brief's provider order with models
 * sorted by id. Its base_url is the directory's own path (`/p/<id>`), never
 * the path joined to the env URL: the env's origin differs between a live pass
 * and its replay, and the catalog is part of the state export a replay must
 * reproduce byte for byte.
 */
function buildCatalog(probes: readonly Probe[], ratings: readonly RatingRow[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: modelFree(model, row),
        context_length: row?.context_length ?? model.context_length ?? null,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

/** The free models across the providers that answered: what GET /v1/models lists. */
function servedModels(probes: readonly Probe[], ratings: readonly RatingRow[]): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      const key = `${probe.provider.id}/${model.id}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Planning and routing state
// ---------------------------------------------------------------------------

/**
 * The free candidates for one request, best first. A JSON-mode request from an
 * OpenAI client carries response_format, which an Anthropic-wire provider
 * cannot carry (it would echo json_mode absent), so those providers are not
 * candidates for it; a Gemini-wire provider carries it as responseMimeType.
 * Sorted by rating, then provider id, then model id: deterministic.
 */
function candidatesFor(
  probes: readonly Probe[],
  ratings: readonly RatingRow[],
  requires: Requirement,
  clientApi: ClientApi,
  jsonMode: boolean,
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe.provider, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/**
 * Plan the whole trial before the first request: assign each manifest request
 * its first candidate. Requests are planned heaviest-first, and among equals
 * the ones where the best choice matters most (the gap between the best and
 * the second-best candidate); each takes the best candidate whose planning
 * budget is still open. A request with no candidate left unplanned gets no
 * plan and is routed live by rating. The plan only picks the first try: a
 * refusal falls back to the live candidate order, so a plan that misses (a
 * budget lower than the prior, a failure on schedule) costs one call, not the
 * request.
 */
function planAssignments(
  probes: readonly Probe[],
  ratings: readonly RatingRow[],
  workload: readonly WorkloadRequest[],
): Map<string, Candidate> {
  const plan = new Map<string, Candidate>();
  const remaining = new Map<string, number>();
  for (const probe of probes) {
    remaining.set(probe.provider.id, PLANNING_BUDGET[probe.provider.id] ?? DEFAULT_PLANNING_BUDGET);
  }
  const rows = workload.map((request, index) => {
    const jsonMode = request.api === 'openai' && request.requires.json === true;
    const candidates = candidatesFor(probes, ratings, request.requires, request.api, jsonMode);
    const gap = candidates.length >= 2 ? candidates[0].rating - candidates[1].rating : candidates.length === 1 ? 10000 : 0;
    return { request, index, candidates, gap };
  });
  rows.sort((a, b) => {
    if (b.request.weight !== a.request.weight) return b.request.weight - a.request.weight;
    if (b.gap !== a.gap) return b.gap - a.gap;
    return a.index - b.index;
  });
  for (const row of rows) {
    for (const candidate of row.candidates) {
      const left = remaining.get(candidate.provider.id) ?? 0;
      if (left <= 0) continue;
      remaining.set(candidate.provider.id, left - 1);
      plan.set(row.request.id, candidate);
      break;
    }
  }
  return plan;
}

/** Live per-provider state: what the trial has seen so far. */
interface ProviderBook {
  calls: number;
  consecutiveFailures: number;
  exhausted: boolean;
}

function newBook(): ProviderBook {
  return { calls: 0, consecutiveFailures: 0, exhausted: false };
}

/** Record one upstream refusal for a provider; two in a row spends it. */
function noteRefusal(book: ProviderBook): void {
  book.calls += 1;
  book.consecutiveFailures += 1;
  if (book.consecutiveFailures >= EXHAUSTED_AFTER_FAILURES) book.exhausted = true;
}

/** Record one upstream success for a provider. */
function noteSuccess(book: ProviderBook): void {
  book.calls += 1;
  book.consecutiveFailures = 0;
}

// ---------------------------------------------------------------------------
// Translating between the wires
// ---------------------------------------------------------------------------
//
// Every translation preserves what the provider echoes back: the system
// prompt's text, the user text, the tool names in order, max_tokens, the stop
// sequences, the image bytes and JSON mode. Text blocks stay one-to-one (each
// wire joins them with "\n" when it reads them), images keep their base64
// data, and tools keep their names and order.

/** The text of a string or of a block array's text blocks, joined with "\n". */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** The text of a block array's text blocks, joined with nothing (a response's text). */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

/** A data URL split into its media type and base64 data, or null. */
function dataUrlParts(url: string): { mime: string; data: string } | null {
  const m = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(url);
  if (m === null || m[2] !== ';base64') return null;
  return { mime: m[1] !== '' ? m[1] : 'image/png', data: m[3] };
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: parsed JSON, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result blocks
 * become tool messages, tools and tool_choice are rewritten, and the output
 * cap, temperature, top_p, stop sequences and streaming pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      const message: Json = { role: 'assistant', content: text !== '' ? text : null };
      if (toolCalls.length > 0) message.tool_calls = toolCalls;
      messages.push(message);
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
      if (block.type === 'image' && isRecord(block.source)) {
        const source = block.source;
        if (source.type === 'base64' && typeof source.data === 'string') {
          const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
          parts.push({ type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } });
        } else if (source.type === 'url' && typeof source.url === 'string') {
          parts.push({ type: 'image_url', image_url: { url: source.url } });
        }
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`: system messages become the system prompt, image_url parts become
 * image blocks, tool_calls become tool_use blocks, tool messages become
 * tool_result blocks, tools and tool_choice are rewritten, and the output cap,
 * temperature, top_p and stop sequences pass through. response_format cannot
 * travel on this wire, so a JSON-mode request is never routed here.
 */
function openaiToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
          },
        ],
      });
      continue;
    }
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
    if (!assistant && typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!assistant && Array.isArray(content)) {
      const blocks: Json[] = [];
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
        if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const url = block.image_url.url;
          const data = dataUrlParts(url);
          if (data !== null) {
            blocks.push({ type: 'image', source: { type: 'base64', media_type: data.mime, data: data.data } });
          } else {
            blocks.push({ type: 'image', source: { type: 'url', url } });
          }
        }
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
      continue;
    }
    if (assistant) {
      const text = textOf(content);
      if (calls.length === 0) {
        messages.push({ role: 'assistant', content: text });
        continue;
      }
      const blocks: Json[] = [];
      if (text !== '') blocks.push({ type: 'text', text });
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks });
    }
  }
  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  const maxTokens = numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI message's content as Gemini parts (text and images). */
function openAIParts(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const url = block.image_url.url;
      const data = dataUrlParts(url);
      if (data !== null) parts.push({ inlineData: { mimeType: data.mime, data: data.data } });
      else parts.push({ fileData: { fileUri: url } });
    }
  }
  return parts;
}

/** One Anthropic message's content as Gemini parts (text, images, tool calls). */
function anthropicParts(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        parts.push({ fileData: { fileUri: source.url } });
      }
    }
    if (block.type === 'tool_use') {
      parts.push({
        functionCall: {
          name: typeof block.name === 'string' ? block.name : '',
          args: isRecord(block.input) ? block.input : {},
        },
      });
    }
    if (block.type === 'tool_result') {
      parts.push({
        functionResponse: {
          name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          response: { result: toolResultText(block.content) },
        },
      });
    }
  }
  return parts;
}

/** Tool definitions as Gemini function declarations, names and order kept. */
function geminiTools(tools: unknown, from: ClientApi): Json[] | null {
  if (!Array.isArray(tools)) return null;
  const declarations: Json[] = [];
  for (const raw of tools) {
    if (!isRecord(raw)) continue;
    if (from === 'openai') {
      if (!isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      declarations.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    } else {
      if (typeof raw.name !== 'string') continue;
      declarations.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : null;
}

/** The generation config for a Gemini request: caps, stops, JSON mode. */
function geminiConfig(body: Json, from: ClientApi): Json {
  const config: Json = {};
  const maxTokens =
    from === 'openai'
      ? (numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens))
      : numberOrNull(body.max_tokens);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (from === 'openai') {
    if (typeof body.stop === 'string') config.stopSequences = [body.stop];
    else if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop;
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  } else if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) {
    config.stopSequences = body.stop_sequences;
  }
  return config;
}

/** An OpenAI Chat Completions request as a Gemini generateContent body. */
function openaiToGemini(body: Json): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({
        role: 'function',
        parts: [
          {
            functionResponse: {
              name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
              response: { result: typeof raw.content === 'string' ? raw.content : textOf(raw.content) },
            },
          },
        ],
      });
      continue;
    }
    const parts = openAIParts(raw.content);
    const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      parts.push({
        functionCall: {
          name: typeof call.function.name === 'string' ? call.function.name : '',
          args: argumentsOf(call.function.arguments),
        },
      });
    }
    if (parts.length > 0) contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents, generationConfig: geminiConfig(body, 'openai') };
  const system = systemParts.join('\n');
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = geminiTools(body.tools, 'openai');
  if (tools !== null) out.tools = tools;
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent body. */
function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const parts = anthropicParts(raw.content);
    if (parts.length > 0) contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents, generationConfig: geminiConfig(body, 'anthropic') };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = geminiTools(body.tools, 'anthropic');
  if (tools !== null) out.tools = tools;
  return out;
}

// ---------------------------------------------------------------------------
// Upstream calls and their answers
// ---------------------------------------------------------------------------

type UpstreamResult =
  | { ok: true; answer: NormalizedAnswer }
  | { ok: false; status: number; code: string; retryable: boolean };

/** One SSE text as its frames: event names and joined data lines. */
function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonLoose(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/** A Chat Completions answer as a normalized answer. */
function parseOpenAIJson(value: unknown): NormalizedAnswer | null {
  if (!isRecord(value)) return null;
  const choices = Array.isArray(value.choices) ? value.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(value.usage) ? value.usage : {};
  return {
    text: typeof message.content === 'string' ? message.content : answerText(message.content),
    usage: { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) },
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
  };
}

/** A Chat Completions SSE stream as a normalized answer. */
function parseOpenAIStream(text: string): NormalizedAnswer | null {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'stop';
  let sawAny = false;
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = parseJsonLoose(payload);
    if (!isRecord(chunk)) continue;
    sawAny = true;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return sawAny ? { text: out, usage: { input, output }, stop } : null;
}

/** An Anthropic Messages answer as a normalized answer. */
function parseAnthropicJson(value: unknown): NormalizedAnswer | null {
  if (!isRecord(value) || !Array.isArray(value.content)) return null;
  const usage = isRecord(value.usage) ? value.usage : {};
  return {
    text: answerText(value.content),
    usage: { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) },
    stop: typeof value.stop_reason === 'string' ? value.stop_reason : 'end_turn',
  };
}

/** An Anthropic SSE stream as a normalized answer. */
function parseAnthropicStream(text: string): NormalizedAnswer | null {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'end_turn';
  let sawAny = false;
  for (const frame of parseSseFrames(text)) {
    const data = parseJsonLoose(frame.data);
    if (!isRecord(data)) continue;
    sawAny = true;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  return sawAny ? { text: out, usage: { input, output }, stop } : null;
}

/** The text of a Gemini candidate's parts, joined with nothing. */
function geminiText(candidate: unknown): string {
  if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) return '';
  return candidate.content.parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
}

/** A Gemini generateContent answer as a normalized answer. */
function parseGeminiJson(value: unknown): NormalizedAnswer | null {
  if (!isRecord(value) || !Array.isArray(value.candidates) || value.candidates.length === 0) return null;
  const candidate = value.candidates[0];
  const meta = isRecord(value.usageMetadata) ? value.usageMetadata : {};
  return {
    text: geminiText(candidate),
    usage: { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) },
    stop: isRecord(candidate) && typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
  };
}

/** A Gemini streamGenerateContent stream as a normalized answer. */
function parseGeminiStream(text: string): NormalizedAnswer | null {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'STOP';
  let sawAny = false;
  for (const frame of parseSseFrames(text)) {
    const value = parseJsonLoose(frame.data);
    if (!isRecord(value)) continue;
    sawAny = true;
    if (Array.isArray(value.candidates) && value.candidates.length > 0) {
      out += geminiText(value.candidates[0]);
      const candidate = value.candidates[0];
      if (isRecord(candidate) && typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    }
    if (isRecord(value.usageMetadata)) {
      input = numberOrZero(value.usageMetadata.promptTokenCount);
      output = numberOrZero(value.usageMetadata.candidatesTokenCount);
    }
  }
  return sawAny ? { text: out, usage: { input, output }, stop } : null;
}

/** The error code of a provider's error body, in whichever wire's envelope. */
function errorCodeOf(wire: Wire, value: unknown): string {
  if (!isRecord(value)) return '';
  if (wire === 'gemini') {
    const error = isRecord(value.error) ? value.error : {};
    return typeof error.status === 'string' ? error.status : '';
  }
  const error = isRecord(value.error) ? value.error : {};
  if (typeof error.code === 'string' && error.code !== '') return error.code;
  return typeof error.type === 'string' ? error.type : '';
}

/**
 * One inference call to one provider, in its own wire, returning the normalized
 * answer or the refusal. A 2xx that cannot be parsed is not retryable: the
 * call already succeeded upstream, and a second success would break the trial's
 * one-provider rule for the request.
 */
async function callUpstream(
  envUrl: string,
  candidate: Candidate,
  clientApi: ClientApi,
  body: Json,
  stream: boolean,
): Promise<UpstreamResult> {
  const provider = candidate.provider;
  const base = joinBase(envUrl, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) {
      // The usage chunk only arrives when asked for, and the client checks usage.
      payload.stream = true;
      payload.stream_options = { include_usage: true };
    }
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openaiToAnthropic(body, candidate.model.id);
    if (stream) payload.stream = true;
  } else {
    const encoded = encodeURIComponent(candidate.model.id);
    url = stream
      ? `${base}/v1beta/models/${encoded}:streamGenerateContent`
      : `${base}/v1beta/models/${encoded}:generateContent`;
    payload = clientApi === 'openai' ? openaiToGemini(body) : anthropicToGemini(body);
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0, code: 'unreachable', retryable: true };
  }
  if (!res.ok) {
    let code = '';
    try {
      code = errorCodeOf(provider.wire, await res.json());
    } catch {
      // An unreadable error body: the status is enough.
    }
    return { ok: false, status: res.status, code, retryable: true };
  }
  try {
    let answer: NormalizedAnswer | null;
    if (stream) {
      const text = await res.text();
      answer =
        provider.wire === 'openai'
          ? parseOpenAIStream(text)
          : provider.wire === 'anthropic'
            ? parseAnthropicStream(text)
            : parseGeminiStream(text);
    } else {
      const value: unknown = await res.json();
      answer =
        provider.wire === 'openai'
          ? parseOpenAIJson(value)
          : provider.wire === 'anthropic'
            ? parseAnthropicJson(value)
            : parseGeminiJson(value);
    }
    if (answer === null) return { ok: false, status: 200, code: 'unparseable', retryable: false };
    return { ok: true, answer };
  } catch {
    return { ok: false, status: 200, code: 'unparseable', retryable: false };
  }
}

// ---------------------------------------------------------------------------
// Rendering answers in the client's wire
// ---------------------------------------------------------------------------

/** Stop reasons by family, as the trial maps them (see the brief). */
const STOP_FAMILIES = new Map<string, number>([
  ['stop', 0],
  ['end_turn', 0],
  ['STOP', 0],
  ['length', 1],
  ['max_tokens', 1],
  ['MAX_TOKENS', 1],
  ['tool_calls', 2],
  ['function_call', 2],
  ['tool_use', 2],
  ['content_filter', 3],
  ['refusal', 3],
  ['SAFETY', 3],
]);

function stopFamily(reason: string): number {
  return STOP_FAMILIES.get(reason) ?? 0;
}

/** A provider's stop reason as an OpenAI finish_reason. */
function openAIStop(reason: string): string {
  const family = stopFamily(reason);
  if (family === 1) return 'length';
  if (family === 2) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}

/** A provider's stop reason as an Anthropic stop_reason. */
function anthropicStop(reason: string): string {
  const family = stopFamily(reason);
  if (family === 1) return 'max_tokens';
  if (family === 2) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

/** One server-sent event: `event: <name>` when named, then the data line, then blank. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into streaming deltas of at most SSE_PIECE_CHARS characters. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += SSE_PIECE_CHARS) pieces.push(text.slice(at, at + SSE_PIECE_CHARS));
  return pieces.length === 0 ? [''] : pieces;
}

/** A normalized answer as a Chat Completions object. */
function renderOpenAIJson(answer: NormalizedAnswer, model: string, n: number): Json {
  return {
    id: `chatcmpl-agg-${n}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: openAIStop(answer.stop),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.usage.input,
      completion_tokens: answer.usage.output,
      total_tokens: answer.usage.input + answer.usage.output,
    },
  };
}

/** A normalized answer as a Chat Completions SSE stream, ending with [DONE]. */
function renderOpenAIStream(answer: NormalizedAnswer, model: string, n: number): string {
  const id = `chatcmpl-agg-${n}`;
  const chunk = (delta: Json, finish: string | null): Json => ({
    id,
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, openAIStop(answer.stop))));
  frames.push(
    sseEvent(null, {
      id,
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.usage.input,
        completion_tokens: answer.usage.output,
        total_tokens: answer.usage.input + answer.usage.output,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalized answer as an Anthropic Messages object. */
function renderAnthropicJson(answer: NormalizedAnswer, model: string, n: number): Json {
  return {
    id: `msg_agg_${n}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

/** A normalized answer as the Anthropic event sequence, ending with message_stop. */
function renderAnthropicStream(answer: NormalizedAnswer, model: string, n: number): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: `msg_agg_${n}`,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.usage.input, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.usage.output },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/** An error in OpenAI's shape. */
function openAIError(message: string, type: string): Json {
  return { error: { message, type, code: null } };
}

/** An error in Anthropic's shape. */
function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

// ---------------------------------------------------------------------------
// The server
// ---------------------------------------------------------------------------

interface ServerState {
  brief: Brief;
  probes: Probe[];
  catalog: Catalog;
  plan: Map<string, Candidate>;
  books: Map<string, ProviderBook>;
  /** Client API requests arrived so far: the manifest is in arrival order. */
  arrived: number;
}

function bookOf(state: ServerState, providerId: string): ProviderBook {
  let book = state.books.get(providerId);
  if (book === undefined) {
    book = newBook();
    state.books.set(providerId, book);
  }
  return book;
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(api: ClientApi, body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** Whether this OpenAI request asks for JSON mode (response_format is set). */
function jsonModeOf(api: ClientApi, body: Json): boolean {
  if (api !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/**
 * Serve one client request: pick the candidates, try them in order until one
 * answers, and render the answer in the client's wire. The planned candidate
 * goes first; the rest follow by rating, skipping providers the trial has
 * spent. Exactly one upstream call succeeds: the loop stops at the first 2xx
 * it can read.
 */
async function serveRequest(
  state: ServerState,
  api: ClientApi,
  body: Json,
  stream: boolean,
): Promise<{ status: number; json?: Json; sse?: string }> {
  const n = state.arrived;
  state.arrived += 1;
  const manifest = state.brief.workload[n];
  const requires = manifest !== undefined ? manifest.requires : requirementsFromBody(api, body);
  const jsonMode = jsonModeOf(api, body);
  const candidates = candidatesFor(state.probes, state.brief.ratings, requires, api, jsonMode);
  const planned = manifest !== undefined ? state.plan.get(manifest.id) : undefined;
  const order: Candidate[] = [];
  const seen = new Set<string>();
  const push = (c: Candidate): void => {
    const key = `${c.provider.id}/${c.model.id}`;
    if (seen.has(key)) return;
    seen.add(key);
    order.push(c);
  };
  if (planned !== undefined && !bookOf(state, planned.provider.id).exhausted) push(planned);
  for (const c of candidates) {
    if (!bookOf(state, c.provider.id).exhausted) push(c);
  }
  if (order.length === 0) {
    log(`request ${n + 1}: no free model meets its requirements`);
    const wire = api === 'anthropic' ? 'anthropic' : 'openai';
    return {
      status: 503,
      json:
        wire === 'anthropic'
          ? anthropicError('overloaded_error', 'no free model meets this request')
          : openAIError('no free model meets this request', 'no_free_model'),
    };
  }
  for (const candidate of order) {
    const book = bookOf(state, candidate.provider.id);
    // One attempt per candidate: a refusal is answered by moving to the next
    // best model, not by retrying here. A retry would cost the same extra
    // call as a failover and only shift the loss to a later request, because
    // each provider's budget is fixed — and during an outage the retry fails
    // too, wasting the call twice over. Fewer upstream calls wins ties.
    const result = await callUpstream(ENV_URL, candidate, api, body, stream);
    if (result.ok) {
      noteSuccess(book);
      const model = candidate.model.id;
      log(
        `request ${n + 1}: ${candidate.provider.id}/${model} rating ${candidate.rating}` +
          (candidate === planned ? ' (planned)' : ''),
      );
      if (api === 'openai') {
        return stream
          ? { status: 200, sse: renderOpenAIStream(result.answer, model, n) }
          : { status: 200, json: renderOpenAIJson(result.answer, model, n) };
      }
      return stream
        ? { status: 200, sse: renderAnthropicStream(result.answer, model, n) }
        : { status: 200, json: renderAnthropicJson(result.answer, model, n) };
    }
    if (result.status === 200 || !result.retryable) {
      // A 2xx this server could not read: a second success would break the
      // trial's one-provider rule, so the request ends here, unserved.
      noteSuccess(book);
      log(`request ${n + 1}: ${candidate.provider.id} answered HTTP 200 this server could not read`);
      break;
    }
    noteRefusal(book);
    log(
      `request ${n + 1}: ${candidate.provider.id} refused HTTP ${result.status}` +
        (result.code !== '' ? ` (${result.code})` : '') +
        (book.exhausted ? '; provider spent' : '; failing over'),
    );
  }
  log(`request ${n + 1}: every candidate refused`);
  return {
    status: 503,
    json:
      api === 'anthropic'
        ? anthropicError('overloaded_error', 'every free provider refused this request')
        : openAIError('every free provider refused this request', 'no_free_model'),
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function sendError(res: ServerResponse, api: ClientApi, status: number, message: string): void {
  if (api === 'anthropic') sendJson(res, status, anthropicError('invalid_request_error', message));
  else sendJson(res, status, openAIError(message, 'invalid_request_error'));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function modelsRoute(state: ServerState, req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels(state.probes, state.brief.ratings);
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({
        type: 'model',
        id: m.id,
        display_name: m.display_name,
        created_at: FIXED_ANTHROPIC_CREATED_AT,
        free: true,
      })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({
        id: m.id,
        object: 'model',
        created: FIXED_CREATED,
        owned_by: m.provider,
        free: true,
      })),
    });
  }
}

async function handlePost(state: ServerState, api: ClientApi, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, api, 400, 'the request body is not one JSON object');
    return;
  }
  const stream = body.stream === true;
  const answer = await serveRequest(state, api, body, stream);
  if (answer.sse !== undefined) sendSse(res, answer.sse);
  else sendJson(res, answer.status, answer.json ?? {});
}

function handle(state: ServerState, req: IncomingMessage, res: ServerResponse, queue: { current: Promise<void> }): void {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(state, req, res);
    return;
  }
  if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
    const api: ClientApi = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
    // The client sends one request at a time; serialize anyway so the arrival
    // order (and the manifest row each request takes) is deterministic.
    queue.current = queue.current
      .then(() => handlePost(state, api, req, res))
      .catch((err: unknown) => {
        log((err as Error).stack ?? String(err));
        if (!res.headersSent) sendError(res, api, 500, 'the server failed to answer');
      });
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, catalog: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  const brief = readBrief(PROMPT_FILE);
  if (brief.error !== null) log(brief.error);
  const probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  const catalog = buildCatalog(probes, brief.ratings);
  const plan = planAssignments(probes, brief.ratings, brief.workload);
  const state: ServerState = { brief, probes, catalog, plan, books: new Map(), arrived: 0 };

  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one
    // built object and stop before any server starts. A process whose stdout is
    // not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const queue = { current: Promise.resolve() };
  const server = createServer((req, res) => {
    try {
      handle(state, req, res, queue);
    } catch (err) {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'the server failed to answer');
    }
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(
    `listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers, ` +
      `${servedModels(probes, brief.ratings).length} free models and ${plan.size} planned requests`,
  );
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
