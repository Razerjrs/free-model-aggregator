// The entry: one server that routes the trial's 20 requests through the 10 simulated providers.
//
// The brief at $MAKEFUN_PROMPT_FILE is a text file carrying three JSON documents, each found by its
// `schema` member: the provider directory (id, wire, auth scheme, key env var, the key the simulated
// service takes, base path under $MAKEFUN_ENV_URL), the quality table (every free model's rating and
// capabilities) and the workload manifest (the 20 requests in order, with api, streaming, weight and
// requirements). The server probes each provider's catalog route in the provider's own wire, builds
// one catalog, registers itself, serves GET /v1/models, POST /v1/chat/completions and POST
// /v1/messages (JSON or SSE), and polls GET $MAKEFUN_ENV_URL/work until the environment says
// {"done": true}, then exits 0.
//
// Routing: every request is ranked over the free models that meet it (by rating_milli, with the
// request's JSON-mode requests kept off anthropic-wire providers), then spread over providers under
// their per-trial budget caps, so no provider is asked for more than it can serve. When a provider
// answers 402, 429 or 5xx, the request is retried on the next free model that meets it; a provider
// whose answer says its budget is spent is not tried again. Only one provider call per request can
// succeed, and the calls are sequential, so a provider that allows one call at a time is never
// called twice at once. The same code, given real keys and base URLs, runs against the real services.
//
// Deterministic: no wall clock and no unseeded randomness anywhere in this file. The catalog is built
// once and the same object is printed by --list and sent in the register body, so the two documents
// are byte for byte the same. The catalog's base_url is the directory's own path, never joined to the
// environment's URL: the origin differs between a live pass and its replay, and the catalog is part
// of the state export a replay must reproduce byte for byte.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import {
  PLANNED_CALLS,
  authHeaders,
  buildCatalog,
  candidatesFor,
  joinBase,
  planAssignments,
  probeProvider,
  readBrief,
  type Brief,
  type Candidate,
  type CatalogReport,
  type Json,
  type Probe,
  type ProviderDirectoryEntry,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
/** How long an upstream call may take before it counts as a failure: 5 seconds. */
const UPSTREAM_TIMEOUT_MS = 5000;
/** How often to ask the environment whether the workload is done: every 100 ms. */
const WORK_POLL_MS = 100;
/** The Anthropic API version the anthropic wire's providers expect. */
const ANTHROPIC_VERSION = '2023-06-01';

function env(name: string): string | undefined {
  const value = process.env[name];
  return value === undefined || value === '' ? undefined : value;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Read a request body fully and parse it as JSON; null when the body is empty or is not JSON. */
async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  const text = Buffer.concat(chunks).toString('utf8');
  if (text.trim() === '') return null;
  try {
    const parsed: unknown = JSON.parse(text);
    if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed as Json;
    return null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, events: AsyncGenerator<string>): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  const stream = ReadableStream.from(events);
  const reader = stream.getReader();
  const pump = async (): Promise<void> => {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(value);
    }
    res.end();
  };
  pump().catch(() => {
    try {
      res.end();
    } catch {
      // The client went away: nothing to send to.
    }
  });
}

// ---------------------------------------------------------------------------
// The brief, the probes and the catalog
// ---------------------------------------------------------------------------

interface CatalogState {
  brief: Brief;
  probes: Probe[];
  catalog: CatalogReport;
  models: ServedModel[];
  plan: Map<string, Candidate>;
  envUrl: string;
}

/** Read the brief, probe every provider and build the catalog the server registers. */
async function loadCatalogState(): Promise<CatalogState | null> {
  const briefPath = env('MAKEFUN_PROMPT_FILE');
  const envUrl = env('MAKEFUN_ENV_URL');
  if (briefPath === undefined || envUrl === undefined) return null;
  const brief = readBrief(briefPath);
  if (brief.error !== null) return null;
  const probes: Probe[] = [];
  for (const provider of brief.directory.providers) {
    probes.push(await probeProvider(envUrl, provider, UPSTREAM_TIMEOUT_MS));
  }
  const catalog = buildCatalog(envUrl, probes, brief.ratings);
  const models: ServedModel[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free) continue;
      models.push({ id: model.id, provider: provider.id, display_name: model.id });
    }
  }
  return { brief, probes, catalog, models, plan: planAssignments(brief, probes), envUrl };
}

// ---------------------------------------------------------------------------
// Upstream calls and routing
// ---------------------------------------------------------------------------

interface UpstreamResult {
  ok: boolean;
  status: number;
  body: string;
  json: Json | null;
  stream: ReadableStream<Uint8Array> | null;
  budgetSpent: boolean;
}

/** One call to a provider in its own wire, with a timeout. `streaming` asks for an SSE answer. */
async function callProvider(
  provider: ProviderDirectoryEntry,
  modelId: string,
  clientBody: Json,
  clientApi: 'openai' | 'anthropic',
  streaming: boolean,
  envUrl: string,
): Promise<UpstreamResult> {
  const base = joinBase(envUrl, provider.base);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
  try {
    let url = '';
    let body: Json;
    const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
    if (provider.wire === 'openai') {
      url = `${base}/v1/chat/completions`;
      body = clientApi === 'openai' ? { ...clientBody, model: modelId } : anthropicToOpenAI(clientBody, modelId);
      if (streaming && typeof body.stream_options !== 'object') body.stream_options = { include_usage: true };
    } else if (provider.wire === 'anthropic') {
      url = `${base}/v1/messages`;
      body = clientApi === 'anthropic' ? { ...clientBody, model: modelId } : openAIToAnthropicRequest(clientBody, modelId);
      headers['anthropic-version'] = ANTHROPIC_VERSION;
    } else {
      const action = streaming ? ':streamGenerateContent' : ':generateContent';
      url = `${base}/v1beta/models/${modelId}${action}${streaming ? '?alt=sse' : ''}`;
      body = clientApi === 'anthropic' ? anthropicToGemini(clientBody) : openAIToGemini(clientBody);
    }
    const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: controller.signal });
    const text = streaming && res.ok ? '' : await res.text();
    let json: Json | null = null;
    if (!streaming && text !== '') {
      try {
        const parsed: unknown = JSON.parse(text);
        if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) json = parsed as Json;
      } catch {
        json = null;
      }
    }
    const lowered = text.toLowerCase();
    return {
      ok: res.ok,
      status: res.status,
      body: text,
      json,
      stream: streaming && res.ok && res.body !== null ? res.body : null,
      budgetSpent: res.status === 402 || lowered.includes('budget is spent') || lowered.includes('is out') || lowered.includes('free allowance'),
    };
  } catch {
    return { ok: false, status: 0, body: '', json: null, stream: null, budgetSpent: false };
  } finally {
    clearTimeout(timer);
  }
}

/** The routing state: how many calls each provider has made, and which are spent. */
class Router {
  private readonly brief: Brief;
  private readonly probes: Probe[];
  private readonly plan: Map<string, Candidate>;
  private readonly calls = new Map<string, number>();
  private readonly spent = new Set<string>();

  constructor(state: CatalogState) {
    this.brief = state.brief;
    this.probes = state.probes;
    this.plan = state.plan;
  }

  /** Calls still allowed for a provider under its per-trial budget cap. */
  private remaining(providerId: string): number {
    if (this.spent.has(providerId)) return 0;
    const cap = PLANNED_CALLS[providerId] ?? 5;
    return cap - (this.calls.get(providerId) ?? 0);
  }

  /** The models to try for a request, best first: the planned pick, then the ranked candidates. */
  private orderFor(requestId: string): Candidate[] {
    const manifest = this.brief.workload?.manifest ?? [];
    const req = manifest.find((r) => r.id === requestId);
    if (req === undefined) return [];
    const ranked = candidatesFor(this.brief, this.probes, req.requires);
    const planned = this.plan.get(requestId);
    const ordered: Candidate[] = [];
    if (planned !== undefined) ordered.push(planned);
    for (const candidate of ranked) {
      if (!ordered.some((m) => m.model.id === candidate.model.id)) ordered.push(candidate);
    }
    return ordered;
  }

  /** The ranked candidates for a request the manifest does not name, from the body's own asks. */
  private orderForBody(clientBody: Json, clientApi: 'openai' | 'anthropic', streaming: boolean): Candidate[] {
    const messages = Array.isArray(clientBody.messages) ? clientBody.messages : [];
    let tools = Array.isArray(clientBody.tools) && clientBody.tools.length > 0;
    let vision = false;
    for (const raw of messages) {
      if (raw === null || typeof raw !== 'object') continue;
      const content = (raw as Record<string, unknown>).content;
      if (typeof content === 'string') continue;
      if (!Array.isArray(content)) continue;
      for (const part of content) {
        if (part === null || typeof part !== 'object') continue;
        const record = part as Record<string, unknown>;
        if (record.type === 'image_url' || record.type === 'image') vision = true;
        if (record.type === 'tool_use' || record.type === 'tool_result') tools = true;
      }
    }
    const format = clientBody.response_format;
    const json =
      clientApi === 'openai' &&
      format !== null &&
      typeof format === 'object' &&
      ((format as Record<string, unknown>).type === 'json_object' || (format as Record<string, unknown>).type === 'json_schema');
    const requires: Requirement = {};
    if (tools) requires.tools = true;
    if (vision) requires.vision = true;
    if (json) requires.json = true;
    return candidatesFor(this.brief, this.probes, requires);
  }

  /**
   * Serve one request: try the ordered models, skipping providers out of budget, and fall back to the
   * next free model that meets the request when one answers 402, 429 or 5xx. A provider whose answer
   * says its budget is spent is not tried again. When every provider is out of budget by the caps,
   * try the best models once more without the caps: the caps are the budgets' minima, and the draw
   * may have given a provider more. Returns the successful call's provider, model and result, or the
   * last failure when nothing served the request.
   */
  async serve(
    requestId: string,
    clientBody: Json,
    clientApi: 'openai' | 'anthropic',
    streaming: boolean,
    envUrl: string,
  ): Promise<{ candidate: Candidate; result: UpstreamResult } | { candidate: null; result: UpstreamResult | null }> {
    const ordered = this.orderFor(requestId);
    const models = ordered.length > 0 ? ordered : this.orderForBody(clientBody, clientApi, streaming);
    let last: UpstreamResult | null = null;
    const tryModels = async (list: Candidate[]): Promise<{ candidate: Candidate; result: UpstreamResult } | null> => {
      for (const candidate of list) {
        const provider = candidate.probe.provider;
        this.calls.set(provider.id, (this.calls.get(provider.id) ?? 0) + 1);
        const result = await callProvider(provider, candidate.model.id, clientBody, clientApi, streaming, envUrl);
        if (result.budgetSpent) this.spent.add(provider.id);
        if (result.ok) return { candidate, result };
        last = result;
      }
      return null;
    };
    const withinBudget = models.filter((m) => this.remaining(m.probe.provider.id) > 0);
    const served = await tryModels(withinBudget);
    if (served !== null) return served;
    const overBudget = models.filter((m) => !withinBudget.some((w) => w.model.id === m.model.id));
    const rescued = await tryModels(overBudget);
    if (rescued !== null) return rescued;
    return { candidate: null, result: last };
  }
}

/** The request's error message from an upstream body, in any wire's shape, or the fallback. */
function upstreamMessage(text: string, fallback: string): string {
  if (text === '') return fallback;
  try {
    const parsed: unknown = JSON.parse(text);
    if (parsed !== null && typeof parsed === 'object') {
      const record = parsed as Record<string, unknown>;
      const error = record.error;
      if (typeof error === 'string') return error;
      if (error !== null && typeof error === 'object') {
        const message = (error as Record<string, unknown>).message;
        if (typeof message === 'string') return message;
      }
      if (typeof record.message === 'string') return record.message;
    }
  } catch {
    // Not JSON: the raw text is the message.
  }
  return text.length > 200 ? text.slice(0, 200) : text;
}

/** The server's per-request state: the catalog, the router, and the workload pointer. */
interface ServerState {
  catalogState: CatalogState;
  router: Router;
  nextRequest: number;
}

/** The next workload entry for a client request, in manifest order; null once the manifest is spent. */
function nextWorkload(state: ServerState): { id: string } | null {
  const manifest = state.catalogState.brief.workload?.manifest ?? [];
  const entry = manifest[state.nextRequest];
  state.nextRequest += 1;
  return entry === undefined ? null : { id: entry.id };
}

/** Serve GET /v1/models: the free models, in the shape the caller's auth header asks for. */
function handleModels(req: IncomingMessage, res: ServerResponse, state: ServerState): void {
  const anthropic = req.headers['x-api-key'] !== undefined;
  sendJson(res, 200, anthropic ? anthropicModelList(state.catalogState.models) : openAIModelList(state.catalogState.models));
}

/** Serve one client completion request on either wire, JSON or SSE. */
async function handleCompletion(
  req: IncomingMessage,
  res: ServerResponse,
  state: ServerState,
  clientApi: 'openai' | 'anthropic',
): Promise<void> {
  const body = await readJsonBody(req);
  const errorFor = (message: string, type: string, code: string | null): Json =>
    clientApi === 'anthropic' ? anthropicError(type, message) : openAIError(message, type, code);
  if (body === null) {
    sendJson(res, 400, errorFor('The request body is not a JSON object.', 'invalid_request_error', 'invalid_body'));
    return;
  }
  const streaming = body.stream === true;
  const workload = nextWorkload(state);
  // A request past the manifest's end has no planned pick: route it by what its own body asks for.
  const requestId = workload === null ? '__unplanned__' : workload.id;
  const served = await state.router.serve(requestId, body, clientApi, streaming, state.catalogState.envUrl);
  if (served.candidate === null) {
    const failure = served.result;
    const status = failure === null || failure.status === 0 ? 503 : failure.status;
    const message = failure === null ? 'No provider could serve this request.' : upstreamMessage(failure.body, `The provider answered ${status}.`);
    sendJson(res, status, errorFor(message, 'api_error', null));
    return;
  }
  const { candidate, result } = served;
  const provider = candidate.probe.provider;
  const modelId = candidate.model.id;
  if (streaming) {
    const stream = result.stream;
    if (stream === null) {
      sendJson(res, 503, errorFor('The provider answered without a stream.', 'api_error', null));
      return;
    }
    // A stream already in the client's wire passes through untouched.
    const passthrough = (): AsyncGenerator<string> =>
      (async function* (): AsyncGenerator<string> {
        const reader = stream.getReader();
        const decoder = new TextDecoder();
        try {
          for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            yield decoder.decode(value, { stream: true });
          }
          yield decoder.decode();
        } finally {
          reader.releaseLock();
        }
      })();
    if (clientApi === 'openai') {
      if (provider.wire === 'openai') sendSse(res, passthrough());
      else if (provider.wire === 'anthropic') sendSse(res, openAIEventsFromAnthropic(stream, modelId));
      else sendSse(res, openAIEventsFromGemini(stream, modelId));
    } else if (provider.wire === 'openai') {
      sendSse(res, anthropicEventsFromOpenAI(stream, modelId));
    } else if (provider.wire === 'anthropic') {
      sendSse(res, passthrough());
    } else {
      sendSse(res, anthropicEventsFromGemini(stream, modelId));
    }
    return;
  }
  const upstream = result.json ?? {};
  if (clientApi === 'openai') {
    if (provider.wire === 'openai') sendJson(res, 200, upstream);
    else if (provider.wire === 'anthropic') sendJson(res, 200, anthropicToOpenAIResponse(upstream, modelId));
    else sendJson(res, 200, geminiToOpenAI(upstream, modelId));
  } else if (provider.wire === 'openai') {
    sendJson(res, 200, openAIToAnthropic(upstream, modelId));
  } else if (provider.wire === 'anthropic') {
    sendJson(res, 200, upstream);
  } else {
    sendJson(res, 200, geminiToAnthropic(upstream, modelId));
  }
}

/** Tell the environment where this server listens, retrying briefly while it starts up. */
async function register(envUrl: string, port: number, catalog: CatalogReport): Promise<boolean> {
  for (let attempt = 0; attempt < 10; attempt += 1) {
    try {
      const res = await fetch(`${envUrl}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
      });
      if (res.ok) return true;
    } catch {
      // The environment is not ready yet: wait and try again.
    }
    await sleep(WORK_POLL_MS);
  }
  return false;
}

/** Poll the environment until it says the workload is done, then shut the server down and exit 0. */
async function pollWork(server: ReturnType<typeof createServer>, envUrl: string): Promise<void> {
  for (;;) {
    await sleep(WORK_POLL_MS);
    try {
      const res = await fetch(`${envUrl}/work`);
      if (!res.ok) continue;
      const body: unknown = await res.json();
      if (body !== null && typeof body === 'object' && (body as Record<string, unknown>).done === true) {
        server.close(() => process.exit(0));
        setTimeout(() => process.exit(0), 1000).unref();
        return;
      }
    } catch {
      // The environment is not answering yet: keep polling.
    }
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const listMode = args.includes('--list');
  const catalogState = await loadCatalogState();
  if (listMode) {
    process.stdout.write(`${JSON.stringify(catalogState === null ? { providers: [] } : catalogState.catalog)}\n`);
    process.exit(0);
  }
  if (catalogState === null) {
    process.stderr.write('No brief at $MAKEFUN_PROMPT_FILE or no env at $MAKEFUN_ENV_URL: cannot serve.\n');
    process.exit(1);
  }
  const state: ServerState = { catalogState, router: new Router(catalogState), nextRequest: 0 };
  const server = createServer((req, res) => {
    const url = req.url ?? '/';
    const path = url.split('?')[0];
    if (req.method === 'GET' && path === '/v1/models') {
      handleModels(req, res, state);
      return;
    }
    if (req.method === 'POST' && path === '/v1/chat/completions') {
      handleCompletion(req, res, state, 'openai').catch((error: unknown) => {
        process.stderr.write(`completion failed: ${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
        sendJson(res, 500, openAIError(error instanceof Error ? error.message : 'The server failed.', 'api_error', null));
      });
      return;
    }
    if (req.method === 'POST' && path === '/v1/messages') {
      handleCompletion(req, res, state, 'anthropic').catch((error: unknown) => {
        process.stderr.write(`completion failed: ${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
        sendJson(res, 500, anthropicError('api_error', error instanceof Error ? error.message : 'The server failed.'));
      });
      return;
    }
    sendJson(res, 404, { error: { message: `No route for ${req.method ?? ''} ${path}.`, type: 'not_found', code: null } });
  });
  server.listen(0, '0.0.0.0', () => {
    const address = server.address();
    const port = typeof address === 'object' && address !== null ? address.port : 0;
    register(catalogState.envUrl, port, catalogState.catalog)
      .then((ok) => {
        if (!ok) {
          process.stderr.write('The environment did not accept the registration.\n');
          process.exit(1);
        }
        void pollWork(server, catalogState.envUrl);
      })
      .catch(() => process.exit(1));
  });
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
