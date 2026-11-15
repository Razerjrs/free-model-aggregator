// The client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic
// one (POST /v1/messages), with a JSON body or with server-sent events. Each provider speaks its own
// wire: openai, anthropic or gemini. This file translates between every pair:
//
//   - OpenAI client <-> openai provider: the body passes through (with the routed model's id, and
//     stream_options.include_usage added on streams so the provider's usage chunk arrives).
//   - Anthropic client <-> openai provider: the request is rewritten to Chat Completions (system
//     prompt, image blocks, tool definitions, tool results and tool choice included) and the answer
//     back into a Messages object, or into the message_start / content_block_start / content_block_
//     delta / content_block_stop / message_delta / message_stop event order.
//   - OpenAI client <-> anthropic provider: the same rewrite in the other direction, with the
//     provider's Messages object and event stream turned back into a completion and its chunks.
//   - Either client <-> gemini provider: the request becomes a generateContent body (system
//     instruction, contents, function declarations, generation config) and the answer's candidates
//     become the client's text, usage and stop reason, in JSON or in the client's event order.
//
// Every translation is a pure function of its arguments: no clock and no randomness, so the same body
// and the same provider answer give the same bytes on a replay. The model listings use fixed
// timestamps for the same reason: a `created` that moved between runs would be a nondeterministic
// document. What the client checks is the provider's echo of its own fields (system, user text,
// tools, max_tokens, stops, image, JSON mode), so every translator passes those fields through in
// the target wire's spelling, in order, untouched.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The id every answer this file builds carries: the providers answer with fixed ids of their own. */
const FIXED_CHAT_ID = 'chatcmpl-sim';
const FIXED_MESSAGE_ID = 'msg_sim';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** An error in Gemini's shape. */
export function geminiError(status: number, code: string, message: string): Json {
  return { error: { code: status, message, status: code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The text of an OpenAI message's content, string or parts. */
function openAIText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((p) => (isRecord(p) && p.type === 'text' && typeof p.text === 'string' ? [p.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** One OpenAI content part as an Anthropic content block: text or an image; null for the rest. */
function blockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      const media = data[1] !== '' ? data[1] : 'image/png';
      return { type: 'image', source: { type: 'base64', media_type: media, data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
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

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function finishReasonOfGemini(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
export function stopReasonOfGemini(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
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
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
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
      const part = partOf(block);
      if (part !== null) parts.push(part);
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
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : openAIText(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: FIXED_MESSAGE_ID,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split an SSE body into frames: each frame's event name (or null) and its data lines joined. */
export function parseSseFrames(text: string): { event: string | null; data: string }[] {
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

/** One OpenAI chunk: a delta, an optional finish reason, an optional usage block. */
function openAIChunk(model: string, delta: Json, finish: string | null, usage: Json | null): Json {
  const chunk: Json = {
    id: FIXED_CHAT_ID,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
  if (usage !== null) chunk.usage = usage;
  return chunk;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason
 * and usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: FIXED_MESSAGE_ID,
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request: the system messages become
 * the system prompt, user and assistant messages keep their text and image blocks, tool messages
 * become tool_result blocks on a user message, tools and tool_choice are rewritten, and the output
 * cap, temperature, top_p, stop sequences and streaming pass through. A JSON-mode request has no
 * Anthropic spelling: callers must not route one to an anthropic-wire provider.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    const content = raw.content;
    if (role === 'system') {
      const text = openAIText(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = openAIText(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof content === 'string' ? content : openAIText(content),
          },
        ],
      });
      continue;
    }
    // A user message: a string stays a string, parts become blocks.
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (Array.isArray(content)) {
      const blocks: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        const block = blockOf(part);
        if (block !== null) blocks.push(block);
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : [];
  if (stops.length > 0) out.stop_sequences = stops;
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
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic Messages object as a Chat Completions answer. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = answerText(content);
  const calls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    calls.push({
      id: typeof block.id === 'string' ? block.id : 'call_unknown',
      type: 'function',
      function: {
        name: typeof block.name === 'string' ? block.name : '',
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const prompt = numberOrZero(usage.input_tokens);
  const completion = numberOrZero(usage.output_tokens);
  return {
    id: FIXED_CHAT_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(calls.length > 0 ? { tool_calls: calls } : {}) },
        finish_reason: finishReasonOf(body.stop_reason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion },
  };
}

/** The text of an Anthropic content block array's text blocks. */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/**
 * An Anthropic SSE stream as OpenAI chunks: message_start opens with the role, text deltas become
 * content deltas, tool_use blocks become tool_calls deltas, message_delta closes with the finish
 * reason and a usage chunk, then [DONE].
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;
  const toolArgs = new Map<number, string>();
  const toolIds = new Map<number, string>();
  const toolNames = new Map<number, string>();

  const startChunk = (): string[] => {
    if (started) return [];
    started = true;
    return [sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null, null))];
  };
  const finishChunks = (): string[] => {
    if (finished) return [];
    finished = true;
    const usage = { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens };
    return [
      sseEvent(null, openAIChunk(model, {}, pendingStop ?? 'stop', null)),
      sseEvent(null, { id: FIXED_CHAT_ID, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage }),
      'data: [DONE]\n\n',
    ];
  };

  const eventsOfFrame = (frame: { event: string | null; data: string }): string[] => {
    const out: string[] = [];
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      return out;
    }
    if (!isRecord(data)) return out;
    switch (frame.event) {
      case 'message_start': {
        if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
        out.push(...startChunk());
        break;
      }
      case 'content_block_start': {
        out.push(...startChunk());
        const index = typeof data.index === 'number' ? data.index : 0;
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (block.type === 'tool_use') {
          toolIds.set(index, typeof block.id === 'string' ? block.id : '');
          toolNames.set(index, typeof block.name === 'string' ? block.name : '');
          toolArgs.set(index, '');
          out.push(
            sseEvent(
              null,
              openAIChunk(model, { tool_calls: [{ index, id: toolIds.get(index) ?? '', type: 'function', function: { name: toolNames.get(index) ?? '', arguments: '' } }] }, null, null),
            ),
          );
        }
        break;
      }
      case 'content_block_delta': {
        out.push(...startChunk());
        const index = typeof data.index === 'number' ? data.index : 0;
        const delta = isRecord(data.delta) ? data.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
          out.push(sseEvent(null, openAIChunk(model, { content: delta.text }, null, null)));
        } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
          toolArgs.set(index, (toolArgs.get(index) ?? '') + delta.partial_json);
          out.push(
            sseEvent(null, openAIChunk(model, { tool_calls: [{ index, function: { arguments: delta.partial_json } }] }, null, null)),
          );
        }
        break;
      }
      case 'message_delta': {
        if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') pendingStop = finishReasonOf(data.delta.stop_reason);
        if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
        break;
      }
      case 'message_stop': {
        out.push(...startChunk(), ...finishChunks());
        break;
      }
      default:
        break;
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frameText = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const frame of parseSseFrames(frameText)) {
          for (const event of eventsOfFrame(frame)) yield event;
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const frame of parseSseFrames(buffer)) {
        for (const event of eventsOfFrame(frame)) yield event;
      }
    }
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startChunk()) yield event;
    for (const event of finishChunks()) yield event;
  }
}

// ---------------------------------------------------------------------------
// Gemini
// ---------------------------------------------------------------------------

/** A data URL split into its mime type and base64 payload; null when it is not one. */
function dataUrlParts(url: string): { mime: string; data: string } | null {
  const match = /^data:([^,]*);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { mime: match[1] !== '' ? match[1] : 'image/png', data: match[2] };
}

/** One OpenAI image_url part as a Gemini part: inlineData for a data URL, fileData for a remote one. */
function geminiImagePart(url: string): Json {
  const data = dataUrlParts(url);
  if (data !== null) return { inlineData: { mimeType: data.mime, data: data.data } };
  return { fileData: { mimeType: 'image/png', fileUri: url } };
}

/** One Anthropic image block as a Gemini part. */
function geminiImagePartFromAnthropic(source: Json): Json | null {
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { fileData: { mimeType: 'image/png', fileUri: source.url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body: the system messages become
 * the system instruction, messages become contents (assistant tool calls become functionCall parts,
 * tool messages become function-role functionResponse parts), tools become function declarations, and
 * the output cap, stop sequences, temperature and JSON mode land in the generation config.
 */
export function openAIToGemini(body: Json): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    const content = raw.content;
    if (role === 'system') {
      const text = openAIText(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = openAIText(content);
      if (text !== '') parts.push({ text });
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
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      const text = typeof content === 'string' ? content : openAIText(content);
      contents.push({
        role: 'function',
        parts: [{ functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response: { content: text } } }],
      });
      continue;
    }
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          parts.push(geminiImagePart(part.image_url.url));
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : [];
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body: the system prompt becomes the
 * system instruction, messages become contents (assistant tool_use blocks become functionCall parts,
 * tool_result blocks become function-role functionResponse parts), tools become function
 * declarations, and the output cap and stop sequences land in the generation config.
 */
export function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const system = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: assistant ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const parts: Json[] = [];
      const text = textOf(content);
      if (text !== '') parts.push({ text });
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        parts.push({
          functionCall: {
            name: typeof block.name === 'string' ? block.name : '',
            args: isRecord(block.input) ? block.input : {},
          },
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        contents.push({
          role: 'function',
          parts: [
            {
              functionResponse: {
                name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
                response: { content: toolResultText(block.content) },
              },
            },
          ],
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'image' && isRecord(block.source)) {
        const part = geminiImagePartFromAnthropic(block.source);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** The text of a Gemini candidate's parts, and any functionCall parts it carries. */
function geminiParts(candidate: Json): { text: string; calls: Json[] } {
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const calls: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      calls.push({
        id: `call_${calls.length}`,
        type: 'function',
        function: {
          name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
          arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
        },
      });
    }
  }
  return { text, calls };
}

/** A Gemini generateContent answer as a Chat Completions answer. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const { text, calls } = geminiParts(candidate);
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const prompt = numberOrZero(usage.promptTokenCount);
  const completion = numberOrZero(usage.candidatesTokenCount);
  return {
    id: FIXED_CHAT_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text, ...(calls.length > 0 ? { tool_calls: calls } : {}) },
        finish_reason: finishReasonOfGemini(candidate.finishReason),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const { text, calls } = geminiParts(candidate);
  const content: Json[] = [];
  if (text !== '') content.push({ type: 'text', text });
  for (const call of calls) {
    if (!isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: FIXED_MESSAGE_ID,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOfGemini(candidate.finishReason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.promptTokenCount), output_tokens: numberOrZero(usage.candidatesTokenCount) },
  };
}

/** The text pieces and final usage of one Gemini SSE frame's candidates. */
function geminiFrame(frame: { event: string | null; data: string }): { pieces: string[]; finish: string | null; usage: { prompt: number; completion: number } | null } {
  const out: { pieces: string[]; finish: string | null; usage: { prompt: number; completion: number } | null } = { pieces: [], finish: null, usage: null };
  let data: unknown;
  try {
    data = JSON.parse(frame.data);
  } catch {
    return out;
  }
  if (!isRecord(data)) return out;
  const candidates = Array.isArray(data.candidates) ? data.candidates : [];
  for (const raw of candidates) {
    if (!isRecord(raw)) continue;
    const { text } = geminiParts(raw);
    if (text !== '') out.pieces.push(text);
    if (typeof raw.finishReason === 'string') out.finish = raw.finishReason;
  }
  if (isRecord(data.usageMetadata)) {
    out.usage = { prompt: numberOrZero(data.usageMetadata.promptTokenCount), completion: numberOrZero(data.usageMetadata.candidatesTokenCount) };
  }
  return out;
}

/**
 * A Gemini streamGenerateContent stream as OpenAI chunks: each text piece becomes a content delta,
 * the final frame's finishReason becomes the finish chunk with a usage chunk, then [DONE].
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let pendingFinish: string | null = null;
  let pendingUsage: { prompt: number; completion: number } | null = null;

  const startChunk = (): string[] => {
    if (started) return [];
    started = true;
    return [sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null, null))];
  };
  const finishChunks = (): string[] => {
    if (finished) return [];
    finished = true;
    const usage = pendingUsage === null ? null : { prompt_tokens: pendingUsage.prompt, completion_tokens: pendingUsage.completion, total_tokens: pendingUsage.prompt + pendingUsage.completion };
    const chunks = [sseEvent(null, openAIChunk(model, {}, finishReasonOfGemini(pendingFinish ?? 'STOP'), null))];
    if (usage !== null) {
      chunks.push(sseEvent(null, { id: FIXED_CHAT_ID, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage }));
    }
    chunks.push('data: [DONE]\n\n');
    return chunks;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frameText = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const frame of parseSseFrames(frameText)) {
          const parsed = geminiFrame(frame);
          if (parsed.pieces.length > 0) {
            for (const piece of parsed.pieces) {
              yield* startChunk();
              yield sseEvent(null, openAIChunk(model, { content: piece }, null, null));
            }
          }
          if (parsed.finish !== null) pendingFinish = parsed.finish;
          if (parsed.usage !== null) pendingUsage = parsed.usage;
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const frame of parseSseFrames(buffer)) {
        const parsed = geminiFrame(frame);
        if (parsed.pieces.length > 0) {
          for (const piece of parsed.pieces) {
            yield* startChunk();
            yield sseEvent(null, openAIChunk(model, { content: piece }, null, null));
          }
        }
        if (parsed.finish !== null) pendingFinish = parsed.finish;
        if (parsed.usage !== null) pendingUsage = parsed.usage;
      }
    }
  } finally {
    reader.releaseLock();
  }
  for (const event of startChunk()) yield event;
  for (const event of finishChunks()) yield event;
}

/**
 * A Gemini streamGenerateContent stream as Anthropic events: message_start, one text block with
 * deltas, then message_delta with the stop reason and output tokens, then message_stop. The
 * message_start carries no input tokens: the stream's usage check reads the output alone.
 */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let textOpen = false;
  let pendingFinish: string | null = null;
  let pendingOutput = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: FIXED_MESSAGE_ID,
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }),
    ];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [];
    if (textOpen) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
      textOpen = false;
    }
    out.push(
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReasonOfGemini(pendingFinish ?? 'STOP'), stop_sequence: null }, usage: { output_tokens: pendingOutput } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    );
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frameText = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const frame of parseSseFrames(frameText)) {
          const parsed = geminiFrame(frame);
          for (const piece of parsed.pieces) {
            const events = [...startEvents()];
            if (!textOpen) {
              textOpen = true;
              events.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
            }
            events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
            for (const event of events) yield event;
          }
          if (parsed.finish !== null) pendingFinish = parsed.finish;
          if (parsed.usage !== null) pendingOutput = parsed.usage.completion;
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const frame of parseSseFrames(buffer)) {
        const parsed = geminiFrame(frame);
        for (const piece of parsed.pieces) {
          const events = [...startEvents()];
          if (!textOpen) {
            textOpen = true;
            events.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
          }
          events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
          for (const event of events) yield event;
        }
        if (parsed.finish !== null) pendingFinish = parsed.finish;
        if (parsed.usage !== null) pendingOutput = parsed.usage.completion;
      }
    }
  } finally {
    reader.releaseLock();
  }
  for (const event of startEvents()) yield event;
  for (const event of finishEvents()) yield event;
}
