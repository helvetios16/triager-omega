import { afterEach, describe, expect, test } from 'bun:test';
import {
  callDeepSeekWithRetry,
  CircuitBreaker,
  extractJson,
  isValid,
  toDistilledText,
  runBatchDistillation,
  type Distillation,
} from '../scripts/distill/deepseek_engine';

const originalFetch = globalThis.fetch;
const originalSleep = Bun.sleep;

afterEach(() => {
  globalThis.fetch = originalFetch;
  Bun.sleep = originalSleep;
});

const valid: Distillation = {
  fault_location: { product: 'TypeScript', component: 'Type Checker', subsystem: 'inference' },
  symptoms: ['wrong type'],
  capabilities: ['type inference'],
};

describe('deepseek_engine characterization', () => {
  test('validates required object fields and string arrays', () => {
    expect(isValid(valid)).toBe(true);
    expect(isValid({ ...valid, symptoms: ['ok', 2] })).toBe(false);
    expect(isValid({ ...valid, fault_location: { product: 'P', component: 'C' } })).toBe(true);
    expect(isValid(null)).toBeFalsy();
  });

  test('serializes canonical [FL][SY][CP] and omits empty location parts', () => {
    expect(toDistilledText(valid)).toBe('[FL] TypeScript Type Checker inference [SY] wrong type [CP] type inference');
    expect(toDistilledText({ ...valid, fault_location: { ...valid.fault_location, subsystem: '' }, symptoms: [], capabilities: [] }))
      .toBe('[FL] TypeScript Type Checker [SY]  [CP]');
  });

  test('extracts nested JSON with braces and escaped quotes inside strings', () => {
    expect(extractJson('prefix ```json\n{"x":{"s":"} \\\" {"}}\n``` suffix'))
      .toEqual({ x: { s: '} " {' } });
    expect(() => extractJson('nothing here')).toThrow('sin objeto JSON');
  });

  test('circuit breaker counts consecutive failures and resets on success', () => {
    const breaker = new CircuitBreaker(2);
    expect(breaker.recordFailure()).toBe(false);
    expect(breaker.failures).toBe(1);
    expect(breaker.recordFailure()).toBe(true);
    breaker.recordSuccess();
    expect(breaker.failures).toBe(0);
    expect(breaker.recordFailure()).toBe(false);
  });

  test('retries HTTP 429 and records response usage after succeeding', async () => {
    const sleeps: number[] = [];
    Bun.sleep = (async (ms: number) => { sleeps.push(ms); }) as typeof Bun.sleep;
    let calls = 0;
    globalThis.fetch = (async () => {
      calls++;
      if (calls === 1) return new Response('busy', { status: 429 });
      return Response.json({
        choices: [{ message: { content: '{"ok":true}' } }],
        usage: { prompt_tokens: 20, completion_tokens: 3, prompt_cache_hit_tokens: 7 },
      });
    }) as typeof fetch;

    const result = await callDeepSeekWithRetry({
      apiKey: 'offline-test-key', baseUrl: 'https://invalid.test', model: 'offline',
      systemMessage: 'system', userPrompt: 'prompt', maxRetries: 2,
    });
    expect(calls).toBe(2);
    expect(sleeps).toEqual([1000]);
    expect(result.text).toBe('{"ok":true}');
    expect(result.usage).toMatchObject({ prompt_tokens: 20, completion_tokens: 3, cache_hit_tokens: 7, cache_miss_tokens: 20 });
  });

  test('batch API uses in-memory rows, bounded workers, completion-order fallback breaker and flush', async () => {
    let active = 0;
    let peak = 0;
    const persisted: string[] = [];
    const fallbackDistillation = { ...valid, _fallback: true };
    const result = await runBatchDistillation<{ id: string }, { id: string; fallback: boolean }>([
      { id: 'a' }, { id: 'b' }, { id: 'c' }, { id: 'd' },
    ], {
      concurrency: 2, checkpointPath: 'unused', flushEvery: 2, breakerThreshold: 10,
      outputDirectory: 'unused', splitName: 'memory',
    }, {
      id: (row) => row.id,
      distill: async (row) => {
        active++;
        peak = Math.max(peak, active);
        await Promise.resolve();
        active--;
        return { distillation: row.id === 'c' ? fallbackDistillation : valid, promptTokens: 0, completionTokens: 0, cacheHitTokens: 0 };
      },
      makeRecord: (row, response) => ({ id: row.id, fallback: Boolean(response.distillation._fallback) }),
      append: async (_path, row) => { persisted.push(row.id); },
      flush: async () => {},
    });
    expect(result).toEqual({ completed: 4, fallbacks: 1, aborted: false });
    expect(peak).toBeLessThanOrEqual(2);
    expect(persisted).toHaveLength(4);
  });

  test('legacy request profile preserves 5 attempts, pre-attempt randomized backoff and thinking setting', async () => {
    const sleeps: number[] = [];
    const bodies: any[] = [];
    let calls = 0;
    const result = await callDeepSeekWithRetry({
      apiKey: 'offline', baseUrl: 'https://invalid.test/', model: 'offline', systemMessage: 'sys', userPrompt: 'p',
      enableThinking: true,
      profile: { attempts: 5, backoff: 'runner-legacy', retryEmptyResponse: true, cacheMiss: 'runner-legacy-or' },
      dependencies: {
        fetch: (async (_url, init) => {
          bodies.push(JSON.parse(String(init?.body)));
          calls++;
          return calls < 5 ? new Response('busy', { status: 429 }) : Response.json({ choices: [{ message: { content: 'ok' } }], usage: { prompt_tokens: 10, prompt_cache_hit_tokens: 4 } });
        }) as typeof fetch,
        sleep: async (ms) => { sleeps.push(ms); }, random: () => 0.25, now: () => 10,
      },
    });
    expect(calls).toBe(5);
    expect(sleeps).toEqual([2125, 4125, 8125, 10000]);
    expect(bodies.every((body) => body.thinking.type === 'enabled')).toBe(true);
    expect(result.usage.cache_miss_tokens).toBe(6);
  });

  test('engine profile omits enabled thinking, measures latency before JSON parsing, and preserves uncapped retry sleeps', async () => {
    const sleeps: number[] = [];
    let tick = 0;
    const payloads: any[] = [];
    const result = await callDeepSeekWithRetry({
      apiKey: 'offline', baseUrl: 'https://invalid.test', model: 'offline', systemMessage: 'sys', userPrompt: 'p',
      enableThinking: true, maxRetries: 3,
      dependencies: {
        now: () => tick,
        sleep: async (ms) => { sleeps.push(ms); },
        fetch: (async (_url, init) => {
          payloads.push(JSON.parse(String(init?.body)));
          tick += 5;
          if (payloads.length < 3) return new Response('busy', { status: 503 });
          return { ok: true, status: 200, json: async () => { tick += 100; return { choices: [{ message: { content: 'ok' } }] }; } } as Response;
        }) as typeof fetch,
      },
    });
    expect(payloads.every((body) => !('thinking' in body))).toBe(true);
    expect(sleeps).toEqual([1000, 2000]);
    expect(result.usage.latency_ms).toBe(5);
  });

  test('engine 503 retries remain uncapped and include the final legacy sleep', async () => {
    const sleeps: number[] = [];
    let calls = 0;
    await expect(callDeepSeekWithRetry({
      apiKey: 'offline', baseUrl: 'https://invalid.test', model: 'offline',
      systemMessage: 'system', userPrompt: 'prompt', maxRetries: 7,
      dependencies: {
        fetch: (async () => { calls++; return new Response('busy', { status: 503 }); }) as typeof fetch,
        sleep: async (ms) => { sleeps.push(ms); },
      },
    })).rejects.toThrow('HTTP 503: busy');
    expect(calls).toBe(7);
    expect(sleeps).toEqual([1000, 2000, 4000, 8000, 16000, 32000, 64000]);
  });

  test('breaker stops new assignments while already in-flight rows finish', async () => {
    const releases = new Map<string, () => void>();
    const assigned: string[] = [];
    let signalStarted!: () => void;
    let signalTripped!: () => void;
    const started = new Promise<void>((resolve) => { signalStarted = resolve; });
    const tripped = new Promise<void>((resolve) => { signalTripped = resolve; });
    const result = runBatchDistillation([
      { id: 'first' }, { id: 'inflight' }, { id: 'unassigned' },
    ], {
      concurrency: 2, checkpointPath: 'unused', flushEvery: 10,
      breakerThreshold: 1, outputDirectory: 'unused', splitName: 'test',
    }, {
      id: (row) => row.id,
      distill: async (row) => {
        assigned.push(row.id);
        await new Promise<void>((resolve) => {
          releases.set(row.id, resolve);
          if (releases.size === 2) signalStarted();
        });
        return {
          distillation: row.id === 'first' ? { ...valid, _fallback: true } : valid,
          promptTokens: 0, completionTokens: 0, cacheHitTokens: 0,
        };
      },
      makeRecord: (row) => row,
      append: async () => {},
      flush: async () => {},
      onBreaker: () => signalTripped(),
    });
    await started;
    expect(assigned).toEqual(['first', 'inflight']);
    releases.get('first')!();
    await tripped;
    releases.get('inflight')!();
    expect(await result).toEqual({ completed: 2, fallbacks: 1, aborted: true });
    expect(assigned).toEqual(['first', 'inflight']);
  });

  test('retries non-retryable HTTP status because thrown status errors enter catch retry path', async () => {
    const sleeps: number[] = [];
    Bun.sleep = (async (ms: number) => { sleeps.push(ms); }) as typeof Bun.sleep;
    let calls = 0;
    globalThis.fetch = (async () => {
      calls++;
      return new Response('bad request', { status: 400 });
    }) as typeof fetch;

    await expect(callDeepSeekWithRetry({
      apiKey: 'offline-test-key', baseUrl: 'https://invalid.test', model: 'offline',
      systemMessage: 'system', userPrompt: 'prompt', maxRetries: 2,
    })).rejects.toThrow('HTTP 400: bad request');
    expect(calls).toBe(2);
    expect(sleeps).toEqual([1000]);
  });
});
