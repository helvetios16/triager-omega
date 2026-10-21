// Motor compartido de destilación semántica vía DeepSeek API.
// Proporciona:
//   - Cliente HTTP con autenticación y exponential backoff
//   - Parseo y validación de la taxonomía canónica [FL][SY][CP]
//   - Circuit breaker para prevenir consumo innecesario ante caídas de API
//   - Serialización al formato estándar [FL] ... [SY] ... [CP] ...

export interface Distillation {
  fault_location: {
    product: string;
    component: string;
    subsystem: string;
  };
  symptoms: string[];
  capabilities: string[];
  _fallback?: boolean;
}

export interface DeepSeekUsage {
  prompt_tokens: number;
  completion_tokens: number;
  cache_hit_tokens: number;
  cache_miss_tokens: number;
  latency_ms: number;
}

export interface DeepSeekResult {
  text: string;
  usage: DeepSeekUsage;
}

export interface BatchDistillationConfig {
  concurrency: number;
  checkpointPath: string;
  flushEvery: number;
  breakerThreshold: number;
  outputDirectory: string;
  splitName: string;
}

export interface BatchDistillationAdapters<Row, RecordType> {
  id(row: Row): string;
  distill(row: Row): Promise<{ distillation: Distillation; promptTokens: number; completionTokens: number; cacheHitTokens: number }>;
  makeRecord(row: Row, result: { distillation: Distillation; promptTokens: number; completionTokens: number; cacheHitTokens: number }, latencyMs: number): RecordType;
  append(path: string, record: RecordType): Promise<void>;
  flush(path: string, outDir: string, split: string): Promise<void>;
  sleep?: (milliseconds: number) => Promise<void>;
  now?: () => number;
  onFallback?(consecutive: number, row: Row): void;
  onBreaker?(consecutive: number): void;
  afterRecord?(completed: number, row: Row, unassignedRows: number): Promise<void> | void;
}

/** Runs in-memory rows with per-invocation worker, breaker and checkpoint state. */
export async function runBatchDistillation<Row, RecordType>(
  rows: readonly Row[],
  config: BatchDistillationConfig,
  adapters: BatchDistillationAdapters<Row, RecordType>,
): Promise<{ completed: number; fallbacks: number; aborted: boolean }> {
  const breaker = new CircuitBreaker(config.breakerThreshold);
  const now = adapters.now ?? (() => performance.now());
  let next = 0;
  let completed = 0;
  let fallbacks = 0;
  let aborted = false;
  const worker = async () => {
    while (!aborted) {
      const index = next++;
      if (index >= rows.length) return;
      const row = rows[index];
      const start = now();
      const result = await adapters.distill(row);
      const record = adapters.makeRecord(row, result, Math.round(now() - start));
      await adapters.append(config.checkpointPath, record);
      completed++;
      if (result.distillation._fallback) {
        fallbacks++;
        const tripped = breaker.recordFailure();
        adapters.onFallback?.(breaker.failures, row);
        if (tripped) {
          aborted = true;
          adapters.onBreaker?.(breaker.failures);
        }
      } else breaker.recordSuccess();
      if (completed % Math.max(1, config.flushEvery) === 0 || completed === rows.length || aborted) {
        await adapters.flush(config.checkpointPath, config.outputDirectory, config.splitName);
      }
      await adapters.afterRecord?.(completed, row, Math.max(0, rows.length - next));
    }
  };
  await Promise.all(Array.from({ length: Math.min(config.concurrency, rows.length) }, worker));
  await adapters.flush(config.checkpointPath, config.outputDirectory, config.splitName);
  return { completed, fallbacks, aborted };
}

export function isValid(d: any): d is Distillation {
  return (
    d &&
    typeof d === 'object' &&
    d.fault_location &&
    typeof d.fault_location === 'object' &&
    typeof d.fault_location.product === 'string' &&
    typeof d.fault_location.component === 'string' &&
    Array.isArray(d.symptoms) &&
    d.symptoms.every((s: unknown) => typeof s === 'string') &&
    Array.isArray(d.capabilities) &&
    d.capabilities.every((s: unknown) => typeof s === 'string')
  );
}

export function toDistilledText(d: Distillation): string {
  const fl = [d.fault_location.product, d.fault_location.component, d.fault_location.subsystem]
    .filter(Boolean)
    .join(' ');
  return `[FL] ${fl} [SY] ${d.symptoms.join(' ')} [CP] ${d.capabilities.join(' ')}`.trim();
}

export function extractJson(raw: string): unknown {
  let s = raw.trim();
  if (s.startsWith('```')) {
    s = s.replace(/^```(?:json)?/, '').replace(/```$/, '').trim();
  }

  const start = s.indexOf('{');
  if (start === -1) throw new Error('sin objeto JSON en la respuesta');

  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < s.length; i++) {
    const c = s[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (c === '\\') {
      escaped = true;
      continue;
    }
    if (c === '"') {
      inString = !inString;
      continue;
    }
    if (!inString) {
      if (c === '{') depth++;
      else if (c === '}') {
        depth--;
        if (depth === 0) {
          return JSON.parse(s.slice(start, i + 1));
        }
      }
    }
  }
  return JSON.parse(s.slice(start));
}

export class CircuitBreaker {
  private consecutiveFailures = 0;
  private readonly threshold: number;

  constructor(threshold: number = 10) {
    this.threshold = threshold;
  }

  recordSuccess() {
    this.consecutiveFailures = 0;
  }

  recordFailure(): boolean {
    this.consecutiveFailures++;
    return this.consecutiveFailures >= this.threshold;
  }

  get failures() {
    return this.consecutiveFailures;
  }
}

export interface DeepSeekRetryProfile {
  /** Total HTTP attempts. The default preserves the historical engine contract. */
  attempts?: number;
  /** Legacy runners sleep before each retry, with their original randomized formula. */
  backoff?: 'engine' | 'runner-legacy';
  /** Runner cache-miss accounting intentionally uses `reported || inferred`. */
  cacheMiss?: 'engine-nullish' | 'runner-legacy-or';
  retryEmptyResponse?: boolean;
  maxErrorBodyChars?: number;
  maxRetryErrorBodyChars?: number;
}

export interface DeepSeekDependencies {
  fetch?: typeof fetch;
  sleep?: (milliseconds: number) => Promise<void>;
  now?: () => number;
  random?: () => number;
}

export async function callDeepSeekWithRetry(options: {
  apiKey: string;
  baseUrl: string;
  model: string;
  systemMessage: string;
  userPrompt: string;
  enableThinking?: boolean;
  maxRetries?: number;
  profile?: DeepSeekRetryProfile;
  dependencies?: DeepSeekDependencies;
}): Promise<DeepSeekResult> {
  const { apiKey, baseUrl, model, systemMessage, userPrompt, enableThinking = false, maxRetries = 4 } = options;
  const profile = options.profile ?? {};
  const deps = options.dependencies ?? {};
  const request = deps.fetch ?? globalThis.fetch;
  const sleep = deps.sleep ?? ((milliseconds: number) => Bun.sleep(milliseconds));
  const now = deps.now ?? (() => performance.now());
  const random = deps.random ?? Math.random;
  const attempts = profile.attempts ?? maxRetries;
  let delay = 1000;
  let lastErr: unknown = new Error('DeepSeek request failed');

  for (let attempt = 0; attempt < attempts; attempt++) {
    if (profile.backoff === 'runner-legacy' && attempt > 0) {
      await sleep(Math.min(1000 * 2 ** attempt + Math.floor(random() * 500), 10000));
    }
    const t0 = now();
    try {
      const bodyPayload: Record<string, unknown> = {
        model,
        messages: [{ role: 'system', content: systemMessage }, { role: 'user', content: userPrompt }],
        temperature: 0.0,
        response_format: { type: 'json_object' },
      };
      if (profile.backoff === 'runner-legacy' || !enableThinking) {
        bodyPayload.thinking = { type: enableThinking ? 'enabled' : 'disabled' };
      }
      const res = await request(`${baseUrl.replace(/\/+$/, '')}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify(bodyPayload),
      });
      const latencyMs = Math.round(now() - t0);
      if (res.status === 429 || res.status >= 500) {
        const errText = await res.text().catch(() => '');
        const maximum = profile.maxRetryErrorBodyChars ?? profile.maxErrorBodyChars;
        lastErr = new Error(`HTTP ${res.status}: ${maximum === undefined ? errText : errText.slice(0, maximum)}`);
        if (profile.backoff !== 'runner-legacy') await sleep(delay);
        delay *= 2;
        continue;
      }
      if (!res.ok) {
        const errText = await res.text();
        const maximum = profile.maxErrorBodyChars;
        throw new Error(`HTTP ${res.status}: ${maximum === undefined ? errText : errText.slice(0, maximum)}`);
      }
      const data: any = await res.json();
      const content = data.choices?.[0]?.message?.content ?? '';
      if (profile.retryEmptyResponse && !content) throw new Error('Respuesta de DeepSeek vacía o sin contenido');
      const usage = data.usage ?? {};
      const prompt = usage.prompt_tokens ?? 0;
      const hit = usage.prompt_cache_hit_tokens ?? 0;
      const inferredMiss = Math.max(0, prompt - hit);
      const reportedMiss = usage.prompt_cache_miss_tokens;
      return {
        text: content,
        usage: {
          prompt_tokens: prompt,
          completion_tokens: usage.completion_tokens ?? 0,
          cache_hit_tokens: hit,
          cache_miss_tokens: profile.cacheMiss === 'runner-legacy-or'
            ? (reportedMiss || inferredMiss)
            : (reportedMiss ?? prompt),
          latency_ms: latencyMs,
        },
      };
    } catch (err) {
      lastErr = err;
      if (attempt + 1 < attempts && profile.backoff !== 'runner-legacy') {
        await sleep(delay);
        delay *= 2;
      }
    }
  }
  throw lastErr;
}
