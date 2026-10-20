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

export async function callDeepSeekWithRetry(options: {
  apiKey: string;
  baseUrl: string;
  model: string;
  systemMessage: string;
  userPrompt: string;
  enableThinking?: boolean;
  maxRetries?: number;
}): Promise<DeepSeekResult> {
  const {
    apiKey,
    baseUrl,
    model,
    systemMessage,
    userPrompt,
    enableThinking = false,
    maxRetries = 4,
  } = options;

  let delay = 1000;
  let lastErr: any = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    const t0 = performance.now();
    try {
      const bodyPayload: any = {
        model,
        messages: [
          { role: 'system', content: systemMessage },
          { role: 'user', content: userPrompt },
        ],
        temperature: 0.0,
        response_format: { type: 'json_object' },
      };

      if (!enableThinking) {
        bodyPayload.thinking = { type: 'disabled' };
      }

      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(bodyPayload),
      });

      const latency_ms = Math.round(performance.now() - t0);

      if (res.status === 429 || res.status >= 500) {
        const errText = await res.text();
        lastErr = new Error(`HTTP ${res.status}: ${errText}`);
        await Bun.sleep(delay);
        delay *= 2;
        continue;
      }

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`HTTP ${res.status}: ${errText}`);
      }

      const data: any = await res.json();
      const content = data.choices?.[0]?.message?.content ?? '';
      const usage = data.usage ?? {};

      return {
        text: content,
        usage: {
          prompt_tokens: usage.prompt_tokens ?? 0,
          completion_tokens: usage.completion_tokens ?? 0,
          cache_hit_tokens: usage.prompt_cache_hit_tokens ?? 0,
          cache_miss_tokens: usage.prompt_cache_miss_tokens ?? (usage.prompt_tokens ?? 0),
          latency_ms,
        },
      };
    } catch (err: any) {
      lastErr = err;
      if (attempt < maxRetries) {
        await Bun.sleep(delay);
        delay *= 2;
      }
    }
  }

  throw lastErr;
}
