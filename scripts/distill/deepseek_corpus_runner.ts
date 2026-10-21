import {
  callDeepSeekWithRetry,
  extractJson,
  isValid,
  runBatchDistillation,
  type BatchDistillationAdapters,
  type BatchDistillationConfig,
  type DeepSeekDependencies,
  type Distillation,
} from './deepseek_engine';
import { calculateCostUsd, loadExistingCheckpoint } from './deepseek_persistence';

export interface CorpusRunConfig extends BatchDistillationConfig {
  extraCheckpoints?: readonly string[];
  dryRun?: boolean;
}

export interface RunnerPreview {
  system: string;
  user: string;
  body: {
    model: string;
    response_format: { type: 'json_object' };
    temperature: number;
    thinking: { type: 'enabled' | 'disabled' };
  };
}

export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  cacheHitTokens: number;
  cacheMissTokens: number;
}

export interface DistillationOutcome {
  distillation: Distillation;
  promptTokens: number;
  completionTokens: number;
  cacheHitTokens: number;
}

export interface CorpusRunResult<RecordType> {
  summary: { completed: number; fallbacks: number; aborted: boolean; skipped: number };
  statistics: TokenUsage & { costUsd: number };
  paths: { checkpoint: string; outputDirectory: string; split: string };
  records: RecordType[];
  preview?: RunnerPreview;
}

/** Validate before loading a corpus or creating output directories. */
export function validateRunConfig(config: {
  concurrency: number;
  flushEvery: number;
  breakerThreshold?: number;
}): number {
  for (const [name, value] of Object.entries({
    concurrency: config.concurrency,
    flushEvery: config.flushEvery,
    breakerThreshold: config.breakerThreshold ?? config.concurrency * 2,
  })) {
    if (!Number.isInteger(value) || value <= 0) {
      throw new Error(`${name} must be an integer > 0`);
    }
  }
  return config.breakerThreshold ?? config.concurrency * 2;
}

/** One transport/validation/fallback path, with accounting isolated per invocation.
 * Successful HTTP usage counts even if JSON/schema validation falls back; the
 * fallback checkpoint token fields remain zero, as in the original runners.
 */
export function createDistiller<Row>(options: {
  apiKey: string;
  baseUrl: string;
  model: string;
  systemMessage: string;
  thinking: boolean;
  userPrompt(row: Row): string;
  fallback(row: Row): Distillation;
  dependencies?: DeepSeekDependencies;
}) {
  const totals: TokenUsage = {
    promptTokens: 0, completionTokens: 0, cacheHitTokens: 0, cacheMissTokens: 0,
  };
  return {
    usage: (): TokenUsage => ({ ...totals }),
    preview(row: Row): RunnerPreview {
      return {
        system: options.systemMessage,
        user: options.userPrompt(row),
        body: {
          model: options.model,
          response_format: { type: 'json_object' },
          temperature: 0,
          thinking: { type: options.thinking ? 'enabled' : 'disabled' },
        },
      };
    },
    async distill(row: Row): Promise<DistillationOutcome> {
      try {
        const response = await callDeepSeekWithRetry({
          apiKey: options.apiKey,
          baseUrl: options.baseUrl,
          model: options.model,
          systemMessage: options.systemMessage,
          userPrompt: options.userPrompt(row),
          enableThinking: options.thinking,
          profile: {
            attempts: 5,
            backoff: 'runner-legacy',
            cacheMiss: 'runner-legacy-or',
            retryEmptyResponse: true,
            maxErrorBodyChars: 300,
            maxRetryErrorBodyChars: 200,
          },
          dependencies: options.dependencies,
        });
        totals.promptTokens += response.usage.prompt_tokens;
        totals.completionTokens += response.usage.completion_tokens;
        totals.cacheHitTokens += response.usage.cache_hit_tokens;
        totals.cacheMissTokens += response.usage.cache_miss_tokens;
        const parsed = extractJson(response.text);
        if (isValid(parsed)) {
          return {
            distillation: parsed,
            promptTokens: response.usage.prompt_tokens,
            completionTokens: response.usage.completion_tokens,
            cacheHitTokens: response.usage.cache_hit_tokens,
          };
        }
        console.error('JSON válido pero con esquema inesperado');
      } catch (error) {
        console.error(`Error destilando: ${(error as Error).message}`);
      }
      return {
        distillation: options.fallback(row),
        promptTokens: 0, completionTokens: 0, cacheHitTokens: 0,
      };
    },
  };
}

export async function runCorpusDistillation<Row, RecordType extends { issue_number?: string | number }>(options: {
  rows: readonly Row[];
  config: CorpusRunConfig;
  preview?(row: Row): RunnerPreview;
  adapters: BatchDistillationAdapters<Row, RecordType> & {
    prices: { hit: number; miss: number; output: number };
    usage(): TokenUsage;
  };
}): Promise<CorpusRunResult<RecordType>> {
  const { rows, config, adapters } = options;
  validateRunConfig(config);
  const completedIds = new Set<string>();
  for (const path of [config.checkpointPath, ...(config.extraCheckpoints ?? [])]) {
    for (const id of loadExistingCheckpoint(path).keys()) completedIds.add(id);
  }
  const pending = rows.filter((row) => !completedIds.has(adapters.id(row)));
  const paths = {
    checkpoint: config.checkpointPath,
    outputDirectory: config.outputDirectory,
    split: config.splitName,
  };
  if (config.dryRun) {
    return {
      summary: { completed: 0, fallbacks: 0, aborted: false, skipped: rows.length - pending.length },
      statistics: { promptTokens: 0, completionTokens: 0, cacheHitTokens: 0, cacheMissTokens: 0, costUsd: 0 },
      paths,
      records: [],
      preview: pending.length ? options.preview?.(pending[0]) : undefined,
    };
  }
  const batch = await runBatchDistillation(pending, config, adapters);
  const records = Array.from(loadExistingCheckpoint<RecordType>(config.checkpointPath).values());
  const usage = adapters.usage();
  return {
    summary: { ...batch, skipped: rows.length - pending.length },
    statistics: {
      ...usage,
      costUsd: calculateCostUsd(usage.cacheHitTokens, usage.cacheMissTokens, usage.completionTokens, adapters.prices),
    },
    paths,
    records,
  };
}
