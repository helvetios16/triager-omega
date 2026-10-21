// Destilación de issues de TypeScript usando la API de DeepSeek (deepseek-flash).
//
// Características:
//   - Conexión directa HTTP OpenAI-compatible (rápido, sin overhead de CLI subprocess)
//   - Concurrencia configurable (default: 8 workers)
//   - Soporte para Thinking Mode activado/desactivado (default: desactivado para ahorro y velocidad)
//   - response_format: { type: 'json_object' } para garantizar salida JSON válida
//   - Registro y métricas de tokens (cache hits, cache misses, output) y cálculo de costo en tiempo real
//   - Reintentos con exponential backoff para 429 y 5xx
//   - Circuit breaker configurable para abortar en caso de errores continuos
//   - Checkpoints en streaming (.jsonl) con mutex asíncrono
//   - Flush periódico a Parquet y CSVs (distilled, raw, both, provenance) compatible con el pipeline
//   - Idempotente y reanudable con soporte para --extra-checkpoints
//
// Ejecución de prueba:
//   bun run scripts/distill_deepseek_ts.ts --limit 3
//
// Ejecución completa (omitiendo los ya procesados en ts_train_union.jsonl):
//   bun run scripts/distill_deepseek_ts.ts --extra-checkpoints artifacts/typescript/distilled/ts_train_union.jsonl

import { mkdirSync } from 'fs';
import { join } from 'path';
import { parseArgs } from 'util';
import {
  Distillation,
  toDistilledText,
} from './deepseek_engine';
import {
  createDistiller, runCorpusDistillation, validateRunConfig,
  type CorpusRunResult,
} from './deepseek_corpus_runner';

const MAX_COMMENT_CHARS = 1500;

// Precios por 1M tokens (deepseek-flash, asumiendo peak como cota conservadora)
const PRICE_CACHE_HIT_PER_1M = 0.006;  // Peak: $0.006 / Off-peak: $0.003
const PRICE_CACHE_MISS_PER_1M = 0.30;  // Peak: $0.30 / Off-peak: $0.15
const PRICE_OUTPUT_PER_1M = 1.20;      // Peak: $1.20 / Off-peak: $0.60

// Prompt refinado para el compilador de TypeScript
const SYSTEM_PROMPT = `You are an expert compiler engineer who reads issue reports and extracts structured technical information for developer triage.

Return ONLY a raw JSON object — no markdown, no code blocks, no extra text.
The JSON must have exactly this structure:
{
  "fault_location": {
    "product": "<exact product name from input, e.g. TypeScript>",
    "component": "<canonical compiler component, e.g. Type Checker, Emitter, Parser, Language Service, Binder, Module Resolution, CLI>",
    "subsystem": "<specific compiler mechanism or feature, max 5 words, e.g. union type assignability, downlevel async transform>"
  },
  "symptoms": ["<short observed symptom or defect, max 6 words>", "..."],
  "capabilities": ["<concrete technical compiler skill required to fix, max 5 words>", "..."]
}
Rules:
- Do NOT attempt to fix, debug, or write code. Only analyze and classify the metadata from the report.
- fault_location.product: must be copied exactly from the input.
- fault_location.component: if provided in input, use it. If empty or '(infer canonical compiler component)', classify into one of the canonical compiler components: Type Checker, Emitter, Parser, Language Service, Binder, Module Resolution, or CLI.
- fault_location.subsystem: the specific language construct or compiler pass involved.
- symptoms: 1 to 5 concise items describing the observed failure, unexpected error, or incorrect code generation.
- capabilities: 1 to 5 concrete compiler engineering skills needed to resolve the issue (e.g. "generic type inference", "AST transformation", "sourcemap generation"). Avoid generic labels like "TypeScript" or "debugging".
- If information is insufficient, use empty list [] or empty string "".`;

const FEW_SHOT_EXAMPLES = [
  {
    input: `Summary: [Errors] Type argument inference from signatures infers fewer candidates
Product: TypeScript
Component: (infer canonical compiler component)
Labels: Bug, Fixed, Spec
Initial report:
When calling a generic function with overloaded signatures, type argument inference fails to consider candidates from subsequent overload signatures, resulting in TS2345 (Argument of type 'X' not assignable to parameter of type 'Y').`,
    output: `{
  "fault_location": {
    "product": "TypeScript",
    "component": "Type Checker",
    "subsystem": "overload type argument inference"
  },
  "symptoms": ["type argument inference fails", "candidates missing from overloads", "TS2345 spurious error"],
  "capabilities": ["generic type inference", "overload resolution", "signature candidate collection"]
}`,
  },
  {
    input: `Summary: [Emit] "use strict" emitted after __extends instead of before
Product: TypeScript
Component: (infer canonical compiler component)
Labels: Bug, Fixed
Initial report:
When compiling an ES6 class hierarchy down to ES5 with --target ES5, the compiler outputs the runtime helper '__extends(Derived, Base)' before the '"use strict";' directive in the generated file header, violating prologue directive ordering.`,
    output: `{
  "fault_location": {
    "product": "TypeScript",
    "component": "Emitter",
    "subsystem": "ES5 helper prologue emission"
  },
  "symptoms": ["use strict emitted after helpers", "prologue directive ordering violated", "strict mode not applied in ES5"],
  "capabilities": ["AST downlevel emitter", "prologue directive handling", "runtime helper generation"]
}`,
  },
];

export interface TsIssueRow {
  issue_number: string;
  issue_title: string;
  description: string;
  component?: string;
  labels?: string;
  owner?: string;
  text?: string;
}

export interface CheckpointRecord {
  issue_number: string;
  owner: string;
  distilled_json: string;
  distilled_text: string;
  raw_text: string;
  fallback: boolean;
  latency_ms: number;
  model?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  cache_hit_tokens?: number;
}

export function buildInput(row: TsIssueRow): string {
  const comp = (row.component || '').trim();
  const parts = [
    `Summary: ${row.issue_title ?? ''}`,
    'Product: TypeScript',
    `Component: ${comp ? comp : '(infer canonical compiler component)'}`,
  ];
  if (row.labels?.trim()) {
    parts.push(`Labels: ${row.labels.trim()}`);
  }
  parts.push('Initial report:');
  parts.push((row.description ?? '').slice(0, MAX_COMMENT_CHARS));
  return parts.join('\n');
}

// Prefijo fijo en el system prompt para que DeepSeek active context cache automáticamente
const shots = FEW_SHOT_EXAMPLES.map(
  (ex) => `Example Input:\n${ex.input}\n\nExample Output:\n${ex.output}`
).join('\n\n');
const SYSTEM_MESSAGE = `${SYSTEM_PROMPT}\n\n## Few-shot Examples\n\n${shots}`;

export function fallback(row: TsIssueRow): Distillation {
  return {
    fault_location: {
      product: 'TypeScript',
      component: row.component?.trim() || 'Compiler',
      subsystem: '',
    },
    symptoms: [],
    capabilities: [],
    _fallback: true,
  };
}

// ============================================================================
// PERSISTENCIA Y CHECKPOINTS
// ============================================================================
import {
  escapeCsvField as sharedEscapeCsvField, writeCsvFile as sharedWriteCsvFile,
  loadExistingCheckpoint as sharedLoadExistingCheckpoint, flushOutputs as sharedFlushOutputs,
  safeAppendRecord as sharedSafeAppendRecord, calculateCostUsd as sharedCalculateCostUsd,
} from './deepseek_persistence';

export const escapeCsvField = sharedEscapeCsvField;
export const writeCsvFile = sharedWriteCsvFile;
export function loadExistingCheckpoint(ckptPath: string): Map<string, CheckpointRecord> { return sharedLoadExistingCheckpoint<CheckpointRecord>(ckptPath); }
export function flushOutputs(ckptPath: string, outDir: string, splitName: string, convertParquet?: (command: string, args: string[]) => Promise<void>) { return sharedFlushOutputs(ckptPath, outDir, splitName, CONFIG.model, (split) => split, convertParquet); }
export function safeAppendRecord(ckptPath: string, record: CheckpointRecord) { return sharedSafeAppendRecord(ckptPath, record); }
export function calculateCostUsd(cacheHit: number, cacheMiss: number, outputTokens: number): number { return sharedCalculateCostUsd(cacheHit, cacheMiss, outputTokens, { hit: PRICE_CACHE_HIT_PER_1M, miss: PRICE_CACHE_MISS_PER_1M, output: PRICE_OUTPUT_PER_1M }); }

// ============================================================================
// MAIN
// ============================================================================
export interface TsRunnerConfig {
  apiKey?: string;
  baseUrl: string;
  model: string;
  concurrency: number;
  flushEvery: number;
  breakerThreshold?: number;
  outputDirectory: string;
  split: string;
  checkpointPath?: string;
  csvPath?: string;
  extraCheckpoints?: string[];
  offset: number;
  limit: number;
  reverse: boolean;
  dryRun: boolean;
  minDelayMs: number;
  maxDelayMs: number;
  thinking: boolean;
}
export const CONFIG: TsRunnerConfig = {
  baseUrl: 'https://api.deepseek.com', model: 'deepseek-flash', concurrency: 8, flushEvery: 20,
  outputDirectory: 'artifacts/typescript/distilled', split: 'train',
  extraCheckpoints: undefined,
  offset: 0, limit: 0, reverse: false, dryRun: false, minDelayMs: 0, maxDelayMs: 0, thinking: false,
};
export interface RunnerDependencies {
  rows?: readonly TsIssueRow[]; fetch?: typeof fetch; sleep?: (ms: number) => Promise<void>;
  random?: () => number; now?: () => number; readRows?: (path: string) => Promise<readonly TsIssueRow[]>; convertParquet?: (command: string, args: string[]) => Promise<void>;
}
export type TsRunnerResult = CorpusRunResult<CheckpointRecord>;
export function configFromArgs(argv: string[], base: TsRunnerConfig = CONFIG): TsRunnerConfig {
  const { values } = parseArgs({
    args: argv,
    options: {
      limit: { type: 'string' }, offset: { type: 'string' },
      concurrency: { type: 'string' }, split: { type: 'string' },
      flushEvery: { type: 'string' }, 'flush-every': { type: 'string' },
      outDir: { type: 'string' }, 'out-dir': { type: 'string' },
      csv: { type: 'string' }, model: { type: 'string' }, m: { type: 'string' },
      apiKey: { type: 'string' }, 'api-key': { type: 'string' },
      baseUrl: { type: 'string' }, 'base-url': { type: 'string' },
      thinking: { type: 'boolean' }, reverse: { type: 'boolean' },
      extraCheckpoints: { type: 'string' }, 'extra-checkpoints': { type: 'string' },
      minDelay: { type: 'string' }, 'min-delay': { type: 'string' },
      maxDelay: { type: 'string' }, 'max-delay': { type: 'string' },
      maxConsecutiveFailures: { type: 'string' }, 'max-consecutive-failures': { type: 'string' },
      dryRun: { type: 'boolean' }, 'dry-run': { type: 'boolean' },
    },
    strict: false,
  });
  const v = values as Record<string, string | boolean | undefined>;
  const split = String(v.split ?? base.split);
  const concurrency = Number(v.concurrency ?? base.concurrency);
  const extra = v['extra-checkpoints'] ?? v.extraCheckpoints;
  return {
    ...base,
    split,
    concurrency,
    limit: Number(v.limit ?? base.limit),
    offset: Number(v.offset ?? base.offset),
    flushEvery: Number(v['flush-every'] ?? v.flushEvery ?? base.flushEvery),
    breakerThreshold: Number(v['max-consecutive-failures'] ?? v.maxConsecutiveFailures ?? base.breakerThreshold ?? concurrency * 2),
    outputDirectory: String(v['out-dir'] ?? v.outDir ?? base.outputDirectory),
    csvPath: String(v.csv ?? base.csvPath ?? defaultCsv(split)),
    model: String(v.m ?? v.model ?? base.model),
    apiKey: (v['api-key'] ?? v.apiKey ?? base.apiKey) as string | undefined,
    baseUrl: String(v['base-url'] ?? v.baseUrl ?? base.baseUrl).replace(/\/+$/, ''),
    thinking: Boolean(v.thinking ?? base.thinking),
    reverse: Boolean(v.reverse ?? base.reverse),
    dryRun: Boolean(v['dry-run'] ?? v.dryRun ?? base.dryRun),
    extraCheckpoints: extra === undefined
      ? base.extraCheckpoints
      : String(extra).split(',').map((path) => path.trim()).filter(Boolean),
    minDelayMs: Math.round(Number(v['min-delay'] ?? v.minDelay ?? base.minDelayMs / 1000) * 1000),
    maxDelayMs: Math.round(Number(v['max-delay'] ?? v.maxDelay ?? base.maxDelayMs / 1000) * 1000),
  };
}
export async function run(
  config: TsRunnerConfig = CONFIG,
  deps: RunnerDependencies = {},
): Promise<TsRunnerResult> {
  const breakerThreshold = validateRunConfig(config);
  const isTest = config.split === 'test' || config.split === 'ts_test';
  const split = isTest ? 'ts_test' : config.split === 'train' ? 'ts_train_deepseek' : config.split;
  const checkpointPath = config.checkpointPath ?? join(config.outputDirectory, `${split}.jsonl`);
  const extraCheckpoints = config.extraCheckpoints ?? (isTest ? [] : [
    'artifacts/typescript/distilled/ts_train_union.jsonl',
    'artifacts/typescript/distilled/ts_train.jsonl',
  ]);
  const allRows = deps.rows ?? await (deps.readRows ?? loadRows)(config.csvPath ?? defaultCsv(config.split));
  const selected = config.limit > 0
    ? allRows.slice(config.offset, config.offset + config.limit)
    : allRows.slice(config.offset);
  const seen = new Set<string>();
  const rows = selected.filter((row) => {
    const id = String(row.issue_number);
    if (seen.has(id)) return false;
    seen.add(id);
    return true;
  });
  if (config.reverse) rows.reverse();

  const apiKey = config.dryRun ? '' : (config.apiKey ?? process.env.DEEPSEEK_API_KEY ?? '').trim();
  if (!config.dryRun && !apiKey) throw new Error('DEEPSEEK_API_KEY no encontrada');
  if (!config.dryRun) mkdirSync(config.outputDirectory, { recursive: true });
  const distiller = createDistiller<TsIssueRow>({
    apiKey,
    baseUrl: config.baseUrl,
    model: config.model,
    systemMessage: SYSTEM_MESSAGE,
    thinking: config.thinking,
    userPrompt: (row) => `Now extract the structured information for this issue report:\n\n${buildInput(row)}`,
    fallback,
    dependencies: deps,
  });
  const result = await runCorpusDistillation({
    rows,
    config: {
      concurrency: config.concurrency,
      checkpointPath,
      flushEvery: config.flushEvery,
      breakerThreshold,
      outputDirectory: config.outputDirectory,
      splitName: split,
      extraCheckpoints,
      dryRun: config.dryRun,
    },
    preview: distiller.preview,
    adapters: {
      id: (row) => String(row.issue_number),
      distill: distiller.distill,
      usage: distiller.usage,
      makeRecord: (row, outcome, latency) => ({
        issue_number: String(row.issue_number),
        owner: row.owner || '',
        distilled_json: JSON.stringify(outcome.distillation),
        distilled_text: toDistilledText(outcome.distillation),
        raw_text: (row.text || `${row.issue_title} ${row.description || ''}`).trim(),
        fallback: Boolean(outcome.distillation._fallback),
        latency_ms: latency,
        model: config.model,
        prompt_tokens: outcome.promptTokens,
        completion_tokens: outcome.completionTokens,
        cache_hit_tokens: outcome.cacheHitTokens,
      }),
      prices: { hit: PRICE_CACHE_HIT_PER_1M, miss: PRICE_CACHE_MISS_PER_1M, output: PRICE_OUTPUT_PER_1M },
      append: safeAppendRecord,
      flush: (path, out, name) => sharedFlushOutputs(path, out, name, config.model, (s) => s, deps.convertParquet),
      sleep: deps.sleep,
      now: deps.now,
      onFallback: (failures, row) => console.warn(`[Fallo ${failures}/${breakerThreshold}] Issue #${row.issue_number} terminó en fallback.`),
      onBreaker: (failures) => console.error(`[CIRCUIT BREAKER] ${failures} fallos consecutivos; abortando.`),
      afterRecord: async (_completed, _row, remaining) => {
        if (config.maxDelayMs > 0 && remaining > 0) {
          const span = Math.max(0, config.maxDelayMs - config.minDelayMs);
          const delay = config.minDelayMs + Math.floor((deps.random ?? Math.random)() * (span + 1));
          await (deps.sleep ?? Bun.sleep)(delay);
        }
      },
    },
  });
  if (!config.dryRun && config.maxDelayMs > 0) await (deps.sleep ?? Bun.sleep)(config.minDelayMs);
  return result;
}
function defaultCsv(split: string): string {
  return split === 'test' || split === 'ts_test'
    ? '/Users/sebastian/Documents/Python/triagerX/df_test_ts.csv'
    : '/Users/sebastian/Documents/Python/triagerX/df_train_ts.csv';
}

async function loadRows(path: string): Promise<readonly TsIssueRow[]> {
  const { parse } = await import('csv-parse/sync');
  return parse(await Bun.file(path).text(), { columns: true, skip_empty_lines: true });
}

if (import.meta.main) {
  const config = configFromArgs(Bun.argv.slice(2));
  run(config).then((result) => {
    console.log(`Destilación finalizada: ${result.summary.completed} registros; fallbacks=${result.summary.fallbacks}; skipped=${result.summary.skipped}; tokens=${result.statistics.promptTokens}in/${result.statistics.completionTokens}out; coste=$${result.statistics.costUsd.toFixed(4)}`);
    if (config.dryRun && result.preview) console.log(JSON.stringify(result.preview, null, 2));
    if (result.summary.aborted) process.exitCode = 1;
  }).catch((error) => {
    console.error(`Error fatal: ${(error as Error).message}`);
    process.exitCode = 1;
  });
}
