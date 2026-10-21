// Destilación semántica del corpus de Eclipse OpenJ9 usando DeepSeek API (deepseek-flash).
//
// Implementa el nuevo prompt especializado en JVM / Runtime Engineering (análogo al nuevo prompt de TypeScript).
//
// Características:
//   - Conexión directa HTTP a DeepSeek API vía Bun fetch (ultrarrápido, ~700ms por issue)
//   - Prompt especializado en arquitectura JVM (VM, JIT, GC, JITServer, JVMTI, Port, Build, Test)
//   - response_format: { type: 'json_object' } para garantizar salida JSON válida
//   - thinking: { type: 'disabled' } para máxima velocidad y ahorro de tokens
//   - Prefix context caching optimizado (system prompt con few-shots fijos para tarifa Cache Hit)
//   - Checkpoints en streaming (.jsonl) y persistencia segura
//   - Generación periódica de CSVs y Parquets canónicos (incluyendo Registro de Constancia de 11 columnas)
//   - Idempotente y reanudable
//
// Uso (para ejecutar cuando el usuario lo indique):
//   bun run scripts/distill_deepseek_openj9.ts --split train
//   bun run scripts/distill_deepseek_openj9.ts --split test
//   bun run scripts/distill_deepseek_openj9.ts --limit 5  (prueba corta)

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

const MAX_BODY_CHARS = 1500;

// Precios por 1M tokens (deepseek-flash)
const PRICE_CACHE_HIT_PER_1M = 0.006;  // Peak: $0.006 / Off-peak: $0.003
const PRICE_CACHE_MISS_PER_1M = 0.30;  // Peak: $0.30 / Off-peak: $0.15
const PRICE_OUTPUT_PER_1M = 1.20;      // Peak: $1.20 / Off-peak: $0.60

// ============================================================================
// PROMPT ESPECIALIZADO EN OPENJ9 / JVM RUNTIME (Análogo al de TypeScript)
// ============================================================================
const SYSTEM_PROMPT = `You are an expert Java Virtual Machine (JVM) and runtime systems engineer who reads Eclipse OpenJ9 issue reports and extracts structured technical information for developer triage.

Return ONLY a raw JSON object — no markdown, no code blocks, no extra text.
The JSON must have exactly this structure:
{
  "fault_location": {
    "product": "OpenJ9",
    "component": "<canonical JVM component: VM, JIT, GC, JITServer, JVMTI, Port, Build, Test, or JCL Extensions>",
    "subsystem": "<specific JVM runtime mechanism, max 5 words, e.g. compressed references heap, escape analysis inlining, concurrent scavenge>"
  },
  "symptoms": ["<short observed defect, crash or error, max 6 words>", "..."],
  "capabilities": ["<concrete JVM engineering skill required to fix, max 5 words>", "..."]
}
Rules:
- Do NOT attempt to fix, debug, or write code. Only classify and extract triage metadata.
- fault_location.product: must be "OpenJ9".
- fault_location.component: if provided in input (e.g. comp:vm -> VM, comp:jit -> JIT, comp:gc -> GC), normalize it. If missing or generic, classify into one of the canonical components:
    * VM (runtime, bytecode interpreter, class loading, threading, monitors, JNI)
    * JIT (Just-In-Time compiler, tree evaluation, optimizer, code generation, AOT)
    * GC (Garbage Collector, heap management, allocation, compaction, memory policies)
    * JITServer (out-of-process remote JIT compilation client/server)
    * JVMTI (JVM Tool Interface, debugging, agents, profiling)
    * Port (portability library, OS abstraction, signals, memory mapping)
    * Build (makefiles, CMake, UMA, toolchain configuration)
    * Test (test suites, sanity, regression frameworks)
    * JCL Extensions (Java Class Library native bindings and extensions)
- fault_location.subsystem: the specific JVM subsystem or algorithm involved.
- symptoms: 1 to 5 concise items describing the observed failure (e.g. SIGSEGV, OutOfMemoryError, slow pause, dead lock, assertion failure).
- capabilities: 1 to 5 concrete JVM systems engineering skills needed to resolve the issue (e.g. "JIT register allocation", "generational GC tuning", "bytecode verification", "native thread synchronization"). Avoid generic labels like "Java" or "debugging".
- If information is insufficient, use empty list [] or empty string "".`;

const FEW_SHOT_EXAMPLES = [
  {
    input: `Summary: JVM crashes with SIGSEGV in compilation of Hotspot loop
Product: OpenJ9
Component: comp:jit
Initial report:
When running SPECjbb2015, JVM crashes with a segmentation fault during JIT compilation of a hot loop in method com.example.Workload.run(). The crash log shows failure in trees evaluator during escape analysis pass. Reproducible on Linux x86_64 with -Xjit:count=0.`,
    output: `{
  "fault_location": {
    "product": "OpenJ9",
    "component": "JIT",
    "subsystem": "escape analysis loop tree evaluation"
  },
  "symptoms": ["SIGSEGV crash during compilation", "segmentation fault in hot loop", "trees evaluator failure on x86_64"],
  "capabilities": ["JIT tree evaluator", "escape analysis optimization", "x86_64 code generation"]
}`,
  },
  {
    input: `Summary: Multi-second GC pause spikes due to heap fragmentation with Balanced GC
Product: OpenJ9
Component: comp:gc
Initial report:
Under high allocation pressure with -Xgcpolicy:balanced, we observe multi-second pause times during PGC (Partial Garbage Collection). GC verbose logs show frequent global mark phases and heap fragmentation in eden regions unable to satisfy large array allocations.`,
    output: `{
  "fault_location": {
    "product": "OpenJ9",
    "component": "GC",
    "subsystem": "balanced GC region allocation"
  },
  "symptoms": ["multi-second GC pause spikes", "heap fragmentation in eden regions", "large array allocation failure"],
  "capabilities": ["garbage collection tuning", "balanced GC policy", "heap memory compaction"]
}`,
  },
];

const shots = FEW_SHOT_EXAMPLES.map(
  (ex) => `Example Input:\n${ex.input}\n\nExample Output:\n${ex.output}`
).join('\n\n');
const SYSTEM_MESSAGE = `${SYSTEM_PROMPT}\n\n## Few-shot Examples\n\n${shots}`;

export interface OpenJ9IssueRow {
  issue_number: string;
  issue_title: string;
  issue_body: string;
  owner: string;
  component?: string;
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

export function normalizeOpenJ9Component(comp: string | undefined): string {
  const c = (comp || '').toLowerCase().trim();
  if (c.includes('jit:aot')) return 'JIT';
  if (c.includes('jitserver')) return 'JITServer';
  if (c.includes('jit')) return 'JIT';
  if (c.includes('gc')) return 'GC';
  if (c.includes('jvmti')) return 'JVMTI';
  if (c.includes('vm')) return 'VM';
  if (c.includes('port')) return 'Port';
  if (c.includes('build')) return 'Build';
  if (c.includes('test')) return 'Test';
  if (c.includes('jclextensions')) return 'JCL Extensions';
  if (c.includes('infra')) return 'Build';
  return comp?.trim() || '(infer canonical JVM component)';
}

export function buildInput(row: OpenJ9IssueRow): string {
  const comp = normalizeOpenJ9Component(row.component);
  const parts = [
    `Summary: ${row.issue_title ?? ''}`,
    'Product: OpenJ9',
    `Component: ${comp}`,
    'Initial report:',
    (row.issue_body ?? '').slice(0, MAX_BODY_CHARS),
  ];
  return parts.join('\n');
}

export function fallback(row: OpenJ9IssueRow): Distillation {
  return {
    fault_location: {
      product: 'OpenJ9',
      component: normalizeOpenJ9Component(row.component) || 'VM',
      subsystem: '',
    },
    symptoms: [],
    capabilities: [],
    _fallback: true,
  };
}

// ============================================================================
// PERSISTENCIA Y CHECKPOINTS (Compatible con TriagerX y Tabla de Constancia)
// ============================================================================
import {
  escapeCsvField as sharedEscapeCsvField, writeCsvFile as sharedWriteCsvFile,
  loadExistingCheckpoint as sharedLoadExistingCheckpoint, flushOutputs as sharedFlushOutputs,
  safeAppendRecord as sharedSafeAppendRecord, calculateCostUsd as sharedCalculateCostUsd,
} from './deepseek_persistence';

export const escapeCsvField = sharedEscapeCsvField;
export const writeCsvFile = sharedWriteCsvFile;
export function loadExistingCheckpoint(ckptPath: string): Map<string, CheckpointRecord> { return sharedLoadExistingCheckpoint<CheckpointRecord>(ckptPath); }
export function flushOutputs(ckptPath: string, outDir: string, splitName: string, convertParquet?: (command: string, args: string[]) => Promise<void>) { return sharedFlushOutputs(ckptPath, outDir, splitName, CONFIG.model, (split) => `openj9_${split}`, convertParquet); }
export function safeAppendRecord(ckptPath: string, record: CheckpointRecord) { return sharedSafeAppendRecord(ckptPath, record); }
export function calculateCostUsd(cacheHit: number, cacheMiss: number, outputTokens: number): number { return sharedCalculateCostUsd(cacheHit, cacheMiss, outputTokens, { hit: PRICE_CACHE_HIT_PER_1M, miss: PRICE_CACHE_MISS_PER_1M, output: PRICE_OUTPUT_PER_1M }); }

// ============================================================================
// MAIN
// ============================================================================
export interface OpenJ9RunnerConfig {
  apiKey?: string;
  baseUrl: string;
  model: string;
  concurrency: number;
  flushEvery: number;
  breakerThreshold?: number;
  outputDirectory: string;
  csvDirectory: string;
  split: string;
  checkpointPath?: string;
  csvPath?: string;
  extraCheckpoints?: string[];
  offset: number;
  limit: number;
  dryRun: boolean;
}
export const CONFIG: OpenJ9RunnerConfig = {
  baseUrl: 'https://api.deepseek.com', model: 'deepseek-flash', concurrency: 5, flushEvery: 10,
  outputDirectory: 'artifacts/openj9/distilled', csvDirectory: 'artifacts/openj9',
  split: 'train', extraCheckpoints: undefined, offset: 0, limit: 0, dryRun: false,
};
export interface OpenJ9RunnerDependencies {
  rows?: readonly OpenJ9IssueRow[]; fetch?: typeof fetch; sleep?: (ms: number) => Promise<void>;
  random?: () => number; now?: () => number; readRows?: (path: string) => Promise<readonly OpenJ9IssueRow[]>; convertParquet?: (command: string, args: string[]) => Promise<void>;
}
export type OpenJ9RunnerResult = CorpusRunResult<CheckpointRecord>;
export function configFromArgs(argv: string[], base: OpenJ9RunnerConfig = CONFIG): OpenJ9RunnerConfig {
  const { values } = parseArgs({
    args: argv,
    options: {
      split: { type: 'string' }, limit: { type: 'string' }, offset: { type: 'string' },
      concurrency: { type: 'string' },
      flushEvery: { type: 'string' }, 'flush-every': { type: 'string' },
      outDir: { type: 'string' }, 'out-dir': { type: 'string' },
      csvDir: { type: 'string' }, 'csv-dir': { type: 'string' },
      model: { type: 'string' }, m: { type: 'string' },
      apiKey: { type: 'string' }, 'api-key': { type: 'string' },
      baseUrl: { type: 'string' }, 'base-url': { type: 'string' },
      extraCheckpoints: { type: 'string' }, 'extra-checkpoints': { type: 'string' },
      maxConsecutiveFailures: { type: 'string' }, 'max-consecutive-failures': { type: 'string' },
      dryRun: { type: 'boolean' }, 'dry-run': { type: 'boolean' },
    },
    strict: false,
  });
  const v = values as Record<string, string | boolean | undefined>;
  const split = String(v.split ?? base.split);
  const csvDirectory = String(v['csv-dir'] ?? v.csvDir ?? base.csvDirectory);
  const concurrency = Number(v.concurrency ?? base.concurrency);
  const extra = v['extra-checkpoints'] ?? v.extraCheckpoints;
  return {
    ...base,
    split,
    csvDirectory,
    concurrency,
    outputDirectory: String(v['out-dir'] ?? v.outDir ?? base.outputDirectory),
    csvPath: base.csvPath ?? join(csvDirectory, `openj9_${split}_50.csv`),
    limit: Number(v.limit ?? base.limit),
    offset: Number(v.offset ?? base.offset),
    flushEvery: Number(v['flush-every'] ?? v.flushEvery ?? base.flushEvery),
    breakerThreshold: Number(v['max-consecutive-failures'] ?? v.maxConsecutiveFailures ?? base.breakerThreshold ?? concurrency * 2),
    model: String(v.m ?? v.model ?? base.model),
    apiKey: (v['api-key'] ?? v.apiKey ?? base.apiKey) as string | undefined,
    baseUrl: String(v['base-url'] ?? v.baseUrl ?? base.baseUrl).replace(/\/+$/, ''),
    dryRun: Boolean(v['dry-run'] ?? v.dryRun ?? base.dryRun),
    extraCheckpoints: extra === undefined
      ? base.extraCheckpoints
      : String(extra).split(',').map((path) => path.trim()).filter(Boolean),
  };
}
export async function run(
  config: OpenJ9RunnerConfig = CONFIG,
  deps: OpenJ9RunnerDependencies = {},
): Promise<OpenJ9RunnerResult> {
  const breakerThreshold = validateRunConfig(config);
  const csvPath = config.csvPath ?? join(config.csvDirectory, `openj9_${config.split}_50.csv`);
  const allRows = deps.rows ?? await (deps.readRows ?? loadRows)(csvPath);
  // OpenJ9 intentionally retains duplicate rows within the selected range.
  const rows = config.limit > 0
    ? allRows.slice(config.offset, config.offset + config.limit)
    : allRows.slice(config.offset);
  const checkpointPath = config.checkpointPath ?? join(config.outputDirectory, `openj9_${config.split}.jsonl`);
  const apiKey = config.dryRun ? '' : (config.apiKey ?? process.env.DEEPSEEK_API_KEY ?? '').trim();
  if (!config.dryRun && !apiKey) throw new Error('DEEPSEEK_API_KEY no encontrada');
  if (!config.dryRun) mkdirSync(config.outputDirectory, { recursive: true });
  const distiller = createDistiller<OpenJ9IssueRow>({
    apiKey,
    baseUrl: config.baseUrl,
    model: config.model,
    systemMessage: SYSTEM_MESSAGE,
    thinking: false,
    userPrompt: (row) => `Now extract the structured information for this OpenJ9 issue report:\n\n${buildInput(row)}`,
    fallback,
    dependencies: deps,
  });
  return runCorpusDistillation({
    rows,
    config: {
      concurrency: config.concurrency,
      checkpointPath,
      flushEvery: config.flushEvery,
      breakerThreshold,
      outputDirectory: config.outputDirectory,
      splitName: config.split,
      extraCheckpoints: config.extraCheckpoints,
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
        raw_text: (row.text || `${row.issue_title} ${row.issue_body || ''}`).trim(),
        fallback: Boolean(outcome.distillation._fallback),
        latency_ms: latency,
        model: config.model,
        prompt_tokens: outcome.promptTokens,
        completion_tokens: outcome.completionTokens,
        cache_hit_tokens: outcome.cacheHitTokens,
      }),
      prices: { hit: PRICE_CACHE_HIT_PER_1M, miss: PRICE_CACHE_MISS_PER_1M, output: PRICE_OUTPUT_PER_1M },
      append: safeAppendRecord,
      flush: (path, out, name) => sharedFlushOutputs(path, out, name, config.model, (s) => `openj9_${s}`, deps.convertParquet),
      sleep: deps.sleep,
      now: deps.now,
      onFallback: (failures, row) => console.warn(`[Fallo ${failures}/${breakerThreshold}] Issue #${row.issue_number} terminó en fallback.`),
      onBreaker: (failures) => console.error(`[CIRCUIT BREAKER] ${failures} fallos consecutivos; abortando.`),
    },
  });
}
async function loadRows(path: string): Promise<readonly OpenJ9IssueRow[]> {
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
