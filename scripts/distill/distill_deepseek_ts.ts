// Destilación de issues de TypeScript usando la API de DeepSeek (deepseek-flash).
//
// Características:
//   - Conexión directa HTTP OpenAI-compatible (rápido, sin overhead de CLI subprocess)
//   - Concurrencia configurable (default: 5 workers)
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

import { parse } from 'csv-parse/sync';
import { parseArgs } from 'util';
import { appendFileSync, existsSync, readFileSync } from 'fs';
import { join } from 'path';

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    limit: { type: 'string', default: '0' },
    offset: { type: 'string', default: '0' },
    concurrency: { type: 'string', default: '8' },
    flushEvery: { type: 'string', default: '20' },
    'flush-every': { type: 'string', default: '20' },
    split: { type: 'string', default: 'train' }, // 'test' | 'train' | custom
    outDir: { type: 'string', default: 'artifacts/typescript/distilled' },
    'out-dir': { type: 'string', default: 'artifacts/typescript/distilled' },
    csv: { type: 'string' },
    model: { type: 'string', default: 'deepseek-flash' },
    m: { type: 'string' },
    apiKey: { type: 'string' },
    'api-key': { type: 'string' },
    baseUrl: { type: 'string', default: 'https://api.deepseek.com' },
    'base-url': { type: 'string', default: 'https://api.deepseek.com' },
    thinking: { type: 'boolean', default: false },
    reverse: { type: 'boolean', default: false },
    extraCheckpoints: { type: 'string' },
    'extra-checkpoints': { type: 'string' },
    minDelay: { type: 'string', default: '0' },
    'min-delay': { type: 'string', default: '0' },
    maxDelay: { type: 'string', default: '0' },
    'max-delay': { type: 'string', default: '0' },
    maxConsecutiveFailures: { type: 'string' },
    'max-consecutive-failures': { type: 'string' },
    dryRun: { type: 'boolean', default: false },
    'dry-run': { type: 'boolean', default: false },
  },
  strict: false,
});

const LIMIT = parseInt(values.limit || '0', 10);
const OFFSET = parseInt(values.offset || '0', 10);
const CONCURRENCY = parseInt(values.concurrency || '8', 10);
const MAX_CONSECUTIVE_FAILURES = parseInt(
  (values['max-consecutive-failures'] || values.maxConsecutiveFailures || String(CONCURRENCY * 2)) as string,
  10
);
const FLUSH_EVERY = parseInt(values['flush-every'] || values.flushEvery || '20', 10);

const rawSplit = (values.split as string) || 'train';
const isTestSplit = rawSplit === 'test' || rawSplit === 'ts_test';
const SPLIT_NAME = isTestSplit ? 'ts_test' : (rawSplit === 'train' ? 'ts_train_deepseek' : rawSplit);
const OUT_DIR = (values['out-dir'] || values.outDir || 'artifacts/typescript/distilled') as string;
const CSV_PATH = (values.csv as string) || (isTestSplit
  ? '/Users/sebastian/Documents/Python/triagerX/df_test_ts.csv'
  : '/Users/sebastian/Documents/Python/triagerX/df_train_ts.csv');
const IS_REVERSE = Boolean(values.reverse);
const IS_DRY_RUN = Boolean(values['dry-run'] || values.dryRun);
const ENABLE_THINKING = Boolean(values.thinking);

const rawExtra = values['extra-checkpoints'] || values.extraCheckpoints;
const EXTRA_CHECKPOINTS = rawExtra !== undefined
  ? String(rawExtra).split(',').map((s) => s.trim()).filter(Boolean)
  : (isTestSplit ? [] : [
      'artifacts/typescript/distilled/ts_train_union.jsonl',
      'artifacts/typescript/distilled/ts_train.jsonl'
    ]);
const MIN_DELAY_SEC = parseFloat((values['min-delay'] || values.minDelay || '0') as string);
const MAX_DELAY_SEC = parseFloat((values['max-delay'] || values.maxDelay || '0') as string);
const MIN_DELAY_MS = Math.round(MIN_DELAY_SEC * 1000);
const MAX_DELAY_MS = Math.round(MAX_DELAY_SEC * 1000);
const MAX_COMMENT_CHARS = 1500;

// Configuración DeepSeek API
const API_KEY = (values['api-key'] || values.apiKey || process.env.DEEPSEEK_API_KEY || '').trim();
const BASE_URL = ((values['base-url'] || values.baseUrl || 'https://api.deepseek.com') as string).replace(/\/+$/, '');
const MODEL = (values.m as string) || (values.model as string) || 'deepseek-flash';

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

interface TsIssueRow {
  issue_number: string;
  issue_title: string;
  description: string;
  component?: string;
  labels?: string;
  owner?: string;
  text?: string;
}

interface Distillation {
  fault_location: { product: string; component: string; subsystem: string };
  symptoms: string[];
  capabilities: string[];
  _fallback?: boolean;
}

interface CheckpointRecord {
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

function buildInput(row: TsIssueRow): string {
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

function isValid(d: any): d is Distillation {
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

function fallback(row: TsIssueRow): Distillation {
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

function toDistilledText(d: Distillation): string {
  const fl = [d.fault_location.product, d.fault_location.component, d.fault_location.subsystem]
    .filter(Boolean)
    .join(' ');
  return `[FL] ${fl} [SY] ${d.symptoms.join(' ')} [CP] ${d.capabilities.join(' ')}`.trim();
}

function extractJson(raw: string): unknown {
  let s = raw.trim();
  if (s.startsWith('```')) s = s.replace(/^```(?:json)?/, '').replace(/```$/, '').trim();

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
    if (inString) continue;
    if (c === '{') depth++;
    else if (c === '}' && --depth === 0) return JSON.parse(s.slice(start, i + 1));
  }
  throw new Error('objeto JSON sin cerrar');
}

// Estadísticas acumuladas
let totalPromptTokens = 0;
let totalCompletionTokens = 0;
let totalCacheHitTokens = 0;
let totalCacheMissTokens = 0;

async function callDeepSeek(
  row: TsIssueRow,
  retries = 4
): Promise<{ text: string; promptTokens: number; completionTokens: number; cacheHitTokens: number }> {
  const url = `${BASE_URL}/chat/completions`;
  const userPrompt = `Now extract the structured information for this issue report:\n\n${buildInput(row)}`;

  const bodyPayload: Record<string, unknown> = {
    model: MODEL,
    messages: [
      { role: 'system', content: SYSTEM_MESSAGE },
      { role: 'user', content: userPrompt },
    ],
    response_format: { type: 'json_object' },
    temperature: 0.0,
    thinking: { type: ENABLE_THINKING ? 'enabled' : 'disabled' },
  };

  let lastError: Error | null = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    if (attempt > 0) {
      const backoffMs = Math.min(1000 * Math.pow(2, attempt) + Math.floor(Math.random() * 500), 10000);
      await Bun.sleep(backoffMs);
    }

    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${API_KEY}`,
        },
        body: JSON.stringify(bodyPayload),
      });

      if (res.status === 429 || res.status >= 500) {
        const errBody = await res.text().catch(() => '');
        lastError = new Error(`HTTP ${res.status}: ${errBody.slice(0, 200)}`);
        continue;
      }

      if (!res.ok) {
        const errBody = await res.text().catch(() => '');
        throw new Error(`HTTP ${res.status}: ${errBody.slice(0, 300)}`);
      }

      const jsonRes = (await res.json()) as any;
      const content = jsonRes.choices?.[0]?.message?.content;
      if (!content) {
        throw new Error('Respuesta de DeepSeek vacía o sin contenido');
      }

      const usage = jsonRes.usage || {};
      const promptTokens = usage.prompt_tokens || 0;
      const completionTokens = usage.completion_tokens || 0;
      const cacheHit = usage.prompt_cache_hit_tokens || 0;
      const cacheMiss = usage.prompt_cache_miss_tokens || Math.max(0, promptTokens - cacheHit);

      totalPromptTokens += promptTokens;
      totalCompletionTokens += completionTokens;
      totalCacheHitTokens += cacheHit;
      totalCacheMissTokens += cacheMiss;

      return {
        text: content,
        promptTokens,
        completionTokens,
        cacheHitTokens: cacheHit,
      };
    } catch (err) {
      lastError = err as Error;
    }
  }

  throw lastError || new Error('Fallaron todos los reintentos al llamar a DeepSeek');
}

let abortTriggered = false;

async function distillOne(
  row: TsIssueRow
): Promise<{ distillation: Distillation; promptTokens: number; completionTokens: number; cacheHitTokens: number }> {
  if (abortTriggered || IS_DRY_RUN) {
    return { distillation: fallback(row), promptTokens: 0, completionTokens: 0, cacheHitTokens: 0 };
  }

  try {
    const result = await callDeepSeek(row);
    const parsed = extractJson(result.text);
    if (isValid(parsed)) {
      return {
        distillation: parsed,
        promptTokens: result.promptTokens,
        completionTokens: result.completionTokens,
        cacheHitTokens: result.cacheHitTokens,
      };
    }
    console.error(`[Issue #${row.issue_number}] JSON válido pero con esquema inesperado`);
  } catch (e) {
    console.error(`[Issue #${row.issue_number}] Error llamando API: ${(e as Error).message}`);
  }

  return { distillation: fallback(row), promptTokens: 0, completionTokens: 0, cacheHitTokens: 0 };
}

// ============================================================================
// PERSISTENCIA Y CHECKPOINTS
// ============================================================================
function escapeCsvField(val: unknown): string {
  if (val === null || val === undefined) return '';
  const str = String(val);
  if (str.includes(',') || str.includes('"') || str.includes('\n') || str.includes('\r')) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

async function writeCsvFile(filePath: string, headers: string[], rows: Record<string, unknown>[]) {
  const headerLine = headers.join(',');
  const lines = rows.map((r) => headers.map((h) => escapeCsvField(r[h])).join(','));
  await Bun.write(filePath, [headerLine, ...lines].join('\n') + '\n');
}

function loadExistingCheckpoint(ckptPath: string): Map<string, CheckpointRecord> {
  const map = new Map<string, CheckpointRecord>();
  if (!existsSync(ckptPath)) return map;

  const content = readFileSync(ckptPath, 'utf-8');
  for (const line of content.split('\n')) {
    if (!line.trim()) continue;
    try {
      const rec = JSON.parse(line) as CheckpointRecord;
      if (rec.issue_number) map.set(String(rec.issue_number), rec);
    } catch {}
  }
  return map;
}

let isFlushing = false;
async function flushOutputs(ckptPath: string, outDir: string, splitName: string) {
  if (isFlushing) return;
  isFlushing = true;
  try {
    const doneMap = loadExistingCheckpoint(ckptPath);
    const records = Array.from(doneMap.values());
    if (records.length === 0) return;

    const distilledCsvPath = join(outDir, `${splitName}_distilled.csv`);
    const rawCsvPath = join(outDir, `${splitName}_raw.csv`);
    const bothCsvPath = join(outDir, `${splitName}_both.csv`);
    const provenanceCsvPath = join(outDir, `${splitName}_provenance.csv`);
    const provenanceParquetPath = join(outDir, `${splitName}_provenance.parquet`);
    const parquetPath = join(outDir, `${splitName}.parquet`);

    // 1. _distilled.csv
    const distRows = records.map((r) => ({
      issue_number: r.issue_number,
      owner: r.owner,
      text: r.distilled_text,
    }));
    await writeCsvFile(distilledCsvPath, ['issue_number', 'owner', 'text'], distRows);

    // 2. _raw.csv
    const rawRows = records.map((r) => ({
      issue_number: r.issue_number,
      owner: r.owner,
      text: r.raw_text,
    }));
    await writeCsvFile(rawCsvPath, ['issue_number', 'owner', 'text'], rawRows);

    // 3. _both.csv
    const bothRows = records.map((r) => ({
      issue_number: r.issue_number,
      owner: r.owner,
      text: `${r.raw_text} ${r.distilled_text}`.trim(),
    }));
    await writeCsvFile(bothCsvPath, ['issue_number', 'owner', 'text'], bothRows);

    // 4. _provenance.csv
    const provRows = records.map((r) => {
      let component = '';
      let subsystem = '';
      let symptomsStr = '';
      let capabilitiesStr = '';
      try {
        const parsed = JSON.parse(r.distilled_json);
        component = parsed.fault_location?.component || '';
        subsystem = parsed.fault_location?.subsystem || '';
        if (Array.isArray(parsed.symptoms)) symptomsStr = parsed.symptoms.join(' | ');
        if (Array.isArray(parsed.capabilities)) capabilitiesStr = parsed.capabilities.join(' | ');
      } catch {}

      let issueTitle = '';
      if (r.raw_text) {
        const m = r.raw_text.match(/Summary:\s*(.*)/i) || r.raw_text.match(/Bug Title:\s*(.*)/i);
        if (m) issueTitle = m[1].trim();
      }

      return {
        issue_number: r.issue_number,
        issue_title: issueTitle,
        owner: r.owner,
        model: r.model || MODEL,
        latency_ms: typeof r.latency_ms === 'number' ? r.latency_ms : 0,
        fallback: Boolean(r.fallback),
        component,
        subsystem,
        symptoms: symptomsStr,
        capabilities: capabilitiesStr,
        distilled_text: r.distilled_text || '',
      };
    });
    await writeCsvFile(
      provenanceCsvPath,
      [
        'issue_number',
        'issue_title',
        'owner',
        'model',
        'latency_ms',
        'fallback',
        'component',
        'subsystem',
        'symptoms',
        'capabilities',
        'distilled_text',
      ],
      provRows
    );

    // 5. .parquet y _provenance.parquet (vía uv python pyarrow si está disponible)
    const pyCode = `
import pandas as pd
try:
    df = pd.read_json('${ckptPath}', lines=True)
    df.to_parquet('${parquetPath}', index=False)
    df_prov = pd.read_csv('${provenanceCsvPath}')
    df_prov.to_parquet('${provenanceParquetPath}', index=False)
except Exception as e:
    pass
`;
    try {
      await Bun.$`uv run python -c ${pyCode}`.quiet();
    } catch {}

    console.log(`💾 [Flush] Guardados ${records.length} registros en Parquet, CSVs y Provenance (${distilledCsvPath})`);
  } finally {
    isFlushing = false;
  }
}

let writeLock = Promise.resolve();
async function safeAppendRecord(ckptPath: string, record: CheckpointRecord) {
  writeLock = writeLock.then(async () => {
    appendFileSync(ckptPath, JSON.stringify(record) + '\n', 'utf-8');
  });
  await writeLock;
}

function calculateCostUsd(cacheHit: number, cacheMiss: number, outputTokens: number): number {
  const costHit = (cacheHit / 1_000_000) * PRICE_CACHE_HIT_PER_1M;
  const costMiss = (cacheMiss / 1_000_000) * PRICE_CACHE_MISS_PER_1M;
  const costOut = (outputTokens / 1_000_000) * PRICE_OUTPUT_PER_1M;
  return costHit + costMiss + costOut;
}

// ============================================================================
// MAIN
// ============================================================================
async function main() {
  await Bun.$`mkdir -p ${OUT_DIR}`.quiet();

  if (!API_KEY && !IS_DRY_RUN) {
    console.error('❌ ERROR: DEEPSEEK_API_KEY no encontrada.');
    console.error('Por favor define la variable de entorno DEEPSEEK_API_KEY en tu archivo .env o en el entorno,');
    console.error('o pásala con: bun run scripts/distill_deepseek_ts.ts --api-key sk-xxxx');
    process.exit(1);
  }

  const ckptPath = join(OUT_DIR, `${SPLIT_NAME}.jsonl`);
  const existingRecords = loadExistingCheckpoint(ckptPath);

  // Unificamos issues ya destilados (propios y de checkpoints previos como ts_train_union.jsonl)
  const completedIds = new Set<string>(existingRecords.keys());
  for (const extraPath of EXTRA_CHECKPOINTS) {
    if (existsSync(extraPath)) {
      const extraMap = loadExistingCheckpoint(extraPath);
      for (const id of extraMap.keys()) {
        completedIds.add(id);
      }
      console.log(`🔗 Checkpoint previo vinculado: ${extraPath} (${extraMap.size} issues omitidos)`);
    }
  }

  console.log(`📂 Dataset: ${CSV_PATH}`);
  console.log(`🤖 Modelo: ${MODEL} (Thinking: ${ENABLE_THINKING ? 'HABILITADO' : 'DESHABILITADO'})`);
  console.log(`🌐 Base URL: ${BASE_URL}`);
  console.log(`📁 Directorio de salida: ${OUT_DIR}`);
  console.log(`🔄 Checkpoint activo: ${ckptPath} (${existingRecords.size} issues en este archivo)`);
  console.log(`🚫 Total issues omitidos: ${completedIds.size}`);
  console.log(`⚙️ Concurrencia: ${CONCURRENCY} workers HTTP`);
  console.log(`🛑 Circuit Breaker: abortar tras ${MAX_CONSECUTIVE_FAILURES} fallos consecutivos`);
  console.log(`↕️ Dirección: ${IS_REVERSE ? 'REVERSO' : 'ADELANTE'}`);
  console.log(`💾 Flush cada: ${FLUSH_EVERY} issues completados\n`);

  const fileContent = await Bun.file(CSV_PATH).text();
  const allRows: TsIssueRow[] = parse(fileContent, {
    columns: true,
    skip_empty_lines: true,
  });

  const slicedRows = LIMIT > 0 ? allRows.slice(OFFSET, OFFSET + LIMIT) : allRows.slice(OFFSET);
  
  // Deduplicación preservando orden
  const seenInSlice = new Set<string>();
  const dedupedRows: TsIssueRow[] = [];
  for (const r of slicedRows) {
    const id = String(r.issue_number);
    if (!seenInSlice.has(id)) {
      seenInSlice.add(id);
      dedupedRows.push(r);
    }
  }

  let pendingRows = dedupedRows.filter((r) => !completedIds.has(String(r.issue_number)));
  if (IS_REVERSE) {
    pendingRows = pendingRows.reverse();
  }

  console.log(`Total en rango: ${slicedRows.length} (Únicos: ${dedupedRows.length}) | Pendientes por destilar: ${pendingRows.length}`);
  if (pendingRows.length === 0) {
    console.log('✅ Todos los issues en el rango ya están procesados.');
    return;
  }

  if (IS_DRY_RUN) {
    console.log('\n🔍 [DRY RUN] Mostrando ejemplo de carga útil que se enviará a DeepSeek:\n');
    const sampleRow = pendingRows[0];
    const userPrompt = `Now extract the structured information for this issue report:\n\n${buildInput(sampleRow)}`;
    console.log('--- SYSTEM MESSAGE (Prefijo cacheable) ---');
    console.log(SYSTEM_MESSAGE.slice(0, 300) + '... [truncado para preview]');
    console.log('\n--- USER MESSAGE ---');
    console.log(userPrompt);
    console.log('\n--- BODY JSON PREVIEW ---');
    console.log(
      JSON.stringify(
        {
          model: MODEL,
          response_format: { type: 'json_object' },
          temperature: 0.0,
          thinking: { type: ENABLE_THINKING ? 'enabled' : 'disabled' },
        },
        null,
        2
      )
    );
    console.log('\n✅ Dry-run completado sin realizar llamadas de red ni escribir archivos.');
    return;
  }

  let completedThisRun = 0;
  let fallbacksThisRun = 0;
  let consecutiveFailures = 0;
  let batchStartTime = performance.now();
  let next = 0;

  const worker = async (workerId: number) => {
    while (true) {
      if (abortTriggered) return;
      const i = next++;
      if (i >= pendingRows.length) return;
      const row = pendingRows[i];

      const start = performance.now();
      const { distillation, promptTokens, completionTokens, cacheHitTokens } = await distillOne(row);
      if (abortTriggered) return;

      const latency = Math.round(performance.now() - start);
      const isFb = Boolean(distillation._fallback);

      if (isFb) {
        fallbacksThisRun++;
        consecutiveFailures++;
        console.warn(
          `⚠️ [Fallo ${consecutiveFailures}/${MAX_CONSECUTIVE_FAILURES}] Issue #${row.issue_number} terminó en fallback.`
        );
      } else {
        consecutiveFailures = 0;
      }

      const distilledText = toDistilledText(distillation);
      const rawText = (row.text || `${row.issue_title} ${row.description || ''}`).trim();

      const record: CheckpointRecord = {
        issue_number: String(row.issue_number),
        owner: row.owner || '',
        distilled_json: JSON.stringify(distillation),
        distilled_text: distilledText,
        raw_text: rawText,
        fallback: isFb,
        latency_ms: latency,
        model: MODEL,
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        cache_hit_tokens: cacheHitTokens,
      };

      await safeAppendRecord(ckptPath, record);
      completedThisRun++;

      const cacheHitPct = promptTokens > 0 ? Math.round((cacheHitTokens / promptTokens) * 100) : 0;
      console.log(
        `[W${workerId}] [${completedThisRun}/${pendingRows.length}] Issue #${row.issue_number} (${latency}ms) | ` +
          `Comp: ${distillation.fault_location.component.slice(0, 18).padEnd(18)} | Sub: ${distillation.fault_location.subsystem.slice(0, 24).padEnd(24)} | ` +
          `Tokens: ${promptTokens}in (${cacheHitPct}% cache) / ${completionTokens}out`
      );

      // Verificación de Circuit Breaker
      if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
        abortTriggered = true;
        console.error(`\n🛑 [CIRCUIT BREAKER ACTIVADO] ${consecutiveFailures} fallos consecutivos. Abortando.`);
        await flushOutputs(ckptPath, OUT_DIR, SPLIT_NAME);
        process.exit(1);
      }

      // Flush periódico
      if (completedThisRun % FLUSH_EVERY === 0 || completedThisRun === pendingRows.length) {
        const batchElapsed = Math.round((performance.now() - batchStartTime) / 1000);
        const curCost = calculateCostUsd(totalCacheHitTokens, totalCacheMissTokens, totalCompletionTokens);
        console.log(
          `\n📊 Lote: ${completedThisRun} completados (${batchElapsed}s) | Fallbacks: ${fallbacksThisRun} | Costo acumulado: ~$${curCost.toFixed(4)} USD`
        );
        await flushOutputs(ckptPath, OUT_DIR, SPLIT_NAME);
        batchStartTime = performance.now();
        console.log('');
      }

      if (MAX_DELAY_MS > 0 && next < pendingRows.length) {
        const span = Math.max(0, MAX_DELAY_MS - MIN_DELAY_MS);
        const waitMs = MIN_DELAY_MS + Math.floor(Math.random() * (span + 1));
        await Bun.sleep(waitMs);
      }
    }
  };

  const activeWorkers = Array.from({ length: Math.min(CONCURRENCY, pendingRows.length) }, (_, id) =>
    worker(id)
  );
  await Promise.all(activeWorkers);

  await flushOutputs(ckptPath, OUT_DIR, SPLIT_NAME);

  const totalCost = calculateCostUsd(totalCacheHitTokens, totalCacheMissTokens, totalCompletionTokens);
  console.log(`\n🎉 Destilación finalizada: ${completedThisRun} issues procesados.`);
  console.log(`📈 Resumen de Tokens:`);
  console.log(`   - Input Tokens: ${totalPromptTokens.toLocaleString()} (Cache Hit: ${totalCacheHitTokens.toLocaleString()}, Cache Miss: ${totalCacheMissTokens.toLocaleString()})`);
  console.log(`   - Output Tokens: ${totalCompletionTokens.toLocaleString()}`);
  console.log(`   - Costo Estimado: ~$${totalCost.toFixed(4)} USD\n`);
}

main().catch((e) => {
  console.error(`Error fatal: ${(e as Error).message}`);
  process.exit(1);
});
