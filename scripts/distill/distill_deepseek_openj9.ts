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

import { parse } from 'csv-parse/sync';
import { parseArgs } from 'util';
import { appendFileSync, existsSync, readFileSync } from 'fs';
import { join } from 'path';
import {
  Distillation,
  isValid,
  toDistilledText,
  extractJson,
} from './deepseek_engine';

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    split: { type: 'string', default: 'train' }, // train | test
    limit: { type: 'string', default: '0' },     // 0 = todo
    offset: { type: 'string', default: '0' },
    concurrency: { type: 'string', default: '5' },
    flushEvery: { type: 'string', default: '10' },
    'flush-every': { type: 'string', default: '10' },
    outDir: { type: 'string', default: 'artifacts/openj9/distilled' },
    'out-dir': { type: 'string', default: 'artifacts/openj9/distilled' },
    csvDir: { type: 'string', default: 'artifacts/openj9' },
    'csv-dir': { type: 'string', default: 'artifacts/openj9' },
    model: { type: 'string', default: 'deepseek-flash' },
    m: { type: 'string' },
    apiKey: { type: 'string' },
    'api-key': { type: 'string' },
    baseUrl: { type: 'string', default: 'https://api.deepseek.com' },
    'base-url': { type: 'string', default: 'https://api.deepseek.com' },
    extraCheckpoints: { type: 'string', default: '' },
    'extra-checkpoints': { type: 'string', default: '' },
    maxConsecutiveFailures: { type: 'string' },
    'max-consecutive-failures': { type: 'string' },
    dryRun: { type: 'boolean', default: false },
    'dry-run': { type: 'boolean', default: false },
  },
  strict: false,
});

const SPLIT = (values.split as string) || 'train';
const LIMIT = parseInt(values.limit || '0', 10);
const OFFSET = parseInt(values.offset || '0', 10);
const CONCURRENCY = parseInt(values.concurrency || '5', 10);
const FLUSH_EVERY = parseInt(values['flush-every'] || values.flushEvery || '10', 10);
const OUT_DIR = (values['out-dir'] || values.outDir || 'artifacts/openj9/distilled') as string;
const CSV_DIR = (values['csv-dir'] || values.csvDir || 'artifacts/openj9') as string;
const CSV_PATH = join(CSV_DIR, `openj9_${SPLIT}_50.csv`);
const IS_DRY_RUN = Boolean(values['dry-run'] || values.dryRun);
const MAX_CONSECUTIVE_FAILURES = parseInt(
  (values['max-consecutive-failures'] || values.maxConsecutiveFailures || String(CONCURRENCY * 2)) as string,
  10
);
const MAX_BODY_CHARS = 1500;

// Configuración DeepSeek API
const API_KEY = (values['api-key'] || values.apiKey || process.env.DEEPSEEK_API_KEY || '').trim();
const BASE_URL = ((values['base-url'] || values.baseUrl || 'https://api.deepseek.com') as string).replace(/\/+$/, '');
const MODEL = (values.m as string) || (values.model as string) || 'deepseek-flash';

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

interface OpenJ9IssueRow {
  issue_number: string;
  issue_title: string;
  issue_body: string;
  owner: string;
  component?: string;
  text?: string;
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

function normalizeOpenJ9Component(comp: string | undefined): string {
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

function buildInput(row: OpenJ9IssueRow): string {
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

function fallback(row: OpenJ9IssueRow): Distillation {
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

// Estadísticas acumuladas
let totalPromptTokens = 0;
let totalCompletionTokens = 0;
let totalCacheHitTokens = 0;
let totalCacheMissTokens = 0;

async function callDeepSeek(
  row: OpenJ9IssueRow,
  retries = 4
): Promise<{ text: string; promptTokens: number; completionTokens: number; cacheHitTokens: number }> {
  const url = `${BASE_URL}/chat/completions`;
  const userPrompt = `Now extract the structured information for this OpenJ9 issue report:\n\n${buildInput(row)}`;

  const bodyPayload: Record<string, unknown> = {
    model: MODEL,
    messages: [
      { role: 'system', content: SYSTEM_MESSAGE },
      { role: 'user', content: userPrompt },
    ],
    response_format: { type: 'json_object' },
    temperature: 0.0,
    thinking: { type: 'disabled' },
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
  row: OpenJ9IssueRow
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
// PERSISTENCIA Y CHECKPOINTS (Compatible con TriagerX y Tabla de Constancia)
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

    const prefix = `openj9_${splitName}`;
    const distilledCsvPath = join(outDir, `${prefix}_distilled.csv`);
    const rawCsvPath = join(outDir, `${prefix}_raw.csv`);
    const bothCsvPath = join(outDir, `${prefix}_both.csv`);
    const provenanceCsvPath = join(outDir, `${prefix}_provenance.csv`);
    const provenanceParquetPath = join(outDir, `${prefix}_provenance.parquet`);
    const parquetPath = join(outDir, `${prefix}.parquet`);

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

    // 4. _provenance.csv (11 columnas canónicas exactas)
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

    // 5. .parquet y _provenance.parquet vía Python PyArrow si está disponible
    const pyCode = `
import pandas as pd
try:
    df = pd.read_json('${ckptPath}', lines=True)
    df.to_parquet('${parquetPath}', index=False)
    df_prov = pd.read_csv('${provenanceCsvPath}')
    df_prov.to_parquet('${provenanceParquetPath}', index=False)
except Exception:
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
    console.error('Por favor define DEEPSEEK_API_KEY en tu archivo .env o en el entorno.');
    process.exit(1);
  }

  if (!existsSync(CSV_PATH)) {
    console.error(`❌ ERROR: Archivo dataset no encontrado: ${CSV_PATH}`);
    process.exit(1);
  }

  const ckptPath = join(OUT_DIR, `openj9_${SPLIT}.jsonl`);
  const existingRecords = loadExistingCheckpoint(ckptPath);

  const completedIds = new Set<string>(existingRecords.keys());
  const extraCheckpoints = values['extra-checkpoints'] || values.extraCheckpoints;
  if (extraCheckpoints) {
    for (const extraPath of String(extraCheckpoints).split(',').map((s) => s.trim())) {
      if (existsSync(extraPath)) {
        const extraMap = loadExistingCheckpoint(extraPath);
        for (const id of extraMap.keys()) {
          completedIds.add(id);
        }
        console.log(`🔗 Checkpoint previo vinculado: ${extraPath} (${extraMap.size} issues omitidos)`);
      }
    }
  }

  console.log(`\n======================================================`);
  console.log(`☕ DESTILACIÓN OPENJ9 (JVM) CON DEEPSEEK API`);
  console.log(`   Split:        ${SPLIT}`);
  console.log(`   Dataset:      ${CSV_PATH}`);
  console.log(`   Modelo:       ${MODEL} (Thinking: DESHABILITADO)`);
  console.log(`   Directorio:   ${OUT_DIR}`);
  console.log(`   Checkpoint:   ${ckptPath} (${existingRecords.size} hechos)`);
  console.log(`   Concurrencia: ${CONCURRENCY} workers`);
  console.log(`======================================================\n`);

  const fileContent = await Bun.file(CSV_PATH).text();
  const allRows: OpenJ9IssueRow[] = parse(fileContent, {
    columns: true,
    skip_empty_lines: true,
  });

  const slicedRows = LIMIT > 0 ? allRows.slice(OFFSET, OFFSET + LIMIT) : allRows.slice(OFFSET);
  const pendingRows = slicedRows.filter((r) => !completedIds.has(String(r.issue_number)));

  console.log(`Total en split: ${slicedRows.length} | Pendientes por destilar: ${pendingRows.length}`);
  if (pendingRows.length === 0) {
    console.log('✅ Todos los issues en el rango ya están procesados.');
    await flushOutputs(ckptPath, OUT_DIR, SPLIT);
    return;
  }

  if (IS_DRY_RUN) {
    console.log('\n🔍 [DRY RUN] Mostrando ejemplo de carga útil que se enviará a DeepSeek:\n');
    const sampleRow = pendingRows[0];
    const userPrompt = `Now extract the structured information for this OpenJ9 issue report:\n\n${buildInput(sampleRow)}`;
    console.log('--- SYSTEM MESSAGE (Prefijo cacheable) ---');
    console.log(SYSTEM_MESSAGE.slice(0, 350) + '... [truncado]');
    console.log('\n--- USER MESSAGE ---');
    console.log(userPrompt);
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
      const rawText = (row.text || `${row.issue_title} ${row.issue_body || ''}`).trim();

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
          `Comp: ${distillation.fault_location.component.slice(0, 15).padEnd(15)} | Sub: ${distillation.fault_location.subsystem.slice(0, 26).padEnd(26)} | ` +
          `Tokens: ${promptTokens}in (${cacheHitPct}% cache) / ${completionTokens}out`
      );

      // Verificación de Circuit Breaker
      if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
        abortTriggered = true;
        console.error(`\n🛑 [CIRCUIT BREAKER ACTIVADO] ${consecutiveFailures} fallos consecutivos. Abortando.`);
        await flushOutputs(ckptPath, OUT_DIR, SPLIT);
        process.exit(1);
      }

      // Flush periódico
      if (completedThisRun % FLUSH_EVERY === 0 || completedThisRun === pendingRows.length) {
        const batchElapsed = Math.round((performance.now() - batchStartTime) / 1000);
        const curCost = calculateCostUsd(totalCacheHitTokens, totalCacheMissTokens, totalCompletionTokens);
        console.log(
          `\n📊 Lote: ${completedThisRun} completados (${batchElapsed}s) | Fallbacks: ${fallbacksThisRun} | Costo acumulado: ~$${curCost.toFixed(4)} USD`
        );
        await flushOutputs(ckptPath, OUT_DIR, SPLIT);
        batchStartTime = performance.now();
        console.log('');
      }
    }
  };

  const activeWorkers = Array.from({ length: Math.min(CONCURRENCY, pendingRows.length) }, (_, id) =>
    worker(id)
  );
  await Promise.all(activeWorkers);

  await flushOutputs(ckptPath, OUT_DIR, SPLIT);

  const totalCost = calculateCostUsd(totalCacheHitTokens, totalCacheMissTokens, totalCompletionTokens);
  console.log(`\n🎉 Destilación OpenJ9 finalizada: ${completedThisRun} issues procesados.`);
  console.log(`📈 Resumen de Tokens:`);
  console.log(`   - Input Tokens: ${totalPromptTokens.toLocaleString()} (Cache Hit: ${totalCacheHitTokens.toLocaleString()}, Cache Miss: ${totalCacheMissTokens.toLocaleString()})`);
  console.log(`   - Output Tokens: ${totalCompletionTokens.toLocaleString()}`);
  console.log(`   - Costo Estimado: ~$${totalCost.toFixed(4)} USD\n`);
}

main().catch((e) => {
  console.error(`Error fatal: ${(e as Error).message}`);
  process.exit(1);
});
