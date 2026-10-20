// Destilación de issues de TypeScript usando el CLI de opencode (opencode/union-alpha).
//
// Características:
//   - Concurrencia paralela (3 workers por defecto con directorios aislados /tmp/opencode-worker-N)
//   - Timeout ampliado a 5 minutos (300,000ms) para issues con bloques extensos de código
//   - Bandera --pure para evitar sobrecarga de plugins
//   - Checkpoints en streaming (.jsonl) con lock asíncrono
//   - Flush periódico cada 5 issues completados a .parquet y .csv (distilled, raw, both)
//   - Idempotente y reanudable
//
// Ejecución:
//   bun run scripts/distill_opencode_ts.ts [--limit 50] [--concurrency 3] [--flush-every 5]

import { parse } from 'csv-parse/sync';
import { parseArgs } from 'util';
import { appendFileSync, existsSync, readFileSync } from 'fs';
import { join } from 'path';

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    limit: { type: 'string', default: '0' }, // 0 = procesar todo
    offset: { type: 'string', default: '0' },
    concurrency: { type: 'string', default: '3' },
    flushEvery: { type: 'string', default: '5' },
    'flush-every': { type: 'string', default: '5' },
    split: { type: 'string', default: 'ts_train' },
    outDir: { type: 'string', default: 'artifacts/typescript/distilled' },
    'out-dir': { type: 'string', default: 'artifacts/typescript/distilled' },
    csv: { type: 'string', default: '/Users/sebastian/Documents/Python/triagerX/df_train_ts.csv' },
    model: { type: 'string', default: 'opencode/union-alpha' },
    m: { type: 'string' },
    reverse: { type: 'boolean', default: false },
    extraCheckpoints: { type: 'string', default: '' },
    'extra-checkpoints': { type: 'string', default: '' },
    minDelay: { type: 'string', default: '0' },
    'min-delay': { type: 'string', default: '0' },
    maxDelay: { type: 'string', default: '0' },
    'max-delay': { type: 'string', default: '0' },
    maxConsecutiveFailures: { type: 'string' },
    'max-consecutive-failures': { type: 'string' },
  },
  strict: false,
});

const LIMIT = parseInt(values.limit || '0', 10);
const OFFSET = parseInt(values.offset || '0', 10);
const CONCURRENCY = parseInt(values.concurrency || '3', 10);
const MAX_CONSECUTIVE_FAILURES = parseInt(
  (values['max-consecutive-failures'] || values.maxConsecutiveFailures || String(CONCURRENCY)) as string,
  10
);
const FLUSH_EVERY = parseInt(values['flush-every'] || values.flushEvery || '5', 10);
const SPLIT_NAME = (values.split as string) || 'ts_train';
const OUT_DIR = (values['out-dir'] || values.outDir || 'artifacts/typescript/distilled') as string;
const CSV_PATH = values.csv as string;
const IS_REVERSE = Boolean(values.reverse);
const EXTRA_CHECKPOINTS = ((values['extra-checkpoints'] || values.extraCheckpoints || '') as string)
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);
const MIN_DELAY_SEC = parseFloat((values['min-delay'] || values.minDelay || '0') as string);
const MAX_DELAY_SEC = parseFloat((values['max-delay'] || values.maxDelay || '0') as string);
const MIN_DELAY_MS = Math.round(MIN_DELAY_SEC * 1000);
const MAX_DELAY_MS = Math.round(MAX_DELAY_SEC * 1000);
const MAX_COMMENT_CHARS = 1500;

// Configuración OpenCode CLI
const OPENCODE_BIN = 'opencode';
const OPENCODE_MODEL = (values.m as string) || (values.model as string) || 'opencode/union-alpha';
const OPENCODE_TIMEOUT_MS = 300_000; // 5 minutos por intento

// Prompt refinado para el compilador de TypeScript con reglas anti-cuelgue
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
- If information is insufficient, use empty list [] or empty string "".
`;

// Few-shot examples del compilador
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

function buildMessage(row: TsIssueRow): string {
  const shots = FEW_SHOT_EXAMPLES.map(
    (ex) => `Input:\n${ex.input}\n\nOutput:\n${ex.output}`,
  ).join('\n\n');
  return `${SYSTEM_PROMPT}\n\n## Examples\n\n${shots}\n\n## Now extract this one\n\n${buildInput(row)}`;
}

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

async function callOpencode(message: string, workerDir: string): Promise<string> {
  const proc = Bun.spawn({
    cmd: [
      OPENCODE_BIN,
      'run',
      '-m',
      OPENCODE_MODEL,
      '--format',
      'json',
      '--pure',
      '--dir',
      workerDir,
      message,
    ],
    cwd: workerDir,
    env: {
      ...process.env,
      OPENCODE_DB: ':memory:',
    },
    stdout: 'pipe',
    stderr: 'pipe',
  });

  const timer = setTimeout(() => proc.kill(), OPENCODE_TIMEOUT_MS);
  let out: string;
  let err: string;
  try {
    [out, err] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ]);
    await proc.exited;
  } finally {
    clearTimeout(timer);
  }

  if (proc.exitCode !== 0) {
    throw new Error(`opencode salió con ${proc.exitCode}: ${err.trim().slice(0, 200)}`);
  }
  if (!out.trim()) {
    throw new Error(`opencode no devolvió salida: ${err.trim().slice(0, 200)}`);
  }

  const texts: string[] = [];
  for (const line of out.split('\n')) {
    if (!line.trim()) continue;
    try {
      const ev = JSON.parse(line);
      if (ev.type === 'text' && typeof ev.part?.text === 'string') texts.push(ev.part.text);
    } catch {}
  }
  if (texts.length === 0) throw new Error('ningún evento "text" en el stream de opencode');
  return texts.join('');
}

let abortTriggered = false;

async function distillOne(row: TsIssueRow, workerDir: string): Promise<Distillation> {
  if (abortTriggered) return fallback(row);
  const message = buildMessage(row);

  for (let attempt = 0; attempt < 2; attempt++) {
    if (abortTriggered) return fallback(row);
    try {
      const rawText = await callOpencode(message, workerDir);
      const parsed = extractJson(rawText);
      if (isValid(parsed)) return parsed;
      console.error(`[Issue #${row.issue_number}] Intento ${attempt + 1}: JSON válido pero estructura inesperada`);
    } catch (e) {
      console.error(`[Issue #${row.issue_number}] Intento ${attempt + 1} falló: ${(e as Error).message}`);
    }
  }
  console.error(`[Issue #${row.issue_number}] -> cayendo a fallback`);
  return fallback(row);
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

    // 4. _provenance.csv (registro de constancia)
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
        const m = r.raw_text.match(/Bug Title:\s*(.*)/);
        if (m) issueTitle = m[1].trim();
      }

      return {
        issue_number: r.issue_number,
        issue_title: issueTitle,
        owner: r.owner,
        model: r.model || OPENCODE_MODEL,
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

    // 5. .parquet y _provenance.parquet (vía uv python pyarrow)
    const pyCode = `
import pandas as pd
df = pd.read_json('${ckptPath}', lines=True)
df.to_parquet('${parquetPath}', index=False)

df_prov = pd.read_csv('${provenanceCsvPath}')
df_prov.to_parquet('${provenanceParquetPath}', index=False)
`;
    try {
      await Bun.$`uv run python -c ${pyCode}`.quiet();
    } catch (err) {
      console.error(`Advertencia: no se pudo actualizar Parquets: ${(err as Error).message}`);
    }

    console.log(`💾 [Flush] Guardados ${records.length} registros en Parquet, CSVs y Provenance (${distilledCsvPath})`);
  } finally {
    isFlushing = false;
  }
}

// Mutex para escritura segura en .jsonl desde múltiples workers concurrentes
let writeLock = Promise.resolve();
async function safeAppendRecord(ckptPath: string, record: CheckpointRecord) {
  writeLock = writeLock.then(async () => {
    appendFileSync(ckptPath, JSON.stringify(record) + '\n', 'utf-8');
  });
  await writeLock;
}

// ============================================================================
// MAIN LOOP
// ============================================================================
async function main() {
  await Bun.$`mkdir -p ${OUT_DIR}`.quiet();

  const ckptPath = join(OUT_DIR, `${SPLIT_NAME}.jsonl`);
  const existingRecords = loadExistingCheckpoint(ckptPath);

  // Conjunto de issues ya completados (checkpoint propio + checkpoints extras)
  const completedIds = new Set<string>(existingRecords.keys());
  for (const extraPath of EXTRA_CHECKPOINTS) {
    if (existsSync(extraPath)) {
      const extraMap = loadExistingCheckpoint(extraPath);
      for (const id of extraMap.keys()) {
        completedIds.add(id);
      }
      console.log(`🔗 Checkpoint adicional vinculado: ${extraPath} (${extraMap.size} issues)`);
    }
  }

  console.log(`📂 Dataset: ${CSV_PATH}`);
  console.log(`🤖 Modelo: ${OPENCODE_MODEL}`);
  console.log(`📁 Directorio de salida: ${OUT_DIR}`);
  console.log(`🔄 Checkpoint propio: ${ckptPath} (${existingRecords.size} issues guardados aquí)`);
  console.log(`🚫 Total issues omitidos (por checkpoint(s)): ${completedIds.size}`);
  console.log(`⚙️ Concurrencia: ${CONCURRENCY} workers paralelos`);
  console.log(`🛑 Circuit Breaker: abortar tras ${MAX_CONSECUTIVE_FAILURES} fallos consecutivos (igual a workers)`);
  console.log(`⏱️ Timeout por intento: ${OPENCODE_TIMEOUT_MS / 1000}s (5 minutos)`);
  console.log(`↕️ Dirección: ${IS_REVERSE ? 'REVERSO (del final hacia el inicio)' : 'ADELANTE (del inicio hacia el final)'}`);
  if (MAX_DELAY_MS > 0) {
    console.log(`⏳ Espera aleatoria: entre ${MIN_DELAY_SEC}s y ${MAX_DELAY_SEC}s por worker entre ejecuciones`);
  }
  console.log(`💾 Flush cada: ${FLUSH_EVERY} issues completados\n`);

  const fileContent = await Bun.file(CSV_PATH).text();
  const allRows: TsIssueRow[] = parse(fileContent, {
    columns: true,
    skip_empty_lines: true,
  });

  const slicedRows = LIMIT > 0 ? allRows.slice(OFFSET, OFFSET + LIMIT) : allRows.slice(OFFSET);
  let pendingRows = slicedRows.filter((r) => !completedIds.has(String(r.issue_number)));
  if (IS_REVERSE) {
    pendingRows = pendingRows.reverse();
  }

  console.log(`Total en rango: ${slicedRows.length} | Pendientes: ${pendingRows.length}`);
  if (pendingRows.length === 0) {
    console.log('✅ Todos los issues en el rango ya están procesados en el checkpoint.');
    await flushOutputs(ckptPath, OUT_DIR, SPLIT_NAME);
    return;
  }

  let completedThisRun = 0;
  let fallbacksThisRun = 0;
  let consecutiveFailures = 0;
  let batchStartTime = performance.now();
  let next = 0;

  // Creamos los directorios aislados por worker y split para evitar colisiones en SQLite de OpenCode
  for (let w = 0; w < CONCURRENCY; w++) {
    await Bun.$`mkdir -p /tmp/opencode-worker-${SPLIT_NAME}-${w}`.quiet();
  }

  const worker = async (workerId: number) => {
    const workerDir = `/tmp/opencode-worker-${SPLIT_NAME}-${workerId}`;

    while (true) {
      if (abortTriggered) return;
      const i = next++;
      if (i >= pendingRows.length) return;
      const row = pendingRows[i];

      const start = performance.now();
      const distilled = await distillOne(row, workerDir);
      if (abortTriggered) return;

      const latency = Math.round(performance.now() - start);

      const isFb = Boolean(distilled._fallback);
      if (isFb) {
        fallbacksThisRun++;
        consecutiveFailures++;
        console.warn(
          `⚠️ [Fallo consecutivo ${consecutiveFailures}/${MAX_CONSECUTIVE_FAILURES}] Issue #${row.issue_number} terminó en fallback.`
        );
      } else {
        consecutiveFailures = 0;
      }

      const distilledText = toDistilledText(distilled);
      const rawText = (row.text || `${row.issue_title} ${row.description || ''}`).trim();

      const record: CheckpointRecord = {
        issue_number: String(row.issue_number),
        owner: row.owner || '',
        distilled_json: JSON.stringify(distilled),
        distilled_text: distilledText,
        raw_text: rawText,
        fallback: isFb,
        latency_ms: latency,
        model: OPENCODE_MODEL,
      };

      // Guardado concurrente seguro con mutex
      await safeAppendRecord(ckptPath, record);
      completedThisRun++;

      console.log(
        `[W${workerId}] [${completedThisRun}/${pendingRows.length}] Issue #${row.issue_number} (${latency}ms) | ` +
          `Comp: ${distilled.fault_location.component} | Sub: ${distilled.fault_location.subsystem}`
      );

      // Verificación de Circuit Breaker: si hay N fallos seguidos (igual a workers), abortar inmediatamente
      if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
        abortTriggered = true;
        console.error(
          `\n🛑======================================================================`
        );
        console.error(
          `🛑 [CIRCUIT BREAKER ACTIVADO] Se alcanzaron ${consecutiveFailures} fallos consecutivos (umbral: ${MAX_CONSECUTIVE_FAILURES} workers).`
        );
        console.error(
          `🛑 El proveedor opencode parece estar caído o respondiendo con error continuo (ej. 'opencode salió con 1').`
        );
        console.error(
          `🛑 Guardando progreso acumulado y deteniendo toda la ejecución para no generar fallbacks masivos.`
        );
        console.error(
          `🛑======================================================================\n`
        );
        await flushOutputs(ckptPath, OUT_DIR, SPLIT_NAME);
        process.exit(1);
      }

      // Flush periódico a Parquet y CSVs cada N registros
      if (completedThisRun % FLUSH_EVERY === 0 || completedThisRun === pendingRows.length) {
        const batchElapsed = Math.round((performance.now() - batchStartTime) / 1000);
        console.log(
          `\n📊 Lote completado: ${completedThisRun} procesados (${batchElapsed}s) | Fallbacks: ${fallbacksThisRun}`
        );
        await flushOutputs(ckptPath, OUT_DIR, SPLIT_NAME);
        batchStartTime = performance.now();
        console.log('');
      }

      // Espera aleatoria por worker antes del siguiente issue si está configurada
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

  // Flush final asegurado
  await flushOutputs(ckptPath, OUT_DIR, SPLIT_NAME);
  console.log(`\n🎉 Destilación finalizada: ${completedThisRun} issues procesados en esta corrida.`);
}

main().catch((e) => {
  console.error(`Error fatal: ${(e as Error).message}`);
  process.exit(1);
});
