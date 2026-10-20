// Reprocesa los issues que cayeron en fallback para re-intentar la destilación con el LLM
//
// Ejecución:
//   bun run scripts/reprocess_fallbacks.ts [--split ts_train_union] [--model opencode/union-alpha] [--concurrency 2] [--limit 10]
//
// Si no se especifican --ids, detecta automáticamente todos los registros con fallback=true en el split.

import { parse } from 'csv-parse/sync';
import { existsSync, readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import { parseArgs } from 'util';

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    split: { type: 'string', default: 'ts_train_union' },
    outDir: { type: 'string', default: 'artifacts/typescript/distilled' },
    'out-dir': { type: 'string', default: 'artifacts/typescript/distilled' },
    csv: { type: 'string' },
    provider: { type: 'string', default: process.env.DEEPSEEK_API_KEY ? 'deepseek' : 'opencode' },
    model: { type: 'string' },
    concurrency: { type: 'string', default: '3' },
    limit: { type: 'string', default: '0' },
    ids: { type: 'string', default: '' },
    timeout: { type: 'string', default: '300000' },
    maxConsecutiveFailures: { type: 'string' },
    'max-consecutive-failures': { type: 'string' },
  },
  strict: false,
});

const SPLIT_NAME = (values.split as string) || 'ts_train_union';
const isTest = SPLIT_NAME.includes('test');
const OUT_DIR = (values['out-dir'] || values.outDir || 'artifacts/typescript/distilled') as string;
const CSV_PATH =
  (values.csv as string) ||
  (isTest
    ? '/Users/sebastian/Documents/Python/triagerX/df_test_ts.csv'
    : '/Users/sebastian/Documents/Python/triagerX/df_train_ts.csv');
const PROVIDER = (values.provider as string) || (process.env.DEEPSEEK_API_KEY ? 'deepseek' : 'opencode');
const MODEL_NAME = (values.model as string) || (PROVIDER === 'deepseek' ? 'deepseek-flash' : 'opencode/union-alpha');
const CONCURRENCY = parseInt(values.concurrency || '3', 10);
const MAX_CONSECUTIVE_FAILURES = parseInt(
  (values['max-consecutive-failures'] || values.maxConsecutiveFailures || String(CONCURRENCY * 2)) as string,
  10
);
const LIMIT = parseInt(values.limit || '0', 10);
const OPENCODE_TIMEOUT_MS = parseInt(values.timeout || '300000', 10);
const OPENCODE_BIN = 'opencode';

const CKPT_PATH = join(OUT_DIR, `${SPLIT_NAME}.jsonl`);

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
- fault_location.component: if provided in input, use it. If empty or '(infer canonical compiler component)', classify into one of the canonical compiler components: Type Checker, Emitter, Parser, Language Service, Binder, Module Resolution, CLI, or Compiler.
- fault_location.subsystem: the specific language construct or compiler pass involved.
- symptoms: 1 to 5 concise items describing the observed failure, unexpected error, or incorrect code generation.
- capabilities: 1 to 5 concrete compiler engineering skills needed to resolve the issue (e.g. "generic type inference", "AST transformation", "sourcemap generation"). Avoid generic labels like "TypeScript" or "debugging".
`;

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

function buildInput(row: any): string {
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
  parts.push((row.description ?? '').slice(0, 1500));
  return parts.join('\n');
}

function buildMessage(row: any): string {
  const shots = FEW_SHOT_EXAMPLES.map(
    (ex) => `Input:\n${ex.input}\n\nOutput:\n${ex.output}`,
  ).join('\n\n');
  return `${SYSTEM_PROMPT}\n\n## Examples\n\n${shots}\n\n## Now extract this one\n\n${buildInput(row)}`;
}

function extractJson(raw: string): any {
  let s = raw.trim();
  if (s.startsWith('```')) s = s.replace(/^```(?:json)?/, '').replace(/```$/, '').trim();
  const start = s.indexOf('{');
  if (start === -1) throw new Error('sin objeto JSON');
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < s.length; i++) {
    const c = s[i];
    if (escaped) { escaped = false; continue; }
    if (c === '\\') { escaped = true; continue; }
    if (c === '"') { inString = !inString; continue; }
    if (inString) continue;
    if (c === '{') depth++;
    else if (c === '}' && --depth === 0) return JSON.parse(s.slice(start, i + 1));
  }
  throw new Error('JSON sin cerrar');
}

function isValid(d: any): boolean {
  return (
    d &&
    typeof d === 'object' &&
    d.fault_location &&
    typeof d.fault_location === 'object' &&
    typeof d.fault_location.product === 'string' &&
    typeof d.fault_location.component === 'string' &&
    d.fault_location.component.trim().length > 0 &&
    Array.isArray(d.symptoms) &&
    d.symptoms.length > 0 &&
    d.symptoms.every((s: unknown) => typeof s === 'string') &&
    Array.isArray(d.capabilities) &&
    d.capabilities.length > 0 &&
    d.capabilities.every((s: unknown) => typeof s === 'string')
  );
}

function toDistilledText(d: any): string {
  const fl = [d?.fault_location?.product, d?.fault_location?.component, d?.fault_location?.subsystem]
    .filter(Boolean)
    .join(' ');
  return `[FL] ${fl} [SY] ${d.symptoms.join(' ')} [CP] ${d.capabilities.join(' ')}`.trim();
}

const shots = FEW_SHOT_EXAMPLES.map(
  (ex) => `Example Input:\n${ex.input}\n\nExample Output:\n${ex.output}`
).join('\n\n');
const SYSTEM_MESSAGE = `${SYSTEM_PROMPT}\n\n## Few-shot Examples\n\n${shots}`;

async function callDeepSeek(row: any): Promise<string> {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error('DEEPSEEK_API_KEY no encontrada en el entorno');

  const userPrompt = `Now extract the structured information for this issue report:\n\n${buildInput(row)}`;

  const res = await fetch('https://api.deepseek.com/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: MODEL_NAME,
      messages: [
        { role: 'system', content: SYSTEM_MESSAGE },
        { role: 'user', content: userPrompt },
      ],
      response_format: { type: 'json_object' },
      temperature: 0.0,
      thinking: { type: 'disabled' },
    }),
  });

  if (!res.ok) {
    const errText = await res.text().catch(() => '');
    throw new Error(`DeepSeek HTTP ${res.status}: ${errText.slice(0, 200)}`);
  }

  const json = (await res.json()) as any;
  return json.choices?.[0]?.message?.content || '';
}

async function callOpencode(message: string, workerDir: string): Promise<string> {
  const proc = Bun.spawn({
    cmd: [OPENCODE_BIN, 'run', '-m', MODEL_NAME, '--format', 'json', '--pure', '--dir', workerDir, message],
    cwd: workerDir,
    env: { ...process.env, OPENCODE_DB: ':memory:' },
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
  if (proc.exitCode !== 0) throw new Error(`opencode salió con ${proc.exitCode}: ${err.slice(0, 150)}`);
  const texts: string[] = [];
  for (const line of out.split('\n')) {
    if (!line.trim()) continue;
    try {
      const ev = JSON.parse(line);
      if (ev.type === 'text' && typeof ev.part?.text === 'string') texts.push(ev.part.text);
    } catch {}
  }
  if (texts.length === 0) throw new Error('sin eventos text');
  return texts.join('');
}

async function main() {
  console.log(`\n======================================================`);
  console.log(`🔁 REPROCESAMIENTO DE FALLBACKS CON LLM`);
  console.log(`   Split:        ${SPLIT_NAME}`);
  console.log(`   Proveedor:    ${PROVIDER}`);
  console.log(`   Modelo:       ${MODEL_NAME}`);
  console.log(`   Concurrencia: ${CONCURRENCY} workers`);
  console.log(`======================================================\n`);

  if (!existsSync(CKPT_PATH)) {
    console.error(`❌ Checkpoint no encontrado: ${CKPT_PATH}`);
    process.exit(1);
  }

  const lines = readFileSync(CKPT_PATH, 'utf-8').trim().split('\n');
  const existingMap = new Map<string, any>();
  const fallbackRecords: any[] = [];

  for (const line of lines) {
    if (!line.trim()) continue;
    const rec = JSON.parse(line);
    existingMap.set(String(rec.issue_number), rec);
    if (rec.fallback) {
      fallbackRecords.push(rec);
    }
  }

  console.log(`Total registros en ${SPLIT_NAME}: ${existingMap.size}`);
  console.log(`Total registros en fallback:     ${fallbackRecords.length}`);

  let targets: string[] = [];
  if (values.ids) {
    targets = (values.ids as string).split(',').map((s) => s.trim()).filter(Boolean);
  } else {
    targets = fallbackRecords.map((r) => String(r.issue_number));
  }

  if (LIMIT > 0 && targets.length > LIMIT) {
    console.log(`Aplicando límite de ${LIMIT} issues.`);
    targets = targets.slice(0, LIMIT);
  }

  if (targets.length === 0) {
    console.log(`🎉 No hay fallbacks que reprocesar en ${SPLIT_NAME}.`);
    process.exit(0);
  }

  console.log(`\n🎯 Issues seleccionados para reprocesar (${targets.length}): ${targets.join(', ')}\n`);

  const fileContent = await Bun.file(CSV_PATH).text();
  const allRows: any[] = parse(fileContent, { columns: true, skip_empty_lines: true });
  const rowMap = new Map<string, any>();
  for (const r of allRows) {
    rowMap.set(String(r.issue_number), r);
  }

  let successCount = 0;
  let failCount = 0;
  let consecutiveFailures = 0;
  let abortTriggered = false;

  console.log(`🛑 Circuit Breaker: abortar tras ${MAX_CONSECUTIVE_FAILURES} fallos consecutivos (igual a workers)\n`);

  // Ejecución paralela con límite de concurrencia
  const queue = [...targets];
  async function runWorker(workerId: number) {
    const workerDir = `/tmp/opencode-fallback-w${workerId}`;
    await Bun.$`mkdir -p ${workerDir}`.quiet();

    while (queue.length > 0) {
      if (abortTriggered) break;
      const issueNum = queue.shift();
      if (!issueNum) break;

      const row = rowMap.get(issueNum);
      if (!row) {
        console.error(`[W${workerId}] Issue #${issueNum} no encontrado en ${CSV_PATH}`);
        failCount++;
        continue;
      }

      console.log(`[W${workerId}] 🚀 Reprocesando Issue #${issueNum} ("${row.issue_title?.slice(0, 50)}")...`);
      const message = buildMessage(row);

      let parsed: any = null;
      let latencyMs = 0;

      for (let attempt = 0; attempt < 2; attempt++) {
        if (abortTriggered) break;
        try {
          const t0 = performance.now();
          const raw = PROVIDER === 'deepseek' ? await callDeepSeek(row) : await callOpencode(message, workerDir);
          latencyMs = Math.round(performance.now() - t0);
          const candidate: any = extractJson(raw);
          if (candidate?.fault_location?.symptoms && (!candidate.symptoms || candidate.symptoms.length === 0)) {
            candidate.symptoms = candidate.fault_location.symptoms;
            delete candidate.fault_location.symptoms;
          }
          if (candidate?.fault_location?.capabilities && (!candidate.capabilities || candidate.capabilities.length === 0)) {
            candidate.capabilities = candidate.fault_location.capabilities;
            delete candidate.fault_location.capabilities;
          }
          if (isValid(candidate)) {
            parsed = candidate;
            break;
          } else {
            console.log(`[W${workerId}] candidate no pasó isValid:`, JSON.stringify(candidate));
          }
        } catch (err) {
          console.error(`[W${workerId}] Issue #${issueNum} intento ${attempt + 1} falló: ${(err as Error).message}`);
        }
      }

      if (parsed) {
        consecutiveFailures = 0;
        const distText = toDistilledText(parsed);
        const rawText = (row.text || `${row.issue_title} ${row.description || ''}`).trim();

        const updated = {
          issue_number: String(row.issue_number),
          owner: row.owner || '',
          distilled_json: JSON.stringify(parsed),
          distilled_text: distText,
          raw_text: rawText,
          fallback: false, // ¡Éxito!
          latency_ms: latencyMs,
          model: MODEL_NAME,
        };

        existingMap.set(issueNum, updated);
        successCount++;
        console.log(`[W${workerId}] ✅ Issue #${issueNum} corregido exitosamente (${latencyMs}ms): [${parsed.fault_location.component}]`);

        // Guardar checkpoint inmediatamente
        const newJsonl = Array.from(existingMap.values()).map((r) => JSON.stringify(r)).join('\n') + '\n';
        writeFileSync(CKPT_PATH, newJsonl, 'utf-8');
      } else {
        failCount++;
        consecutiveFailures++;
        console.error(`[W${workerId}] ❌ Issue #${issueNum} volvió a fallar. Se conserva como fallback.`);
        console.warn(
          `⚠️ [Fallo consecutivo ${consecutiveFailures}/${MAX_CONSECUTIVE_FAILURES}] en reprocess_fallbacks.`
        );

        if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
          abortTriggered = true;
          console.error(
            `\n🛑======================================================================`
          );
          console.error(
            `🛑 [CIRCUIT BREAKER ACTIVADO] Se alcanzaron ${consecutiveFailures} fallos consecutivos en reprocess_fallbacks (umbral: ${MAX_CONSECUTIVE_FAILURES} workers).`
          );
          console.error(
            `🛑 El proveedor opencode parece estar caído o respondiendo con error continuo (ej. 'opencode salió con 1').`
          );
          console.error(
            `🛑 Abortando el reprocesamiento para evitar llamadas fallidas innecesarias.`
          );
          console.error(
            `🛑======================================================================\n`
          );
          process.exit(1);
        }
      }
    }
  }

  const workers = Array.from({ length: CONCURRENCY }, (_, i) => runWorker(i + 1));
  await Promise.all(workers);

  console.log(`\n======================================================`);
  console.log(`📊 RESULTADO DEL REPROCESAMIENTO:`);
  console.log(`   - Corregidos exitosamente: ${successCount}`);
  console.log(`   - Fallidos:                 ${failCount}`);
  console.log(`======================================================\n`);

  if (successCount > 0) {
    console.log(`🔄 Sincronizando CSVs y Parquet con merge_distillations.ts...`);
    const mergeProc = Bun.spawn({
      cmd: ['bun', 'run', 'scripts/merge_distillations.ts', '--target', SPLIT_NAME, '--sources', CKPT_PATH],
      stdout: 'inherit',
      stderr: 'inherit',
    });
    await mergeProc.exited;
    console.log(`\n✨ Sincronización completada.`);
  }
}

main().catch((e) => {
  console.error(`Error: ${(e as Error).message}`);
  process.exit(1);
});
