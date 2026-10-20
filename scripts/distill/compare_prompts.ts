// Comparativa competitiva: Prompt Original (Mozilla/genérico) vs Prompt Nuevo (Compiler/TypeScript)
// Ejecución: bun run scripts/compare_prompts.ts

import { parse } from 'csv-parse/sync';

const CSV_PATH = '/Users/sebastian/Documents/Python/triagerX/df_train_ts.csv';
const OPENCODE_BIN = 'opencode';
const OPENCODE_MODEL = 'opencode/union-alpha';
const OPENCODE_DIR = '/tmp/opencode-distill';
const OPENCODE_TIMEOUT_MS = 120_000;
const MAX_CHARS = 1500;

// ============================================================================
// 1. CONFIGURACIÓN DEL PROMPT ORIGINAL (bun-nvidia/index-opencode.ts)
// ============================================================================
const OLD_SYSTEM_PROMPT = `You are an expert software engineer that reads bug reports and extracts structured information.

Return ONLY a raw JSON object — no markdown, no code blocks, no extra text.
The JSON must have exactly this structure:
{
  "fault_location": {
    "product": "<exact product name from input>",
    "component": "<exact component name from input>",
    "subsystem": "<inferred subsystem, max 5 words, or empty string>"
  },
  "symptoms": ["<short symptom 1, max 6 words>", "..."],
  "capabilities": ["<technical skill 1>", "..."]
}
Rules:
- fault_location.product and fault_location.component must be copied exactly from the input.
- symptoms: 1 to 5 items describing what the user observes.
- capabilities: 1 to 5 concrete technical skills a developer needs to fix this bug.
- If information is insufficient, use empty list [] or empty string "".
`;

const OLD_FEW_SHOT_EXAMPLES = [
  {
    input: `Summary: Bookmarks toolbar disappears after update
Product: Firefox
Component: Bookmarks & History
Severity: major
Priority: P2
Initial report:
After upgrading to Firefox 125, the bookmarks toolbar is gone. Checked View > Toolbars but it shows enabled. Restarting does not help. Profile is not corrupted (tested with new profile — same issue).`,
    output: `{
  "fault_location": {
    "product": "Firefox",
    "component": "Bookmarks & History",
    "subsystem": "toolbar visibility state"
  },
  "symptoms": ["toolbar disappears after update", "toolbar shows as enabled in menu", "persists after restart"],
  "capabilities": ["Firefox toolbar UI", "profile migration", "XUL/CSS layout"]
}`,
  },
  {
    input: `Summary: Email with large attachment causes Thunderbird to freeze
Product: Thunderbird
Component: Message Compose Window
Severity: major
Priority: P2
Initial report:
When attaching a file larger than 50MB and clicking Send, the compose window freezes indefinitely. CPU usage spikes to 100%. Only force-quitting resolves it. Tested on Windows 11 and Linux.`,
    output: `{
  "fault_location": {
    "product": "Thunderbird",
    "component": "Message Compose Window",
    "subsystem": "large attachment send path"
  },
  "symptoms": ["UI freeze on send", "100% CPU spike", "requires force quit"],
  "capabilities": ["MIME encoding", "async IO", "attachment streaming", "cross-platform threading"]
}`,
  },
];

function buildOldInput(row: any): string {
  return [
    `Summary: ${row.issue_title ?? ''}`,
    'Product: TypeScript',
    `Component: ${row.component ?? ''}`,
    'Initial report:',
    (row.description ?? '').slice(0, MAX_CHARS),
  ].join('\n');
}

function buildOldMessage(row: any): string {
  const shots = OLD_FEW_SHOT_EXAMPLES.map(
    (ex) => `Input:\n${ex.input}\n\nOutput:\n${ex.output}`,
  ).join('\n\n');
  return `${OLD_SYSTEM_PROMPT}\n\n## Examples\n\n${shots}\n\n## Now extract this one\n\n${buildOldInput(row)}`;
}

// ============================================================================
// 2. CONFIGURACIÓN DEL PROMPT NUEVO (Especializado en Compilador / Triage)
// ============================================================================
const NEW_SYSTEM_PROMPT = `You are an expert compiler engineer who reads issue reports and extracts structured technical information for developer triage.

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
- fault_location.product: must be copied exactly from the input.
- fault_location.component: if provided in input, use it. If empty or '(infer canonical compiler component)', classify into one of the canonical compiler components: Type Checker, Emitter, Parser, Language Service, Binder, Module Resolution, or CLI.
- fault_location.subsystem: the specific language construct or compiler pass involved.
- symptoms: 1 to 5 concise items describing the observed failure, unexpected error, or incorrect code generation.
- capabilities: 1 to 5 concrete compiler engineering skills needed to resolve the issue (e.g. "generic type inference", "AST transformation", "sourcemap generation"). Avoid generic labels like "TypeScript" or "debugging".
- If information is insufficient, use empty list [] or empty string "".
`;

const NEW_FEW_SHOT_EXAMPLES = [
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

function buildNewInput(row: any): string {
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
  parts.push((row.description ?? '').slice(0, MAX_CHARS));
  return parts.join('\n');
}

function buildNewMessage(row: any): string {
  const shots = NEW_FEW_SHOT_EXAMPLES.map(
    (ex) => `Input:\n${ex.input}\n\nOutput:\n${ex.output}`,
  ).join('\n\n');
  return `${NEW_SYSTEM_PROMPT}\n\n## Examples\n\n${shots}\n\n## Now extract this one\n\n${buildNewInput(row)}`;
}

// ============================================================================
// HELPERS COMUNES
// ============================================================================
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

async function callOpencode(message: string): Promise<string> {
  const proc = Bun.spawn({
    cmd: [OPENCODE_BIN, 'run', '-m', OPENCODE_MODEL, '--format', 'json', '--dir', OPENCODE_DIR, message],
    cwd: OPENCODE_DIR,
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
  if (proc.exitCode !== 0) throw new Error(`opencode code ${proc.exitCode}: ${err.slice(0, 150)}`);
  const texts: string[] = [];
  for (const line of out.split('\n')) {
    if (!line.trim()) continue;
    try {
      const ev = JSON.parse(line);
      if (ev.type === 'text' && typeof ev.part?.text === 'string') texts.push(ev.part.text);
    } catch {}
  }
  return texts.join('');
}

function toDistilledText(d: any): string {
  const fl = [d?.fault_location?.product, d?.fault_location?.component, d?.fault_location?.subsystem]
    .filter(Boolean).join(' ');
  const sy = Array.isArray(d?.symptoms) ? d.symptoms.join(' ') : '';
  const cp = Array.isArray(d?.capabilities) ? d.capabilities.join(' ') : '';
  return `[FL] ${fl} [SY] ${sy} [CP] ${cp}`.trim();
}

async function runPrompt(label: string, message: string) {
  const t0 = performance.now();
  const raw = await callOpencode(message);
  const ms = Math.round(performance.now() - t0);
  const parsed = extractJson(raw);
  return { label, ms, parsed, text: toDistilledText(parsed) };
}

// ============================================================================
// MAIN: CORRIDA COMPETITIVA
// ============================================================================
async function main() {
  await Bun.$`mkdir -p ${OPENCODE_DIR}`.quiet();

  const fileContent = await Bun.file(CSV_PATH).text();
  const rows: any[] = parse(fileContent, { columns: true, skip_empty_lines: true });

  // Probamos sobre 2 casos distintos:
  // Fila 0 (issue #1): Propuesta de lenguaje / tipos (Suggestion: 'protected' modifier)
  // Fila 8 (issue #41): Bug de compilador / emisión ([Emit] Module name not underscored...)
  const testIndices = [0, 8];

  console.log(`\n======================================================================`);
  console.log(`🏆 INICIANDO CORRIDA COMPETITIVA: PROMPT VIEJO vs PROMPT NUEVO`);
  console.log(`Modelo: ${OPENCODE_MODEL} (OpenCode CLI)`);
  console.log(`======================================================================\n`);

  for (const idx of testIndices) {
    const row = rows[idx];
    console.log(`\n----------------------------------------------------------------------`);
    console.log(`📌 CASO DE PRUEBA: Issue #${row.issue_number} - "${row.issue_title}"`);
    console.log(`   Asignado original (Ground Truth): @${row.owner}`);
    console.log(`----------------------------------------------------------------------`);

    console.log(`\n[1/2] Ejecutando con PROMPT VIEJO (Mozilla/Genérico)...`);
    const oldRes = await runPrompt('PROMPT VIEJO', buildOldMessage(row));
    console.log(`  Completado en ${oldRes.ms}ms`);

    console.log(`\n[2/2] Ejecutando con PROMPT NUEVO (Compiler/Especializado)...`);
    const newRes = await runPrompt('PROMPT NUEVO', buildNewMessage(row));
    console.log(`  Completado en ${newRes.ms}ms`);

    console.log(`\n📊 RESULTADOS COMPARATIVOS PARA ISSUE #${row.issue_number}:`);
    console.log(`\n--- [PROMPT VIEJO] (${oldRes.ms}ms) ---`);
    console.log(JSON.stringify(oldRes.parsed, null, 2));
    console.log(`Texto serializado:`);
    console.log(`👉 ${oldRes.text}`);

    console.log(`\n--- [PROMPT NUEVO] (${newRes.ms}ms) ---`);
    console.log(JSON.stringify(newRes.parsed, null, 2));
    console.log(`Texto serializado:`);
    console.log(`👉 ${newRes.text}`);
    console.log(`----------------------------------------------------------------------\n`);
  }

  console.log(`\n🏁 Corrida competitiva completada.`);
}

main().catch((e) => {
  console.error(`Error: ${(e as Error).message}`);
  process.exit(1);
});
