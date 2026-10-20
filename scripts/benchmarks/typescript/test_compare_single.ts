// Test de comparación directa: deepseek-flash vs opencode/union-alpha en un issue real (#43662)
// Verifica:
//   1. Calidad de extracción técnica (component, subsystem, symptoms, capabilities).
//   2. Cumplimiento de la Tabla de Constancia (11 columnas canónicas).
//   3. Latencia y consumo de tokens con costo exacto.

import { parse } from 'csv-parse/sync';
import { readFileSync } from 'fs';

const CSV_PATH = '/Users/sebastian/Documents/Python/triagerX/df_train_ts.csv';
const ISSUE_TO_TEST = '43662';

const API_KEY = process.env.DEEPSEEK_API_KEY;
if (!API_KEY) {
  console.error('❌ ERROR: process.env.DEEPSEEK_API_KEY no está definido.');
  process.exit(1);
}

// 1. Obtener la destilación previa de union-alpha
const unionJsonl = readFileSync('artifacts/typescript/distilled/ts_train_union.jsonl', 'utf-8');
let unionRecord: any = null;
for (const line of unionJsonl.split('\n')) {
  if (!line.trim()) continue;
  const r = JSON.parse(line);
  if (String(r.issue_number) === ISSUE_TO_TEST) {
    unionRecord = r;
    break;
  }
}

if (!unionRecord) {
  console.error(`❌ No se encontró el issue #${ISSUE_TO_TEST} en ts_train_union.jsonl`);
  process.exit(1);
}

// 2. Obtener el issue crudo desde el CSV original
const csvContent = readFileSync(CSV_PATH, 'utf-8');
const rows: any[] = parse(csvContent, { columns: true, skip_empty_lines: true });
const targetRow = rows.find((r) => String(r.issue_number) === ISSUE_TO_TEST);

if (!targetRow) {
  console.error(`❌ No se encontró el issue #${ISSUE_TO_TEST} en ${CSV_PATH}`);
  process.exit(1);
}

// 3. Preparar prompt
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

const shots = FEW_SHOT_EXAMPLES.map(
  (ex) => `Example Input:\n${ex.input}\n\nExample Output:\n${ex.output}`
).join('\n\n');
const SYSTEM_MESSAGE = `${SYSTEM_PROMPT}\n\n## Few-shot Examples\n\n${shots}`;

const comp = (targetRow.component || '').trim();
const userParts = [
  `Summary: ${targetRow.issue_title ?? ''}`,
  'Product: TypeScript',
  `Component: ${comp ? comp : '(infer canonical compiler component)'}`,
];
if (targetRow.labels?.trim()) {
  userParts.push(`Labels: ${targetRow.labels.trim()}`);
}
userParts.push('Initial report:');
userParts.push((targetRow.description ?? '').slice(0, 1500));
const userPrompt = `Now extract the structured information for this issue report:\n\n${userParts.join('\n')}`;

console.log(`\n======================================================================`);
console.log(`🔬 PRUEBA COMPARATIVA: Issue #${ISSUE_TO_TEST}`);
console.log(`   Título: ${targetRow.issue_title}`);
console.log(`   Owner:  @${targetRow.owner}`);
console.log(`======================================================================\n`);

// 4. Llamar a DeepSeek API
const start = performance.now();
const res = await fetch('https://api.deepseek.com/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${API_KEY}`,
  },
  body: JSON.stringify({
    model: 'deepseek-flash',
    messages: [
      { role: 'system', content: SYSTEM_MESSAGE },
      { role: 'user', content: userPrompt },
    ],
    response_format: { type: 'json_object' },
    temperature: 0.0,
    thinking: { type: 'disabled' },
  }),
});

const latencyMs = Math.round(performance.now() - start);

if (!res.ok) {
  const errText = await res.text();
  console.error(`❌ Error en DeepSeek API (${res.status}): ${errText}`);
  process.exit(1);
}

const data = (await res.json()) as any;
const deepseekContent = data.choices[0].message.content;
const deepseekJson = JSON.parse(deepseekContent);
const usage = data.usage || {};

// Cost calculation
const hitTokens = usage.prompt_cache_hit_tokens || 0;
const missTokens = (usage.prompt_tokens || 0) - hitTokens;
const outTokens = usage.completion_tokens || 0;
const costUsd = (hitTokens / 1e6) * 0.006 + (missTokens / 1e6) * 0.30 + (outTokens / 1e6) * 1.20;

// Construir registro de constancia canónico (11 columnas)
const unionParsed = JSON.parse(unionRecord.distilled_json);
const deepseekDistilledText = `[FL] ${[deepseekJson.fault_location?.product, deepseekJson.fault_location?.component, deepseekJson.fault_location?.subsystem].filter(Boolean).join(' ')} [SY] ${(deepseekJson.symptoms || []).join(' ')} [CP] ${(deepseekJson.capabilities || []).join(' ')}`;

const provenanceRecord = {
  issue_number: ISSUE_TO_TEST,
  issue_title: targetRow.issue_title,
  owner: targetRow.owner,
  model: 'deepseek-flash',
  latency_ms: latencyMs,
  fallback: false,
  component: deepseekJson.fault_location?.component || '',
  subsystem: deepseekJson.fault_location?.subsystem || '',
  symptoms: (deepseekJson.symptoms || []).join(' | '),
  capabilities: (deepseekJson.capabilities || []).join(' | '),
  distilled_text: deepseekDistilledText,
};

console.log('📊 COMPARATIVA TÉCNICA:\n');
console.log('┌──────────────────────┬──────────────────────────────────────────┬──────────────────────────────────────────┐');
console.log('│ Métrica / Campo      │ opencode/union-alpha                     │ deepseek-flash                           │');
console.log('├──────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────┤');
console.log(`│ Latencia             │ ${(unionRecord.latency_ms + 'ms').padEnd(40)} │ ${(latencyMs + 'ms').padEnd(40)} │`);
console.log(`│ Componente           │ ${(unionParsed.fault_location.component || '').padEnd(40)} │ ${(deepseekJson.fault_location.component || '').padEnd(40)} │`);
console.log(`│ Subsistema           │ ${(unionParsed.fault_location.subsystem || '').slice(0, 40).padEnd(40)} │ ${(deepseekJson.fault_location.subsystem || '').slice(0, 40).padEnd(40)} │`);
console.log('└──────────────────────┴──────────────────────────────────────────┴──────────────────────────────────────────┘');

console.log('\n📝 SÍNTOMAS (Symptoms):');
console.log('• union-alpha:   ', JSON.stringify(unionParsed.symptoms));
console.log('• deepseek-flash:', JSON.stringify(deepseekJson.symptoms));

console.log('\n🛠️ CAPACIDADES (Capabilities):');
console.log('• union-alpha:   ', JSON.stringify(unionParsed.capabilities));
console.log('• deepseek-flash:', JSON.stringify(deepseekJson.capabilities));

console.log('\n🔤 TEXTO FORMAL DESTILADO [FL][SY][CP]:');
console.log('• union-alpha:   ', unionRecord.distilled_text);
console.log('• deepseek-flash:', deepseekDistilledText);

console.log('\n📈 MÉTRICAS DE API Y COSTO (deepseek-flash):');
console.log(`• Input tokens:  ${usage.prompt_tokens} (Cache Hit: ${hitTokens}, Cache Miss: ${missTokens})`);
console.log(`• Output tokens: ${outTokens}`);
console.log(`• Costo real:    $${costUsd.toFixed(6)} USD (aprox ${((costUsd * 2831)).toFixed(2)} USD para los 2,831 issues)`);

console.log('\n📋 VERIFICACIÓN DE TABLA DE CONSTANCIA (11 Columnas Canónicas):');
console.log(JSON.stringify(provenanceRecord, null, 2));
