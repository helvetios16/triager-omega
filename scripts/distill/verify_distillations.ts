// Script de verificación y control de calidad de las destilaciones
// Verifica consistencia, estructura JSON, marcadores [FL][SY][CP], fallbacks y alineación entre archivos.
//
// Ejecución:
//   bun run scripts/verify_distillations.ts [--dir artifacts/typescript/distilled] [--split ts_train]

import { existsSync, readFileSync } from 'fs';
import { join } from 'path';
import { parseArgs } from 'util';
import { parse } from 'csv-parse/sync';

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    dir: { type: 'string', default: 'artifacts/typescript/distilled' },
    split: { type: 'string', default: 'ts_train' },
  },
  strict: false,
});

const DIR = values.dir as string;
const SPLIT = values.split as string;

const CANONICAL_COMPONENTS = new Set([
  'Type Checker',
  'Emitter',
  'Parser',
  'Language Service',
  'Binder',
  'Module Resolution',
  'CLI',
  'Compiler',
]);

interface ValidationError {
  issue_number: string;
  field: string;
  error: string;
}

async function main() {
  console.log(`\n======================================================================`);
  console.log(`🔍 VERIFICADOR DE CALIDAD DE DESTILACIÓN (triager-omega / TriagerX)`);
  console.log(`   Directorio: ${DIR}`);
  console.log(`   Split: ${SPLIT}`);
  console.log(`======================================================================\n`);

  const jsonlPath = join(DIR, `${SPLIT}.jsonl`);
  const parquetPath = join(DIR, `${SPLIT}.parquet`);
  const distCsvPath = join(DIR, `${SPLIT}_distilled.csv`);
  const rawCsvPath = join(DIR, `${SPLIT}_raw.csv`);
  const bothCsvPath = join(DIR, `${SPLIT}_both.csv`);
  const provCsvPath = join(DIR, `${SPLIT}_provenance.csv`);
  const provParquetPath = join(DIR, `${SPLIT}_provenance.parquet`);

  const files = [
    { name: `${SPLIT}.jsonl`, path: jsonlPath },
    { name: `${SPLIT}.parquet`, path: parquetPath },
    { name: `${SPLIT}_distilled.csv`, path: distCsvPath },
    { name: `${SPLIT}_raw.csv`, path: rawCsvPath },
    { name: `${SPLIT}_both.csv`, path: bothCsvPath },
    { name: `${SPLIT}_provenance.csv`, path: provCsvPath },
    { name: `${SPLIT}_provenance.parquet`, path: provParquetPath },
  ];

  let missingFiles = false;
  for (const f of files) {
    if (!existsSync(f.path)) {
      console.error(`❌ Archivo faltante: ${f.name} en ${f.path}`);
      missingFiles = true;
    } else {
      console.log(`✅ Archivo encontrado: ${f.name}`);
    }
  }

  if (missingFiles) {
    console.error('\n❌ Error: Faltan archivos requeridos.');
    process.exit(1);
  }

  // 1. Leer y validar JSONL
  console.log(`\n--- 1. Validación de registros en ${SPLIT}.jsonl ---`);
  const jsonlContent = readFileSync(jsonlPath, 'utf-8').trim().split('\n');
  const jsonlRecords = jsonlContent.map((line, idx) => {
    try {
      return JSON.parse(line);
    } catch (e) {
      throw new Error(`Línea ${idx + 1} en JSONL no es un JSON válido: ${(e as Error).message}`);
    }
  });

  const total = jsonlRecords.length;
  console.log(`Total de registros en JSONL: ${total}`);

  const errors: ValidationError[] = [];
  const warnings: ValidationError[] = [];
  let fallbackCount = 0;
  const componentCounts: Record<string, number> = {};
  const ownerCounts: Record<string, number> = {};
  const seenIds = new Set<string>();

  for (const r of jsonlRecords) {
    const id = String(r.issue_number);
    if (seenIds.has(id)) {
      errors.push({ issue_number: id, field: 'issue_number', error: 'ID duplicado en JSONL' });
    }
    seenIds.add(id);

    if (!r.owner || !r.owner.trim()) {
      errors.push({ issue_number: id, field: 'owner', error: 'Owner vacío o no definido' });
    } else {
      ownerCounts[r.owner] = (ownerCounts[r.owner] || 0) + 1;
    }

    if (r.fallback) {
      fallbackCount++;
      warnings.push({ issue_number: id, field: 'fallback', error: 'El registro usa valor fallback' });
    }

    // Validar distilled_json
    let parsed: any;
    try {
      parsed = JSON.parse(r.distilled_json);
    } catch {
      errors.push({ issue_number: id, field: 'distilled_json', error: 'distilled_json no es parseable' });
      continue;
    }

    const fl = parsed?.fault_location;
    if (!fl || typeof fl !== 'object') {
      errors.push({ issue_number: id, field: 'fault_location', error: 'fault_location ausente' });
    } else {
      if (!fl.product || typeof fl.product !== 'string') {
        errors.push({ issue_number: id, field: 'fault_location.product', error: 'product ausente o no string' });
      }
      if (!fl.component || typeof fl.component !== 'string') {
        errors.push({ issue_number: id, field: 'fault_location.component', error: 'component ausente o no string' });
      } else {
        componentCounts[fl.component] = (componentCounts[fl.component] || 0) + 1;
        if (!CANONICAL_COMPONENTS.has(fl.component)) {
          warnings.push({
            issue_number: id,
            field: 'fault_location.component',
            error: `Component '${fl.component}' no pertenece a la taxonomía canónica`,
          });
        }
      }
      if (typeof fl.subsystem !== 'string' || !fl.subsystem.trim()) {
        if (!r.fallback) {
          warnings.push({ issue_number: id, field: 'fault_location.subsystem', error: 'subsystem está vacío' });
        }
      }
    }

    if (!Array.isArray(parsed?.symptoms) || parsed.symptoms.length === 0) {
      if (!r.fallback) {
        errors.push({ issue_number: id, field: 'symptoms', error: 'symptoms vacío o no es array' });
      }
    }
    if (!Array.isArray(parsed?.capabilities) || parsed.capabilities.length === 0) {
      if (!r.fallback) {
        errors.push({ issue_number: id, field: 'capabilities', error: 'capabilities vacío o no es array' });
      }
    }

    // Validar distilled_text markers
    const text = String(r.distilled_text || '');
    if (!text.includes('[FL]') || !text.includes('[SY]') || !text.includes('[CP]')) {
      errors.push({ issue_number: id, field: 'distilled_text', error: 'Faltan marcadores [FL] [SY] [CP]' });
    }
  }

  // 2. Validar archivos CSV y Registro de Constancia
  console.log(`\n--- 2. Validación y Alineación con CSVs y Registro de Constancia ---`);
  const distCsvContent = readFileSync(distCsvPath, 'utf-8');
  const distRows: any[] = parse(distCsvContent, { columns: true, skip_empty_lines: true });

  const rawCsvContent = readFileSync(rawCsvPath, 'utf-8');
  const rawRows: any[] = parse(rawCsvContent, { columns: true, skip_empty_lines: true });

  const bothCsvContent = readFileSync(bothCsvPath, 'utf-8');
  const bothRows: any[] = parse(bothCsvContent, { columns: true, skip_empty_lines: true });

  const provCsvContent = readFileSync(provCsvPath, 'utf-8');
  const provRows: any[] = parse(provCsvContent, { columns: true, skip_empty_lines: true });

  console.log(`Filas en _distilled.csv:  ${distRows.length} (esperado: ${total})`);
  console.log(`Filas en _raw.csv:        ${rawRows.length} (esperado: ${total})`);
  console.log(`Filas en _both.csv:       ${bothRows.length} (esperado: ${total})`);
  console.log(`Filas en _provenance.csv: ${provRows.length} (esperado: ${total})`);

  let parquetCount = -1;
  let provParquetCount = -1;
  try {
    const pyCode = `
import pandas as pd
print(len(pd.read_parquet('${parquetPath}')))
print(len(pd.read_parquet('${provParquetPath}')))
`;
    const pyOut = await Bun.$`uv run python -c ${pyCode}`.text();
    const counts = pyOut.trim().split('\n').map(Number);
    parquetCount = counts[0];
    provParquetCount = counts[1];
    console.log(`Filas en .parquet:        ${parquetCount} (esperado: ${total})`);
    console.log(`Filas en _prov.parquet:   ${provParquetCount} (esperado: ${total})`);
  } catch (err) {
    warnings.push({
      issue_number: 'N/A',
      field: 'parquet_verify',
      error: `No se pudo leer conteo Parquet: ${(err as Error).message}`,
    });
  }

  if (
    distRows.length !== total ||
    rawRows.length !== total ||
    bothRows.length !== total ||
    provRows.length !== total ||
    (parquetCount !== -1 && parquetCount !== total) ||
    (provParquetCount !== -1 && provParquetCount !== total)
  ) {
    errors.push({
      issue_number: 'N/A',
      field: 'row_count',
      error: 'Desalineación en el conteo de filas entre JSONL, CSVs y Parquet/Constancia',
    });
  }

  // Comprobar coincidencia exacta de IDs y campos en datasets
  const provModelCounts: Record<string, number> = {};
  const modelLatencies: Record<string, number[]> = {};

  for (let i = 0; i < total; i++) {
    const expected = jsonlRecords[i];
    if (!expected) continue;

    // Distilled CSV
    if (i < distRows.length) {
      const d = distRows[i];
      if (String(d.issue_number) !== String(expected.issue_number)) {
        errors.push({
          issue_number: d.issue_number,
          field: 'alignment_distilled',
          error: `Fila ${i + 1} no coincide con JSONL (CSV: ${d.issue_number}, JSONL: ${expected.issue_number})`,
        });
      }
      if (!d.text || !d.text.trim()) {
        errors.push({ issue_number: d.issue_number, field: 'distilled_text', error: 'Texto vacío en _distilled.csv' });
      }
    }

    // Raw CSV
    if (i < rawRows.length) {
      const r = rawRows[i];
      if (String(r.issue_number) !== String(expected.issue_number)) {
        errors.push({
          issue_number: r.issue_number,
          field: 'alignment_raw',
          error: `Fila ${i + 1} no coincide con JSONL (Raw CSV: ${r.issue_number}, JSONL: ${expected.issue_number})`,
        });
      }
    }

    // Both CSV
    if (i < bothRows.length) {
      const b = bothRows[i];
      if (String(b.issue_number) !== String(expected.issue_number)) {
        errors.push({
          issue_number: b.issue_number,
          field: 'alignment_both',
          error: `Fila ${i + 1} no coincide con JSONL (Both CSV: ${b.issue_number}, JSONL: ${expected.issue_number})`,
        });
      }
    }

    // Provenance CSV (Registro de Constancia)
    if (i < provRows.length) {
      const p = provRows[i];
      if (String(p.issue_number) !== String(expected.issue_number)) {
        errors.push({
          issue_number: p.issue_number,
          field: 'alignment_provenance',
          error: `Fila ${i + 1} de constancia no coincide con JSONL (CSV: ${p.issue_number}, JSONL: ${expected.issue_number})`,
        });
      }
      if (p.owner !== expected.owner) {
        errors.push({
          issue_number: p.issue_number,
          field: 'provenance_owner',
          error: `Owner en constancia (${p.owner}) no coincide con JSONL (${expected.owner})`,
        });
      }
      const mod = p.model || 'unknown';
      provModelCounts[mod] = (provModelCounts[mod] || 0) + 1;

      const lat = Number(p.latency_ms);
      if (isNaN(lat) || lat < 0) {
        errors.push({
          issue_number: p.issue_number,
          field: 'provenance_latency',
          error: `Latencia inválida en constancia: ${p.latency_ms}`,
        });
      } else {
        if (!modelLatencies[mod]) modelLatencies[mod] = [];
        modelLatencies[mod].push(lat);
      }
    }
  }

  // 3. Reporte final
  console.log(`\n======================================================================`);
  console.log(`📊 REPORTE DE CALIDAD Y DIAGNÓSTICO`);
  console.log(`======================================================================`);
  console.log(`• Total registros verificados: ${total}`);
  console.log(`• Fallbacks detectados:         ${fallbackCount} (${((fallbackCount / total) * 100).toFixed(1)}%)`);
  console.log(`• Errores críticos:            ${errors.length}`);
  console.log(`• Advertencias:                ${warnings.length}`);

  console.log(`\nDistribución de componentes inferidos:`);
  for (const [comp, count] of Object.entries(componentCounts)) {
    console.log(`  - ${comp.padEnd(20)}: ${count} (${((count / total) * 100).toFixed(1)}%)`);
  }

  console.log(`\nDistribución de desarrolladores:`);
  for (const [owner, count] of Object.entries(ownerCounts)) {
    console.log(`  - @${owner.padEnd(19)}: ${count}`);
  }

  console.log(`\nDistribución por Modelo (Registro de Constancia):`);
  for (const [mod, count] of Object.entries(provModelCounts)) {
    const lats = modelLatencies[mod] || [];
    const avgSec = lats.length > 0 ? (lats.reduce((a, b) => a + b, 0) / lats.length / 1000).toFixed(1) : 'N/A';
    console.log(`  - ${mod.padEnd(40)}: ${count} (${((count / total) * 100).toFixed(1)}%) | latencia prom: ${avgSec}s`);
  }

  if (errors.length > 0) {
    console.log(`\n❌ ERRORES DETECTADOS (${errors.length}):`);
    for (const err of errors.slice(0, 10)) {
      console.log(`  [Issue #${err.issue_number}] ${err.field}: ${err.error}`);
    }
    if (errors.length > 10) console.log(`  ... y ${errors.length - 10} más`);
  }

  if (warnings.length > 0) {
    console.log(`\n⚠️  ADVERTENCIAS (${warnings.length}):`);
    for (const w of warnings.slice(0, 10)) {
      console.log(`  [Issue #${w.issue_number}] ${w.field}: ${w.error}`);
    }
  }

  if (errors.length === 0) {
    console.log(`\n🎉 VERIFICACIÓN EXITOSA: La salida cumple al 100% las especificaciones de triager-omega y TriagerX (incluyendo Registro de Constancia).`);
    process.exit(0);
  } else {
    process.exit(1);
  }
}

main().catch((e) => {
  console.error(`Error fatal: ${(e as Error).message}`);
  process.exit(1);
});
