// Script de unión y consolidación de checkpoints de destilación.
//
// Combina múltiples archivos .jsonl (ej. ts_train.jsonl, ts_train_union.jsonl, ts_train_deepseek.jsonl)
// deduplica por issue_number, ordena según el dataset original y regenera:
//   - <split>.jsonl
//   - <split>_distilled.csv
//   - <split>_raw.csv
//   - <split>_both.csv
//   - <split>.parquet
//
// Ejecución:
//   bun run scripts/merge_distillations.ts [--target ts_train] [--sources ts_train.jsonl,ts_train_union.jsonl]

import { parse } from 'csv-parse/sync';
import { parseArgs } from 'util';
import { existsSync, readdirSync, readFileSync } from 'fs';
import { join } from 'path';

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    target: { type: 'string', default: 'ts_train' },
    outDir: { type: 'string', default: 'artifacts/typescript/distilled' },
    'out-dir': { type: 'string', default: 'artifacts/typescript/distilled' },
    sources: { type: 'string', default: '' },
    csv: { type: 'string', default: '/Users/sebastian/Documents/Python/triagerX/df_train_ts.csv' },
  },
  strict: false,
});

const TARGET = (values.target as string) || 'ts_train';
const OUT_DIR = (values['out-dir'] || values.outDir || 'artifacts/typescript/distilled') as string;
const CSV_PATH = (values.csv as string) || '/Users/sebastian/Documents/Python/triagerX/df_train_ts.csv';

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

async function main() {
  console.log(`\n======================================================`);
  console.log(`🔄 Consolidación y Merge de Destilaciones`);
  console.log(`======================================================`);
  console.log(`📁 Directorio: ${OUT_DIR}`);
  console.log(`🎯 Target final: ${TARGET}`);
  console.log(`📂 Referencia dataset: ${CSV_PATH}\n`);

  let sourceFiles: string[] = [];

  if (values.sources) {
    sourceFiles = (values.sources as string)
      .split(',')
      .map((s) => s.trim())
      .map((s) => {
        if (existsSync(s)) return s;
        if (existsSync(join(OUT_DIR, s))) return join(OUT_DIR, s);
        return s;
      });
  } else {
    // Detectar automáticamente archivos pertenecientes al split target
    const allFiles = readdirSync(OUT_DIR);
    sourceFiles = allFiles
      .filter(
        (f) =>
          f.endsWith('.jsonl') &&
          (f === `${TARGET}.jsonl` ||
            (f.startsWith(`${TARGET}_`) && !f.includes('union') && (f.includes('part') || f.includes('ckpt') || f.includes('deepseek'))))
      )
      .map((f) => join(OUT_DIR, f));

    // Si no encontró variantes específicas, usar el archivo target directo
    if (sourceFiles.length === 0 && existsSync(join(OUT_DIR, `${TARGET}.jsonl`))) {
      sourceFiles = [join(OUT_DIR, `${TARGET}.jsonl`)];
    }
  }

  sourceFiles = sourceFiles.filter((f) => existsSync(f));

  if (sourceFiles.length === 0) {
    console.error(`❌ No se encontraron archivos de entrada para fusionar en ${OUT_DIR}`);
    process.exit(1);
  }

  console.log(`Archivos de origen encontrados:`);
  sourceFiles.forEach((f) => console.log(`  - ${f}`));

  const mergedMap = new Map<string, CheckpointRecord>();
  let totalRead = 0;

  for (const file of sourceFiles) {
    const content = readFileSync(file, 'utf-8');
    let fileCount = 0;
    for (const line of content.split('\n')) {
      if (!line.trim()) continue;
      try {
        const rec = JSON.parse(line) as CheckpointRecord;
        if (rec.issue_number) {
          totalRead++;
          fileCount++;
          // Si ya existe uno sin fallback, priorizar el que no sea fallback
          if (!mergedMap.has(String(rec.issue_number)) || (mergedMap.get(String(rec.issue_number))?.fallback && !rec.fallback)) {
            mergedMap.set(String(rec.issue_number), rec);
          }
        }
      } catch {}
    }
    console.log(`  ✔ Leídos ${fileCount} registros de ${file}`);
  }

  console.log(`\nTotal leídos (incluyendo posibles duplicados): ${totalRead}`);
  console.log(`Total únicos deduplicados: ${mergedMap.size}`);

  // Ordenar respetando el orden original de df_train_ts.csv si existe
  let orderedRecords: CheckpointRecord[] = [];
  const titleMap = new Map<string, string>();
  if (existsSync(CSV_PATH)) {
    const csvContent = await Bun.file(CSV_PATH).text();
    const rows = parse(csvContent, { columns: true, skip_empty_lines: true }) as { issue_number: string; issue_title?: string }[];
    for (const row of rows) {
      if (row.issue_title) titleMap.set(String(row.issue_number), row.issue_title);
    }
    for (const row of rows) {
      const rec = mergedMap.get(String(row.issue_number));
      if (rec) {
        orderedRecords.push(rec);
        mergedMap.delete(String(row.issue_number));
      }
    }
    // Agregar cualquier sobrante no listado en el CSV
    for (const rec of mergedMap.values()) {
      orderedRecords.push(rec);
    }
  } else {
    orderedRecords = Array.from(mergedMap.values());
  }

  // Rutas finales
  const targetJsonl = join(OUT_DIR, `${TARGET}.jsonl`);
  const distilledCsvPath = join(OUT_DIR, `${TARGET}_distilled.csv`);
  const rawCsvPath = join(OUT_DIR, `${TARGET}_raw.csv`);
  const bothCsvPath = join(OUT_DIR, `${TARGET}_both.csv`);
  const provenanceCsvPath = join(OUT_DIR, `${TARGET}_provenance.csv`);
  const provenanceParquetPath = join(OUT_DIR, `${TARGET}_provenance.parquet`);
  const parquetPath = join(OUT_DIR, `${TARGET}.parquet`);

  // 1. Escribir .jsonl consolidado
  const jsonlLines = orderedRecords.map((r) => JSON.stringify(r)).join('\n') + '\n';
  await Bun.write(targetJsonl, jsonlLines);
  console.log(`\n💾 Guardado JSONL consolidado: ${targetJsonl}`);

  // 2. Escribir CSVs
  const distRows = orderedRecords.map((r) => ({
    issue_number: r.issue_number,
    owner: r.owner,
    text: r.distilled_text,
  }));
  await writeCsvFile(distilledCsvPath, ['issue_number', 'owner', 'text'], distRows);
  console.log(`💾 Guardado CSV distilled: ${distilledCsvPath}`);

  const rawRows = orderedRecords.map((r) => ({
    issue_number: r.issue_number,
    owner: r.owner,
    text: r.raw_text,
  }));
  await writeCsvFile(rawCsvPath, ['issue_number', 'owner', 'text'], rawRows);
  console.log(`💾 Guardado CSV raw: ${rawCsvPath}`);

  const bothRows = orderedRecords.map((r) => ({
    issue_number: r.issue_number,
    owner: r.owner,
    text: `${r.raw_text} ${r.distilled_text}`.trim(),
  }));
  await writeCsvFile(bothCsvPath, ['issue_number', 'owner', 'text'], bothRows);
  console.log(`💾 Guardado CSV both: ${bothCsvPath}`);

  // 2b. Escribir Provenance CSV (registro de constancia y auditoría por modelo)
  const provRows = orderedRecords.map((r) => {
    let component = '';
    let subsystem = '';
    let symptomsStr = '';
    let capabilitiesStr = '';
    try {
      const parsed = JSON.parse(r.distilled_json);
      component = parsed.fault_location?.component || '';
      subsystem = parsed.fault_location?.subsystem || '';
      if (Array.isArray(parsed.symptoms)) {
        symptomsStr = parsed.symptoms.join(' | ');
      }
      if (Array.isArray(parsed.capabilities)) {
        capabilitiesStr = parsed.capabilities.join(' | ');
      }
    } catch {}

    let issueTitle = titleMap.get(String(r.issue_number)) || '';
    if (!issueTitle && r.raw_text) {
      const m = r.raw_text.match(/Bug Title:\s*(.*)/);
      if (m) issueTitle = m[1].trim();
    }

    return {
      issue_number: r.issue_number,
      issue_title: issueTitle,
      owner: r.owner,
      model: r.model || 'unknown',
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
  console.log(`💾 Guardado CSV provenance (registro de constancia): ${provenanceCsvPath}`);

  // 3. Escribir Parquets (Dataset principal + Registro de Constancia)
  const pyCode = `
import pandas as pd
df = pd.read_json('${targetJsonl}', lines=True)
df.to_parquet('${parquetPath}', index=False)

df_prov = pd.read_csv('${provenanceCsvPath}')
df_prov.to_parquet('${provenanceParquetPath}', index=False)
`;
  try {
    await Bun.$`uv run python -c ${pyCode}`.quiet();
    console.log(`💾 Guardado Parquet principal: ${parquetPath}`);
    console.log(`💾 Guardado Parquet provenance: ${provenanceParquetPath}`);
  } catch (err) {
    console.error(`Advertencia al generar Parquets: ${(err as Error).message}`);
  }

  const fallbacks = orderedRecords.filter((r) => r.fallback).length;
  const modelCounts: Record<string, number> = {};
  for (const r of orderedRecords) {
    const m = r.model || 'unknown';
    modelCounts[m] = (modelCounts[m] || 0) + 1;
  }

  console.log(`\n✅ Consolidación finalizada con éxito:`);
  console.log(`   - Total registros consolidados: ${orderedRecords.length}`);
  console.log(`   - Fallbacks: ${fallbacks} (${((fallbacks / orderedRecords.length) * 100).toFixed(1)}%)`);
  console.log(`   - Desglose por Modelo:`);
  for (const [mod, count] of Object.entries(modelCounts)) {
    console.log(`     • ${mod}: ${count} issues (${((count / orderedRecords.length) * 100).toFixed(1)}%)`);
  }
}

main().catch((err) => {
  console.error(`Error fatal: ${(err as Error).message}`);
  process.exit(1);
});
