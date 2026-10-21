import { appendFileSync, existsSync, readFileSync } from 'fs';
import { join } from 'path';

export function escapeCsvField(val: unknown): string {
  if (val === null || val === undefined) return '';
  const str = String(val);
  if (str.includes(',') || str.includes('"') || str.includes('\n') || str.includes('\r')) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

export async function writeCsvFile(filePath: string, headers: string[], rows: Record<string, unknown>[]) {
  const headerLine = headers.join(',');
  const lines = rows.map((r) => headers.map((h) => escapeCsvField(r[h])).join(','));
  await Bun.write(filePath, [headerLine, ...lines].join('\n') + '\n');
}

export function loadExistingCheckpoint<RecordType extends { issue_number?: string | number }>(ckptPath: string): Map<string, RecordType> {
  const map = new Map<string, RecordType>();
  if (!existsSync(ckptPath)) return map;

  const content = readFileSync(ckptPath, 'utf-8');
  for (const line of content.split('\n')) {
    if (!line.trim()) continue;
    try {
      const rec = JSON.parse(line) as RecordType;
      if (rec.issue_number) map.set(String(rec.issue_number), rec);
    } catch {}
  }
  return map;
}

const flushingPaths = new Map<string, Promise<void>>();
export async function flushOutputs(
  ckptPath: string,
  outDir: string,
  splitName: string,
  model: string,
  prefix: (splitName: string) => string,
  convertParquet: (command: string, args: string[]) => Promise<void> = async (_command, args) => { await Bun.$`uv run python -c ${args[3]}`.quiet(); }
) {
  const prior = flushingPaths.get(ckptPath);
  if (prior) return prior;
  const current = Promise.resolve().then(async () => {
  try {
    const doneMap = loadExistingCheckpoint<any>(ckptPath);
    const records = Array.from(doneMap.values());
    if (records.length === 0) return;

    const distilledCsvPath = join(outDir, `${prefix(splitName)}_distilled.csv`);
    const rawCsvPath = join(outDir, `${prefix(splitName)}_raw.csv`);
    const bothCsvPath = join(outDir, `${prefix(splitName)}_both.csv`);
    const provenanceCsvPath = join(outDir, `${prefix(splitName)}_provenance.csv`);
    const provenanceParquetPath = join(outDir, `${prefix(splitName)}_provenance.parquet`);
    const parquetPath = join(outDir, `${prefix(splitName)}.parquet`);

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
        model: r.model || model,
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
      await convertParquet('uv', ['run', 'python', '-c', pyCode]);
    } catch {}

    console.log(`💾 [Flush] Guardados ${records.length} registros en Parquet, CSVs y Provenance (${distilledCsvPath})`);
  } finally {
    flushingPaths.delete(ckptPath);
  }
  });
  flushingPaths.set(ckptPath, current);
  return current;
}

const appendLocks = new Map<string, Promise<void>>();
export async function safeAppendRecord<RecordType>(ckptPath: string, record: RecordType) {
  const previous = appendLocks.get(ckptPath) ?? Promise.resolve();
  const current = previous.catch(() => {}).then(async () => {
    appendFileSync(ckptPath, JSON.stringify(record) + '\n', 'utf-8');
  });
  appendLocks.set(ckptPath, current);
  try { await current; } finally { if (appendLocks.get(ckptPath) === current) appendLocks.delete(ckptPath); }
}

export function calculateCostUsd(cacheHit: number, cacheMiss: number, outputTokens: number, prices: { hit: number; miss: number; output: number }): number {
  return (cacheHit / 1_000_000) * prices.hit + (cacheMiss / 1_000_000) * prices.miss + (outputTokens / 1_000_000) * prices.output;
}

