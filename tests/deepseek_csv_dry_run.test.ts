import { afterEach, expect, test } from 'bun:test';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';


// Integration tests: no rows/readRows injection and no parser mocks. Bun must
// resolve the real csv-parse package (currently available in its local cache).
const directories: string[] = [];
afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

function fixture(csv: string) {
  const directory = mkdtempSync(join(tmpdir(), 'deepseek-csv-dry-run-'));
  directories.push(directory);
  const csvPath = join(directory, 'input.csv');
  const outputDirectory = join(directory, 'output-must-not-exist');
  writeFileSync(csvPath, csv);
  return {
    directory, csvPath, outputDirectory,
    verifyNoEffects(envelope: { fetchCalls: number; conversionCalls: number }) {
      expect(envelope.fetchCalls).toBe(0);
      expect(envelope.conversionCalls).toBe(0);
      expect(existsSync(outputDirectory)).toBe(false);
      expect(readFileSync(csvPath, 'utf8')).toBe(csv);
      expect(readdirSync(directory)).toEqual(['input.csv']);
    },
  };
}

// Bun's test resolver does not use the runtime's automatic package resolution.
// Launch the actual runtime rather than mocking or redirecting csv-parse. The
// package is already cached; --prefer-offline avoids registry freshness checks.
function runtimePreview(corpus: 'ts' | 'openj9', directory: string, config: Record<string, unknown>) {
  const script = join(directory, 'driver.ts');
  const runner = join(import.meta.dir, '..', 'scripts', 'distill', `distill_deepseek_${corpus}.ts`);
  writeFileSync(script, `
    import { CONFIG, run } from ${JSON.stringify(runner)};
    let fetchCalls = 0;
    let conversionCalls = 0;
    const result = await run({ ...CONFIG, ...${JSON.stringify(config)} }, {
      fetch: async () => { fetchCalls++; throw new Error('unexpected HTTP'); },
      convertParquet: async () => { conversionCalls++; throw new Error('unexpected conversion'); },
    });
    console.log(JSON.stringify({ result, fetchCalls, conversionCalls }));
  `);
  try {
    const child = Bun.spawnSync([process.execPath, '--prefer-offline', 'run', script], {
      stdout: 'pipe', stderr: 'pipe',
    });
    expect(child.exitCode, child.stderr.toString()).toBe(0);
    return JSON.parse(child.stdout.toString());
  } finally {
    rmSync(script, { force: true });
  }
}

test('TypeScript loads a real CSV and previews the selected quoted multiline row without effects', () => {
  const input = fixture([
    'issue_number,owner,issue_title,description,component,labels',
    '1,ignored,Outside selected range,ignored,Parser,Bug',
    '2,developer,"Quoted, ""TypeScript"" regression","line one\nline two",Type Checker,"Bug, Fixed"',
    '',
  ].join('\n'));
  const envelope = runtimePreview('ts', input.directory, {
    csvPath: input.csvPath,
    outputDirectory: input.outputDirectory,
    apiKey: '',
    dryRun: true,
    split: 'train',
    offset: 1,
    limit: 1,
    extraCheckpoints: [],
    model: 'offline-csv-check',
    thinking: true,
  });
  const { result } = envelope;

  expect(result.preview?.user).toBe([
    'Now extract the structured information for this issue report:',
    '',
    'Summary: Quoted, "TypeScript" regression',
    'Product: TypeScript',
    'Component: Type Checker',
    'Labels: Bug, Fixed',
    'Initial report:',
    'line one',
    'line two',
  ].join('\n'));
  expect(result.preview?.system).toContain('expert compiler engineer');
  expect(result.preview?.body).toEqual({
    model: 'offline-csv-check', response_format: { type: 'json_object' },
    temperature: 0, thinking: { type: 'enabled' },
  });
  expect(result.paths).toEqual({
    checkpoint: join(input.outputDirectory, 'ts_train_deepseek.jsonl'),
    outputDirectory: input.outputDirectory,
    split: 'ts_train_deepseek',
  });
  expect(result.summary).toEqual({ completed: 0, fallbacks: 0, aborted: false, skipped: 0 });
  expect(result.statistics.costUsd).toBe(0);
  expect(result.records).toEqual([]);
  input.verifyNoEffects(envelope);
});

test('OpenJ9 loads a real CSV and normalizes its component in a no-effects preview', () => {
  const input = fixture([
    'issue_number,owner,issue_title,issue_body,component',
    '1,ignored,Outside selected range,ignored,comp:vm',
    '2,developer,"Quoted, ""OpenJ9"" crash","first JVM line\nsecond JVM line",comp:jit:aot',
    '',
  ].join('\n'));
  const envelope = runtimePreview('openj9', input.directory, {
    csvPath: input.csvPath,
    outputDirectory: input.outputDirectory,
    apiKey: '',
    dryRun: true,
    split: 'test',
    offset: 1,
    limit: 1,
    extraCheckpoints: [],
    model: 'offline-jvm-check',
  });
  const { result } = envelope;

  expect(result.preview?.user).toBe([
    'Now extract the structured information for this OpenJ9 issue report:',
    '',
    'Summary: Quoted, "OpenJ9" crash',
    'Product: OpenJ9',
    'Component: JIT',
    'Initial report:',
    'first JVM line',
    'second JVM line',
  ].join('\n'));
  expect(result.preview?.system).toContain('expert Java Virtual Machine');
  expect(result.preview?.body).toEqual({
    model: 'offline-jvm-check', response_format: { type: 'json_object' },
    temperature: 0, thinking: { type: 'disabled' },
  });
  expect(result.paths).toEqual({
    checkpoint: join(input.outputDirectory, 'openj9_test.jsonl'),
    outputDirectory: input.outputDirectory,
    split: 'test',
  });
  expect(result.summary).toEqual({ completed: 0, fallbacks: 0, aborted: false, skipped: 0 });
  expect(result.statistics.costUsd).toBe(0);
  expect(result.records).toEqual([]);
  input.verifyNoEffects(envelope);
});
