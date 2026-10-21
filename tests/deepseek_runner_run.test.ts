import { afterEach, describe, expect, test } from 'bun:test';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { CONFIG as TS_CONFIG, configFromArgs as tsConfigFromArgs, run as runTs } from '../scripts/distill/distill_deepseek_ts';
import { CONFIG as OPENJ9_CONFIG, configFromArgs as openj9ConfigFromArgs, run as runOpenJ9 } from '../scripts/distill/distill_deepseek_openj9';

const dirs: string[] = [];
function temp() { const dir = mkdtempSync(join(tmpdir(), 'distill-run-')); dirs.push(dir); return dir; }
afterEach(() => { for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true }); });
const response = (content: string) => Response.json({ choices: [{ message: { content } }], usage: { prompt_tokens: 10, completion_tokens: 4, prompt_cache_hit_tokens: 3, prompt_cache_miss_tokens: 0 } });

describe('stateless corpus runners', () => {
  test('TypeScript runs in memory, accounts transport on invalid schema, and skips completed fallback on repeat', async () => {
    const dir = temp(); const checkpointPath = join(dir, 'ts.jsonl'); let calls = 0;
    const config = { ...TS_CONFIG, apiKey: 'test-secret', checkpointPath, outputDirectory: dir, extraCheckpoints: [], limit: 0, offset: 0, concurrency: 2, flushEvery: 1 };
    const deps = { rows: [{ issue_number: '1', issue_title: 'one', description: 'body', component: 'Checker', owner: 'dev' }],
      fetch: (async () => { calls++; return response('{"fault_location":{"product":2},"symptoms":[],"capabilities":[]}'); }) as typeof fetch,
      sleep: async () => {}, now: () => 10, random: () => 0, convertParquet: async () => {} };
    const first = await runTs(config, deps);
    expect(first.summary).toMatchObject({ completed: 1, fallbacks: 1, skipped: 0 });
    expect(first.statistics).toMatchObject({ promptTokens: 10, completionTokens: 4, cacheHitTokens: 3, cacheMissTokens: 7 });
    expect(first.records[0].prompt_tokens).toBe(0);
    const second = await runTs(config, { ...deps, fetch: (async () => { throw new Error('must skip'); }) as typeof fetch });
    expect(second.summary).toMatchObject({ completed: 0, skipped: 1 });
    expect(calls).toBe(1);
    expect(readFileSync(checkpointPath, 'utf8').trim().split('\n')).toHaveLength(1);
  });

  test('CLI configuration resolves aliases, defaults, and explicitly supplied base settings', () => {
    const ts = tsConfigFromArgs(['--concurrency', '3', '--flush-every', '7', '--split', 'test', '--model', 'camel', '--m', 'alias']);
    expect(ts).toMatchObject({ concurrency: 3, flushEvery: 7, breakerThreshold: 6, split: 'test', model: 'alias', extraCheckpoints: undefined, csvPath: '/Users/sebastian/Documents/Python/triagerX/df_test_ts.csv' });
    const base = { ...TS_CONFIG, model: 'configured', concurrency: 11, breakerThreshold: undefined, extraCheckpoints: ['explicit.jsonl'] };
    expect(tsConfigFromArgs([], base)).toMatchObject({ model: 'configured', concurrency: 11, breakerThreshold: 22, extraCheckpoints: ['explicit.jsonl'] });
    expect(tsConfigFromArgs(['--split', 'test'], base).extraCheckpoints).toEqual(['explicit.jsonl']);
    expect(openj9ConfigFromArgs(['--split', 'test', '--csv-dir', '/tmp/data', '--concurrency', '4'])).toMatchObject({ csvPath: '/tmp/data/openj9_test_50.csv', breakerThreshold: 8 });
    expect(openj9ConfigFromArgs(['--model', 'camel'])).toMatchObject({ model: 'camel' });
    expect(tsConfigFromArgs([]).apiKey).toBeUndefined();
    expect(openj9ConfigFromArgs([]).apiKey).toBeUndefined();
  });

  test('dry-run returns a preview without API calls or filesystem writes for either corpus', async () => {
    for (const [run, base, rows] of [
      [runTs, TS_CONFIG, [{ issue_number: '1', issue_title: 'title', description: 'body', owner: 'dev' }]],
      [runOpenJ9, OPENJ9_CONFIG, [{ issue_number: '2', issue_title: 'title', issue_body: 'body', owner: 'dev' }]],
    ] as const) {
      const root = join(temp(), 'not-created');
      const result = await run({ ...base, dryRun: true, outputDirectory: root, checkpointPath: join(root, 'run.jsonl'), extraCheckpoints: [] }, {
        rows,
        fetch: (async () => { throw new Error('dry run must not fetch'); }) as typeof fetch,
        convertParquet: async () => { throw new Error('dry run must not convert'); },
      } as never);
      expect(result.summary.completed).toBe(0);
      expect(result.preview?.user).toContain('Initial report:');
      expect(existsSync(root)).toBe(false);
    }
  });

  test('loaders are used when run receives no rows, and OpenJ9 request uses real newlines', async () => {
    const dir = temp(); let loadedPath = ''; let requestBody: any;
    const tsDir = join(dir, 'ts');
    await runTs({ ...TS_CONFIG, apiKey: 'key', model: 'test-model', baseUrl: 'https://example.test/v1/', thinking: true, outputDirectory: tsDir, checkpointPath: join(tsDir, 'run.jsonl'), extraCheckpoints: [], concurrency: 1 }, {
      readRows: async (path) => { loadedPath = path; return [{ issue_number: '3', issue_title: 'ts', description: 'report', owner: 'dev' }]; },
      fetch: (async (url, init) => { expect(url).toBe('https://example.test/v1/chat/completions'); requestBody = JSON.parse(String(init?.body)); return response('{"fault_location":{"product":"TypeScript","component":"Checker","subsystem":"x"},"symptoms":[],"capabilities":[]}'); }) as typeof fetch,
      sleep: async () => {}, convertParquet: async () => {},
    });
    expect(loadedPath).toBe('/Users/sebastian/Documents/Python/triagerX/df_train_ts.csv');
    expect(requestBody.messages[1].content).toContain('Initial report:\nreport');
    expect(requestBody.model).toBe('test-model');
    expect(requestBody.thinking).toEqual({ type: 'enabled' });

    const openDir = join(dir, 'open');
    await runOpenJ9({ ...OPENJ9_CONFIG, apiKey: 'key', model: 'open-model', baseUrl: 'https://open.example/v1', outputDirectory: openDir, csvDirectory: '/tmp/csv', checkpointPath: join(openDir, 'run.jsonl'), extraCheckpoints: [], concurrency: 1 }, {
      readRows: async (path) => { loadedPath = path; return [{ issue_number: '4', issue_title: 'j9', issue_body: 'body', owner: 'dev' }]; },
      fetch: (async (url, init) => { expect(url).toBe('https://open.example/v1/chat/completions'); requestBody = JSON.parse(String(init?.body)); return response('{"fault_location":{"product":"OpenJ9","component":"JIT","subsystem":"x"},"symptoms":[],"capabilities":[]}'); }) as typeof fetch,
      sleep: async () => {}, convertParquet: async () => {},
    });
    expect(loadedPath).toBe('/tmp/csv/openj9_train_50.csv');
    expect(requestBody.messages[1].content).toContain('Initial report:\nbody');
    expect(requestBody.messages[1].content).not.toContain('report:\\n\\n');
    expect(requestBody.model).toBe('open-model');
    expect(requestBody.thinking).toEqual({ type: 'disabled' });
  });

  test('OpenJ9 keeps duplicate in-memory rows and fallback records resume without HTTP', async () => {
    const dir = temp(); const checkpointPath = join(dir, 'openj9.jsonl'); let calls = 0;
    const config = { ...OPENJ9_CONFIG, apiKey: 'test-secret', checkpointPath, outputDirectory: dir, extraCheckpoints: [], limit: 0, offset: 0, concurrency: 1, flushEvery: 1 };
    const row = { issue_number: '9', issue_title: 'nine', issue_body: 'body', owner: 'dev', component: 'comp:jit' };
    const first = await runOpenJ9(config, { rows: [row, row], fetch: (async () => { calls++; throw new Error('offline'); }) as typeof fetch,
      sleep: async () => {}, now: () => 10, random: () => 0, convertParquet: async () => {} });
    expect(first.summary).toMatchObject({ completed: 2, fallbacks: 2 });
    expect(first.records).toHaveLength(1);
    const second = await runOpenJ9(config, { rows: [row], fetch: (async () => { throw new Error('must skip'); }) as typeof fetch, convertParquet: async () => {} });
    expect(second.summary).toMatchObject({ completed: 0, skipped: 1 });
    expect(calls).toBe(10);
  });
});
