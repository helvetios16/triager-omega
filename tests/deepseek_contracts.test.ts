import { afterEach, describe, expect, test } from 'bun:test';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { CONFIG as TS_CONFIG, run as runTs } from '../scripts/distill/distill_deepseek_ts';
import { CONFIG as J9_CONFIG, run as runJ9 } from '../scripts/distill/distill_deepseek_openj9';

const directories: string[] = [];
function temporaryDirectory() {
  const directory = mkdtempSync(join(tmpdir(), 'deepseek-contract-'));
  directories.push(directory);
  return directory;
}
afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

function row(id: string) {
  return { issue_number: id, issue_title: `title-${id}`, description: 'report', issue_body: 'report', owner: 'developer' };
}
const valid = JSON.stringify({
  fault_location: { product: 'Product', component: 'Component', subsystem: 'Subsystem' },
  symptoms: ['symptom'], capabilities: ['capability'],
});
function response(content = valid) {
  return Response.json({
    choices: [{ message: { content } }],
    usage: { prompt_tokens: 10, completion_tokens: 4, prompt_cache_hit_tokens: 3, prompt_cache_miss_tokens: 0 },
  });
}
function checkpoint(id: string, owner = 'developer') {
  return {
    issue_number: id, owner, distilled_json: valid, distilled_text: '[FL] Product Component',
    raw_text: `Summary: title-${id}`, fallback: true, latency_ms: 0,
  };
}
const offline = { sleep: async (_ms: number) => {}, now: () => 100, random: () => 0, convertParquet: async () => {} };
const corpora = [
  { name: 'TypeScript', run: runTs, base: TS_CONFIG, prefix: 'ts_train_deepseek', deduplicates: true },
  { name: 'OpenJ9', run: runJ9, base: J9_CONFIG, prefix: 'openj9_train', deduplicates: false },
] as const;

for (const corpus of corpora) {
  describe(`${corpus.name} execution contract`, () => {
    test('success, own last-record precedence, extra IDs, resume and fresh accounting', async () => {
      const directory = temporaryDirectory();
      const path = join(directory, 'own.jsonl');
      const extra = join(directory, 'extra.jsonl');
      writeFileSync(path, [checkpoint('own', 'first'), checkpoint('own', 'last')].map((record) => JSON.stringify(record)).join('\n') + '\n');
      writeFileSync(extra, JSON.stringify(checkpoint('external')) + '\n');
      const config = {
        ...corpus.base, apiKey: 'offline', checkpointPath: path, outputDirectory: directory,
        concurrency: 1, breakerThreshold: 10, extraCheckpoints: [extra],
      };
      let calls = 0;
      const rows = [row('own'), row('external'), row('new')];
      const first = await corpus.run(config, {
        ...offline, rows, fetch: (async () => { calls++; return response(); }) as typeof fetch,
      });
      expect(calls).toBe(1);
      expect(first.summary).toMatchObject({ completed: 1, skipped: 2, fallbacks: 0 });
      expect(first.records.map((record) => record.issue_number)).toEqual(['own', 'new']);
      expect(first.records[0].owner).toBe('last');
      expect(first.records[1].fallback).toBe(false);
      expect(first.records[1].prompt_tokens).toBe(10);
      expect(first.statistics.costUsd).toBeCloseTo((3 * .006 + 7 * .30 + 4 * 1.20) / 1_000_000, 12);
      const csv = readFileSync(join(directory, `${corpus.prefix}_distilled.csv`), 'utf8');
      expect(csv).not.toContain('external,');
      const second = await corpus.run(config, {
        ...offline, rows,
        fetch: (async () => { throw new Error('resume must not request HTTP'); }) as typeof fetch,
      });
      expect(second.summary).toMatchObject({ completed: 0, skipped: 3, aborted: false });
      expect(second.statistics).toEqual({ promptTokens: 0, completionTokens: 0, cacheHitTokens: 0, cacheMissTokens: 0, costUsd: 0 });
    });

    test('input duplicate policy is preserved without relying on checkpoint collapse', async () => {
      const directory = temporaryDirectory();
      let calls = 0;
      const result = await corpus.run({
        ...corpus.base, apiKey: 'offline', outputDirectory: directory,
        extraCheckpoints: [], concurrency: 1, breakerThreshold: 10,
      }, {
        ...offline, rows: [row('duplicate'), row('duplicate')],
        fetch: (async () => { calls++; return response(); }) as typeof fetch,
      });
      expect(calls).toBe(corpus.deduplicates ? 1 : 2);
      expect(result.summary.completed).toBe(calls);
      expect(result.records).toHaveLength(1);
    });

    test('invalid JSON counts successful usage while checkpoint fallback token fields are zero', async () => {
      const directory = temporaryDirectory();
      let calls = 0;
      const result = await corpus.run({
        ...corpus.base, apiKey: 'offline', outputDirectory: directory,
        extraCheckpoints: [], concurrency: 1, breakerThreshold: 10,
      }, {
        ...offline, rows: [row('invalid')],
        fetch: (async () => { calls++; return response('not-json'); }) as typeof fetch,
      });
      expect(calls).toBe(1);
      expect(result.summary.fallbacks).toBe(1);
      expect(result.statistics.promptTokens).toBe(10);
      expect(result.statistics.costUsd).toBeCloseTo((3 * .006 + 7 * .30 + 4 * 1.20) / 1_000_000, 12);
      expect(result.records[0].prompt_tokens).toBe(0);
      expect(result.records[0].fallback).toBe(true);
    });

    test('five failed requests trip breaker, and a subsequent run has independent state', async () => {
      const directory = temporaryDirectory();
      const config = {
        ...corpus.base, apiKey: 'offline', outputDirectory: directory,
        extraCheckpoints: [], concurrency: 1, breakerThreshold: 1,
      };
      let calls = 0;
      const sleeps: number[] = [];
      const first = await corpus.run(config, {
        ...offline, rows: [row('failed'), row('later')],
        sleep: async (ms) => { sleeps.push(ms); },
        fetch: (async () => { calls++; return new Response('unavailable', { status: 503 }); }) as typeof fetch,
      });
      expect(calls).toBe(5);
      expect(sleeps).toEqual([2000, 4000, 8000, 10000]);
      expect(first.summary).toMatchObject({ completed: 1, fallbacks: 1, aborted: true });
      expect(first.statistics.costUsd).toBe(0);
      const second = await corpus.run(config, {
        ...offline, rows: [row('failed'), row('later')],
        fetch: (async () => { calls++; return response(); }) as typeof fetch,
      });
      expect(calls).toBe(6);
      expect(second.summary).toMatchObject({ completed: 1, skipped: 1, fallbacks: 0, aborted: false });
      expect(second.statistics.promptTokens).toBe(10);
    });

    test('dry-run previews the first pending row, with canonical paths and no writes', async () => {
      const directory = temporaryDirectory();
      const own = join(directory, 'own.jsonl');
      const extra = join(directory, 'external.jsonl');
      writeFileSync(own, JSON.stringify(checkpoint('first')) + '\n');
      writeFileSync(extra, JSON.stringify(checkpoint('second')) + '\n');
      const before = readFileSync(own, 'utf8');
      const outputDirectory = join(directory, 'not-created');
      const result = await corpus.run({
        ...corpus.base, dryRun: true, checkpointPath: own,
        outputDirectory, extraCheckpoints: [extra],
      }, { rows: [row('first'), row('second'), row('third')] });
      expect(result.preview?.user).toContain('title-third');
      expect(result.summary.skipped).toBe(2);
      expect(result.paths.split).toBe(corpus.name === 'TypeScript' ? 'ts_train_deepseek' : 'train');
      expect(readFileSync(own, 'utf8')).toBe(before);
      expect(existsSync(outputDirectory)).toBe(false);
    });

    test('invalid runtime limits fail before loading or writing', async () => {
      const outputDirectory = join(temporaryDirectory(), 'not-created');
      for (const overrides of [{ concurrency: 0 }, { flushEvery: 0 }, { breakerThreshold: 0 }]) {
        await expect(corpus.run({ ...corpus.base, outputDirectory, ...overrides }, {
          readRows: async () => { throw new Error('loader must not run'); },
        })).rejects.toThrow('must be an integer > 0');
      }
      expect(existsSync(outputDirectory)).toBe(false);
    });
  });
}

test('TypeScript slices before deduplicating and reversing the selected range', async () => {
  const directory = temporaryDirectory();
  const seen: string[] = [];
  const rows = ['outside', 'a', 'a', 'b', 'after'].map(row);
  const result = await runTs({
    ...TS_CONFIG, apiKey: 'offline', outputDirectory: directory,
    offset: 1, limit: 3, reverse: true, extraCheckpoints: [], concurrency: 1,
  }, {
    ...offline, rows,
    fetch: (async (_url, init) => {
      const request = JSON.parse(String(init?.body));
      seen.push(request.messages[1].content.match(/Summary: title-(\w+)/)[1]);
      return response();
    }) as typeof fetch,
  });
  expect(seen).toEqual(['b', 'a']);
  expect(result.summary.completed).toBe(2);
});
