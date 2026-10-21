import { afterEach, beforeAll, describe, expect, test } from 'bun:test';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
let ts: typeof import('../scripts/distill/distill_deepseek_ts');
let openj9: typeof import('../scripts/distill/distill_deepseek_openj9');
beforeAll(async () => {
  [ts, openj9] = await Promise.all([
    import('../scripts/distill/distill_deepseek_ts'),
    import('../scripts/distill/distill_deepseek_openj9'),
  ]);
});

const tempDirs: string[] = [];
afterEach(() => {
  for (const dir of tempDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

function tempDir() {
  const dir = mkdtempSync(join(tmpdir(), 'deepseek-characterization-'));
  tempDirs.push(dir);
  return dir;
}

const record = (issue_number: string, overrides: Record<string, unknown> = {}) => ({
  issue_number,
  owner: 'owner',
  distilled_json: JSON.stringify({ fault_location: { component: 'Compiler', subsystem: 'inference' }, symptoms: ['bad, "thing"'], capabilities: ['types'] }),
  distilled_text: '[FL] TypeScript Compiler inference [SY] bad [CP] types',
  raw_text: 'Summary: line one, "quoted"\nline two',
  fallback: false,
  latency_ms: 12,
  model: 'offline',
  ...overrides,
});

for (const name of ['TypeScript', 'OpenJ9'] as const) {
  const getRunner = () => name === 'TypeScript' ? ts : openj9;
  describe(`${name} runner persistence`, () => {
    test('imports without starting main and exposes corpus-specific input and fallback builders', () => {
      if (name === 'TypeScript') {
        expect(ts.buildInput({ issue_number: '1', issue_title: 'Title', description: 'Body', component: '' })).toContain('(infer canonical compiler component)');
        expect(ts.fallback({ issue_number: '1', issue_title: 'Title', description: '' })._fallback).toBe(true);
      } else {
        expect(openj9.buildInput({ issue_number: '1', issue_title: 'Title', issue_body: 'Body', owner: '', component: 'comp:jit' })).toContain('Component: JIT');
        expect(openj9.normalizeOpenJ9Component('comp:jclextensions')).toBe('JCL Extensions');
        expect(openj9.fallback({ issue_number: '1', issue_title: 'Title', issue_body: '', owner: '' })._fallback).toBe(true);
      }
      expect(getRunner().calculateCostUsd(1_000_000, 1_000_000, 1_000_000)).toBeCloseTo(1.506);
    });

    test('checkpoint is streaming-compatible, ignores invalid lines and uses last valid line per ID', async () => {
      const dir = tempDir();
      const path = join(dir, 'checkpoint.jsonl');
      writeFileSync(path, `${JSON.stringify(record('7', { owner: 'first' }))}\nnot-json\n\n${JSON.stringify(record('7', { issue_number: 7, owner: 'last' }))}\n${JSON.stringify({ owner: 'missing id' })}\n`);
      expect(getRunner().loadExistingCheckpoint(path).get('7')?.owner).toBe('last');
      await Promise.all(['8', '9', '11', '12'].map((id) => getRunner().safeAppendRecord(path, record(id))));
      const lines = readFileSync(path, 'utf8').trim().split('\n');
      expect(lines.slice(-4).map((line) => JSON.parse(line).issue_number).sort()).toEqual(['11', '12', '8', '9']);
    });

    test('flush writes exact CSV views and provenance headers; conversion is injectable and failures stay silent', async () => {
      const dir = tempDir();
      const checkpoint = join(dir, 'checkpoint.jsonl');
      writeFileSync(checkpoint, `${JSON.stringify(record('10'))}\n${JSON.stringify(record('10', { owner: 'winner' }))}\n{bad json}\n`);
      const calls: Array<{ command: string; args: string[] }> = [];
      await getRunner().flushOutputs(checkpoint, dir, 'sample', async (command, args) => { calls.push({ command, args }); });
      const prefix = name === 'OpenJ9' ? 'openj9_sample' : 'sample';
      const csv = (suffix: string) => readFileSync(join(dir, `${prefix}_${suffix}.csv`), 'utf8');
      expect(csv('distilled').split('\n')[0]).toBe('issue_number,owner,text');
      expect(csv('distilled')).toContain('10,winner,');
      expect(csv('raw')).toContain('"Summary: line one, ""quoted""\nline two"');
      expect(csv('both')).toContain('"Summary: line one, ""quoted""\nline two [FL]');
      expect(csv('provenance').split('\n')[0]).toBe('issue_number,issue_title,owner,model,latency_ms,fallback,component,subsystem,symptoms,capabilities,distilled_text');
      expect(csv('provenance')).toContain('10,"line one, ""quoted""",winner,offline,12,false,Compiler,inference,"bad, ""thing""",types,');
      expect(calls).toHaveLength(1);
      expect(calls[0].command).toBe('uv');
      expect(calls[0].args.slice(0, 3)).toEqual(['run', 'python', '-c']);
      expect(calls[0].args[3]).toContain('to_parquet');

      await getRunner().flushOutputs(checkpoint, dir, 'sample', async () => { throw new Error('conversion unavailable'); });
      expect(csv('raw')).toContain('"Summary: line one, ""quoted""\nline two"');
    });
  });
}
