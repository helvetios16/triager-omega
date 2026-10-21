import { afterEach, describe, expect, test } from 'bun:test';
import { mkdirSync, mkdtempSync, readFileSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { flushOutputs, safeAppendRecord } from '../scripts/distill/deepseek_persistence';

const tempDirs: string[] = [];
function tempDir() {
  const dir = mkdtempSync(join(tmpdir(), 'deepseek-shared-persistence-'));
  tempDirs.push(dir);
  return dir;
}

describe('shared DeepSeek persistence locks', () => {
  test('serializes appends independently for different checkpoints', async () => {
    const dir = tempDir();
    const first = join(dir, 'first.jsonl');
    const second = join(dir, 'second.jsonl');
    await Promise.all([
      ...Array.from({ length: 12 }, (_, i) => safeAppendRecord(first, { issue_number: `a${i}` })),
      ...Array.from({ length: 12 }, (_, i) => safeAppendRecord(second, { issue_number: `b${i}` })),
    ]);
    expect(readFileSync(first, 'utf8').trim().split('\n')).toHaveLength(12);
    expect(readFileSync(second, 'utf8').trim().split('\n')).toHaveLength(12);
  });

  test('empty flush releases its path lock before a later append and flush', async () => {
    const dir = tempDir();
    const checkpoint = join(dir, 'empty.jsonl');
    let conversions = 0;
    const convert = async () => { conversions++; };
    await flushOutputs(checkpoint, dir, 'sample', 'model', (split) => split, convert);
    await safeAppendRecord(checkpoint, { issue_number: 'later', owner: 'dev', distilled_json: '{}', distilled_text: '', raw_text: '' });
    await flushOutputs(checkpoint, dir, 'sample', 'model', (split) => split, convert);
    expect(readFileSync(join(dir, 'sample_distilled.csv'), 'utf8')).toContain('later');
    expect(conversions).toBe(1);
  });

  test('append failure does not permanently poison the checkpoint lock', async () => {
    const dir = tempDir();
    const path = join(dir, 'blocked');
    mkdirSync(path);
    await expect(safeAppendRecord(path, { issue_number: 'bad' })).rejects.toThrow();
    rmSync(path, { recursive: true, force: true });
    await safeAppendRecord(path, { issue_number: 'good' });
    expect(readFileSync(path, 'utf8')).toBe('{"issue_number":"good"}\n');
  });
});

afterEach(() => {
  for (const dir of tempDirs) rmSync(dir, { recursive: true, force: true });
});
