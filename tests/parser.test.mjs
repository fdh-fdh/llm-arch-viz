// Node unit tests for the parser (no browser needed): node tests/parser.test.mjs
import { buildGraph, fmtParams } from '../src/parser/ir.js';
import { SAMPLES } from '../src/samples.js';

let failures = 0;
function check(label, actual, expected, tolerance = 0) {
  const ok = tolerance ? Math.abs(actual - expected) <= tolerance : actual === expected;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}: ${actual}${ok ? '' : ` (expected ${expected})`}`);
  if (!ok) failures++;
}
function approx(label, actual, expected, relTol = 0.02) {
  const ok = Math.abs(actual - expected) / expected <= relTol;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}: ${fmtParams(actual)}${ok ? '' : ` (expected ~${fmtParams(expected)})`}`);
  if (!ok) failures++;
}

const byId = Object.fromEntries(SAMPLES.map((s) => [s.id, s]));

// 1. Qwen3-MoE 2-layer — exact numbers hand-verified earlier in the session.
{
  const g = buildGraph(byId['qwen3-moe-2layer'].config, byId['qwen3-moe-2layer'].source);
  check('qwen3-moe-2layer total', g.meta.params.total, 1868573184);
  check('qwen3-moe-2layer active', g.meta.params.active, 736111104);
  check('qwen3-moe-2layer embedding', g.meta.params.embedding, 311164928);
  check('qwen3-moe-2layer tier', g.meta.tier, 'T2'); // 128 experts >= 64
  check('qwen3-moe-2layer stacks', g.stacks.length, 1);
  check('qwen3-moe-2layer layer count', g.stacks[0].count, 2);
}

// 2. Qwen3-30B-A3B — official: 30.5B total, 3.3B active.
{
  const g = buildGraph(byId['qwen3-30b-a3b'].config, byId['qwen3-30b-a3b'].source);
  approx('qwen3-30b total ~30.5B', g.meta.params.total, 30.5e9, 0.03);
  approx('qwen3-30b active ~3.3B', g.meta.params.active, 3.3e9, 0.05);
}

// 3. Llama-3.1-8B — official 8.03B.
{
  const g = buildGraph(byId['llama-3.1-8b'].config, byId['llama-3.1-8b'].source);
  approx('llama-3.1-8b total ~8.03B', g.meta.params.total, 8.03e9, 0.01);
  check('llama tier', g.meta.tier, 'T1');
  const attn = g.stacks[0].segments.find((s) => s.kind === 'attn');
  check('llama GQA variant', attn.meta.variant, 'GQA');
}

// 4. GPT-2 — 124M (tied embedding counted once) + wpe.
{
  const g = buildGraph(byId['gpt2'].config, byId['gpt2'].source);
  approx('gpt2 total ~124M', g.meta.params.total, 124.4e6, 0.01);
  check('gpt2 tied', g.meta.params.tied, true);
  check('gpt2 tier', g.meta.tier, 'T0');
}

// 5. Mixtral-8x7B — official 46.7B total, 12.9B active.
{
  const g = buildGraph(byId['mixtral-8x7b'].config, byId['mixtral-8x7b'].source);
  approx('mixtral total ~46.7B', g.meta.params.total, 46.7e9, 0.01);
  approx('mixtral active ~12.9B', g.meta.params.active, 12.9e9, 0.01);
  check('mixtral tier', g.meta.tier, 'T2');
}

// 6. DeepSeek-V3 — official 671B total, 37B active.
{
  const g = buildGraph(byId['deepseek-v3'].config, byId['deepseek-v3'].source);
  approx('deepseek-v3 total ~671B', g.meta.params.total, 671e9, 0.02);
  approx('deepseek-v3 active ~37B', g.meta.params.active, 37e9, 0.03);
  check('deepseek-v3 tier', g.meta.tier, 'T3');
  check('deepseek-v3 two stacks (dense + moe)', g.stacks.length, 2);
  check('deepseek-v3 dense count', g.stacks[0].count, 3);
  check('deepseek-v3 moe count', g.stacks[1].count, 58);
  const attn = g.stacks[1].segments.find((s) => s.kind === 'attn');
  check('deepseek-v3 MLA', attn.meta.variant, 'MLA');
}

// 7. Generic fallback: unknown model_type with llama-like fields.
{
  const cfg = { ...byId['llama-3.1-8b'].config, model_type: 'totally_new_arch_2027' };
  const g = buildGraph(cfg, { kind: 'manual' });
  check('fallback degraded', g.meta.degraded, true);
  approx('fallback total still ~8B', g.meta.params.total, 8.03e9, 0.01);
}

console.log(failures === 0 ? '\nALL TESTS PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
