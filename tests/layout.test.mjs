// Layout + llmarch round-trip tests (Node): node tests/layout.test.mjs
import { buildGraph } from '../src/parser/ir.js';
import { LayoutBuilder } from '../src/layout.js';
import { exportLlmarch, parseLlmarch } from '../src/llmarch.js';
import { buildSVG } from '../src/viz2d.js';
import { SAMPLES } from '../src/samples.js';

let failures = 0;
function check(label, cond, extra = '') {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${label}${extra ? '  (' + extra + ')' : ''}`);
  if (!cond) failures++;
}

const builder = new LayoutBuilder();

for (const s of SAMPLES) {
  const g = buildGraph(s.config, s.source);

  // collapsed default
  let st = { expanded: new Set(), expandedExperts: new Set() };
  let lay = builder.build(g, st);
  check(`${s.id} collapsed layout instances < 200`, lay.count === undefined ? lay.soa.count < 200 : lay.count < 200, `${lay.soa.count} instances`);
  check(`${s.id} bounds finite`, Number.isFinite(lay.bounds.min[1]) && Number.isFinite(lay.bounds.max[1]));

  // first layer expanded + experts expanded
  st = { expanded: new Set(['0:0']), expandedExperts: new Set(['0:0']) };
  lay = builder.build(g, st);
  check(`${s.id} expanded layout ok`, lay.soa.count > 10 && lay.soa.count < 5000, `${lay.soa.count} instances`);
  check(`${s.id} expandedRanges tracked`, (builder.expandedRanges || []).length === 1);
  // every item index < count, and no NaN in SoA
  let nan = false;
  for (let i = 0; i < lay.soa.count * 3; i++) {
    if (!Number.isFinite(lay.soa.pos[i]) || !Number.isFinite(lay.soa.scale[i])) { nan = true; break; }
  }
  check(`${s.id} SoA finite`, !nan);

  // full expand (worst case instance count)
  const full = { expanded: new Set(), expandedExperts: new Set() };
  g.stacks.forEach((stk, si) => { for (let li = 0; li < stk.count; li++) { full.expanded.add(si + ':' + li); full.expandedExperts.add(si + ':' + li); } });
  lay = builder.build(g, full);
  check(`${s.id} full-expand instances sane`, lay.soa.count > 0 && lay.soa.count < 120000, `${lay.soa.count} instances`);

  // SVG builds
  const svg = buildSVG(g);
  check(`${s.id} SVG generated`, svg.startsWith('<svg') && svg.length > 2000, `${(svg.length / 1024).toFixed(0)} KB`);
}

// llmarch round trip
{
  const s = SAMPLES[0];
  // Node lacks DOM; exportLlmarch uses only JSON + Date — fine.
  const text = exportLlmarch({ config: s.config, source: s.source, name: 'test', view: { expanded: ['0:0'] } });
  const doc = parseLlmarch(text);
  check('llmarch round-trip config intact', JSON.stringify(doc.config) === JSON.stringify(s.config));
  check('llmarch view preserved', doc.view.expanded[0] === '0:0');
  // bare config.json channel
  const bare = parseLlmarch(JSON.stringify(s.config));
  check('bare config.json accepted', bare.config.model_type === 'qwen3_moe');
  // garbage rejected
  let threw = false;
  try { parseLlmarch('{"hello": 1}'); } catch { threw = true; }
  check('non-config JSON rejected', threw);
}

console.log(failures === 0 ? '\nALL TESTS PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
