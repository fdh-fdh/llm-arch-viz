// v2 unit tests: knowledge completeness, formula renderer, inspector, IR semantics.
// node tests/v2.test.mjs
import { buildGraph, kvCacheBytesPerToken, flopsPerToken } from '../src/parser/ir.js';
import { KNOWLEDGE, getKnowledge } from '../src/knowledge.js';
import { fm, formulaHTML } from '../src/formula.js';
import { renderInspector } from '../src/inspector.js';
import { LayoutBuilder } from '../src/layout.js';
import { SAMPLES } from '../src/samples.js';

let failures = 0;
function check(label, cond, extra = '') {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${label}${extra ? '  (' + extra + ')' : ''}`);
  if (!cond) failures++;
}

// 1. knowledge base completeness
for (const [key, k] of Object.entries(KNOWLEDGE)) {
  check(`knowledge ${key} complete`, !!(k.title && k.formula && k.intuition && k.details !== undefined && Array.isArray(k.refs)));
}
check('knowledge count >= 20', Object.keys(KNOWLEDGE).length >= 20, String(Object.keys(KNOWLEDGE).length));

// 2. formula renderer
{
  const ln = getKnowledge('layernorm');
  const html = formulaHTML(ln.formula);
  check('layernorm formula renders frac', html.includes('fm-frac'));
  check('layernorm formula renders sqrt', html.includes('fm-sqrt'));
  check('formula escapes html', fm('<x>').includes('&lt;'));
  // every formula in the KB renders without throwing and non-empty
  let allOk = true;
  for (const [key, k] of Object.entries(KNOWLEDGE)) {
    try {
      const h = formulaHTML(k.formula);
      if (!h || h.length < 15) { allOk = false; console.log('  empty formula:', key); }
    } catch (e) { allOk = false; console.log('  formula throws:', key, e.message); }
  }
  check('all KB formulas render', allOk);
}

// 3. IR semantics: kb keys attached and resolvable
const byId = Object.fromEntries(SAMPLES.map((s) => [s.id, s]));
{
  const g = buildGraph(byId['qwen3-30b-a3b'].config, byId['qwen3-30b-a3b'].source);
  const attn = g.stacks[0].segments.find((s) => s.kind === 'attn');
  check('attn kb has gqa', attn.kb.includes('gqa'), attn.kb.join(','));
  check('attn kb has qk_norm (qwen3)', attn.kb.includes('qk_norm'));
  const moe = g.stacks[0].segments.find((s) => s.kind === 'moe');
  check('moe kb', moe.kb.includes('moe_router'));
  for (const seg of g.stacks[0].segments) {
    for (const key of seg.kb || []) {
      if (!getKnowledge(key)) { check(`kb key resolvable: ${key}`, false); }
    }
  }
  check('kv cache per token sane (qwen3-30b: 48*2*4*128*2B)', kvCacheBytesPerToken(g) === 48 * 2 * 4 * 128 * 2, String(kvCacheBytesPerToken(g)));
  check('flops per token > 0', flopsPerToken(g) > 1e9);
}
{
  const g = buildGraph(byId['deepseek-v3'].config, byId['deepseek-v3'].source);
  const attn = g.stacks[1].segments.find((s) => s.kind === 'attn');
  check('deepseek attn kb has mla', attn.kb.includes('mla'));
  // MLA cache: (kv_lora 512 + rope 64) * 61 layers * 2B
  check('deepseek MLA kv cache', kvCacheBytesPerToken(g) === (512 + 64) * 61 * 2, String(kvCacheBytesPerToken(g)));
}

// 4. layout carries elem/subdiv/kb metadata
{
  const g = buildGraph(byId['qwen3-30b-a3b'].config, byId['qwen3-30b-a3b'].source);
  const b = new LayoutBuilder();
  const lay = b.build(g, { expanded: new Set(['0:0']), expandedExperts: new Set() });
  const items = lay.items.slice(0, lay.soa.count).filter(Boolean);
  const q = items.find((it) => it.role === 'q_proj');
  check('q_proj item exists', !!q);
  check('q_proj elem dims', q.elem.rows === 4096 && q.elem.cols === 2048, JSON.stringify(q.elem && [q.elem.rows, q.elem.cols]));
  check('q_proj subdiv 32 heads / group 8', q.subdiv && q.subdiv[0] === 32 && q.subdiv[1] === 8, JSON.stringify(q.subdiv));
  check('q_proj kb', q.kb.includes('attention_core'));
  const kproj = items.find((it) => it.role === 'k_proj');
  check('k_proj subdiv 4 kv heads', kproj.subdiv && kproj.subdiv[0] === 4, JSON.stringify(kproj.subdiv));
  // grid buffer written & clamped
  const gi = q.idx * 2;
  check('grid buffer clamped to 512', lay.soa.grid[gi] === 512 && lay.soa.grid[gi + 1] === 512, `${lay.soa.grid[gi]},${lay.soa.grid[gi + 1]}`);
  const emb = items.find((it) => it.path && it.path[0] === 'embed_tokens');
  check('embedding elem real dims', emb.elem.rows === 151936, String(emb.elem.rows));

  // 5. inspector renders for various selections (pure string, no DOM needed)
  const overview = renderInspector(null, g);
  check('inspector overview mentions MoE', overview.includes('MoE') && overview.includes('crumb'));
  const insQ = renderInspector(q, g);
  check('inspector q_proj has breadcrumb + formula', insQ.includes('Layer 0') && insQ.includes('fm-frac'));
  check('inspector q_proj has GQA diagram', insQ.includes('<svg'));
  check('inspector q_proj has KV widget', insQ.includes('data-kv'));
  const insEl = renderInspector({ element: { item: q, row: 812, col: 96 } }, g);
  check('inspector element view', insEl.includes('W[812, 96]'));
  const agg = items.find((it) => it.type === 'stackAgg');
  const insAgg = renderInspector(agg, g);
  check('inspector stackAgg lists pipeline', insAgg.includes('层内管线') && insAgg.includes('data-nav="seg:'));
}

// 6. v2.2: γ/β strips, residual rails, MLP neuron bands, LOD metadata
{
  const gGpt = buildGraph(byId['gpt2'].config, byId['gpt2'].source);
  const b2 = new LayoutBuilder();
  const st = { expanded: new Set(['0:0']), expandedExperts: new Set() };
  const layG = b2.build(gGpt, st);
  const itemsG = layG.items.slice(0, layG.soa.count).filter(Boolean);
  const gammas = itemsG.filter((it) => it.role === 'γ');
  const betas = itemsG.filter((it) => it.role === 'β');
  check('gpt2 layer has γ strips', gammas.length >= 2, String(gammas.length));
  check('gpt2 layer has β strips (LayerNorm bias)', betas.length >= 2, String(betas.length));

  const gLl = buildGraph(byId['llama-3.1-8b'].config, byId['llama-3.1-8b'].source);
  const layL = b2.build(gLl, st);
  const itemsL = layL.items.slice(0, layL.soa.count).filter(Boolean);
  check('llama RMSNorm has γ only', itemsL.some((it) => it.role === 'γ') && !itemsL.some((it) => it.role === 'β'));
  const up = itemsL.find((it) => it.role === 'up_proj');
  check('llama mlp neuron bands [8,1]', up && up.subdiv && up.subdiv[0] === 8, JSON.stringify(up && up.subdiv));

  // residual rails: expanded layer adds flag=1 rail boxes (3 per ⊕, 2 ⊕ per layer)
  const collapsed = b2.build(gLl, { expanded: new Set(), expandedExperts: new Set() });
  const nConnCollapsed = Array.from({length: collapsed.soa.count}, (_, i) => collapsed.soa.flag[i]).filter((f) => f === 1).length;
  const expanded = b2.build(gLl, st);
  const nConnExpanded = Array.from({length: expanded.soa.count}, (_, i) => expanded.soa.flag[i]).filter((f) => f === 1).length;
  check('residual rails present (>=6 extra rail segments)', nConnExpanded >= nConnCollapsed + 6, `${nConnCollapsed} -> ${nConnExpanded}`);
}

console.log(failures === 0 ? '\nALL TESTS PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
