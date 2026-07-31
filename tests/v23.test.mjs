// v2.3 unit tests: tour engine, safetensors inference, 2D nav attributes.
// node tests/v23.test.mjs
import { buildGraph } from '../src/parser/ir.js';
import { LayoutBuilder } from '../src/layout.js';
import { buildTour, resolveStation } from '../src/tour.js';
import { parseSafetensorsHeader, inferConfigFromTensors } from '../src/safetensors.js';
import { buildSVG } from '../src/viz2d.js';
import { SAMPLES } from '../src/samples.js';

let failures = 0;
function check(label, cond, extra = '') {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${label}${extra ? '  (' + extra + ')' : ''}`);
  if (!cond) failures++;
}
const byId = Object.fromEntries(SAMPLES.map((s) => [s.id, s]));
const builder = new LayoutBuilder();

// ---- 1. tour engine: every station resolvable on every sample ----
for (const s of SAMPLES) {
  const g = buildGraph(s.config, s.source);
  const stations = buildTour(g);
  check(`${s.id} tour has >= 10 stations`, stations.length >= 10, String(stations.length));
  // simulate the controller: expand what each station requires, then resolve
  const st = { expanded: new Set(), expandedExperts: new Set() };
  let unresolved = 0;
  for (const station of stations) {
    if (station.expand) st.expanded.add(station.expand.si + ':' + station.expand.li);
    const lay = builder.build(g, st);
    if (!resolveStation(station, lay)) {
      unresolved++;
      console.log(`  UNRESOLVED: ${s.id} station "${station.label}"`);
    }
  }
  check(`${s.id} all stations resolve`, unresolved === 0, `${stations.length - unresolved}/${stations.length}`);
}
{
  const g = buildGraph(byId['qwen3-30b-a3b'].config, byId['qwen3-30b-a3b'].source);
  const labels = buildTour(g).map((s) => s.label).join('|');
  check('qwen moe tour covers router+experts', labels.includes('Router') && labels.includes('专家阵列'));
  const gd = buildGraph(byId['deepseek-v3'].config, byId['deepseek-v3'].source);
  const labelsD = buildTour(gd).map((s) => s.label).join('|');
  check('deepseek tour covers dense AND moe stacks', labelsD.includes('进入') && labelsD.includes('Router'));
}

// ---- 2. safetensors header parse + config inference ----
function mkHeader(tensorsObj) {
  const json = Buffer.from(JSON.stringify(tensorsObj));
  const buf = Buffer.alloc(8 + json.length);
  buf.writeBigUInt64LE(BigInt(json.length), 0);
  json.copy(buf, 8);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length);
}
{
  const T = (shape) => ({ dtype: 'BF16', shape, data_offsets: [0, 0] });
  const tensorsObj = {
    '__metadata__': { format: 'pt' },
    'model.embed_tokens.weight': T([1000, 256]),
    'model.layers.0.self_attn.q_proj.weight': T([512, 256]),
    'model.layers.0.self_attn.k_proj.weight': T([256, 256]),
    'model.layers.0.self_attn.v_proj.weight': T([256, 256]),
    'model.layers.0.self_attn.o_proj.weight': T([256, 512]),
    'model.layers.0.mlp.gate_proj.weight': T([1024, 256]),
    'model.layers.0.mlp.up_proj.weight': T([1024, 256]),
    'model.layers.0.mlp.down_proj.weight': T([256, 1024]),
    'model.layers.0.input_layernorm.weight': T([256]),
    'model.layers.1.self_attn.q_proj.weight': T([512, 256]),
    'model.norm.weight': T([256]),
  };
  const tensors = parseSafetensorsHeader(mkHeader(tensorsObj));
  check('safetensors parse count (skips __metadata__)', tensors.length === 11, String(tensors.length));
  const { config, notes } = inferConfigFromTensors(tensors, 'model.safetensors');
  check('infer hidden 256', config.hidden_size === 256);
  check('infer layers 2', config.num_hidden_layers === 2);
  check('infer heads 4 / kv 2 (head_dim 128)', config.num_attention_heads === 4 && config.num_key_value_heads === 2, `${config.num_attention_heads}/${config.num_key_value_heads} hd=${config.head_dim}`);
  check('infer tied (no lm_head)', config.tie_word_embeddings === true);
  check('infer intermediate 1024', config.intermediate_size === 1024);
  check('notes present', notes.length >= 2, notes[0]);
  const g = buildGraph(config, { kind: 'manual', name: 'inferred' });
  check('inferred config builds graph', g.meta.params.total > 0, String(g.meta.params.total));
}
{
  // gpt2 naming (Conv1D [in, out])
  const T = (shape) => ({ dtype: 'F32', shape, data_offsets: [0, 0] });
  const obj = {
    'wte.weight': T([50257, 768]), 'wpe.weight': T([1024, 768]),
    'h.0.attn.c_attn.weight': T([768, 2304]), 'h.0.mlp.c_fc.weight': T([768, 3072]),
    'h.11.mlp.c_proj.weight': T([3072, 768]),
  };
  const { config } = inferConfigFromTensors(parseSafetensorsHeader(mkHeader(obj)), 'gpt2.safetensors');
  check('gpt2 naming: model_type', config.model_type === 'gpt2');
  check('gpt2 naming: 12 layers / n_inner 3072', config.n_layer === 12 && config.n_inner === 3072, `${config.n_layer}/${config.n_inner}`);
  const g = buildGraph(config, { kind: 'manual' });
  check('gpt2 inferred graph ~124M', Math.abs(g.meta.params.total - 124.4e6) / 124.4e6 < 0.02, String(g.meta.params.total));
}
{
  // unknown naming rejected
  const T = (shape) => ({ dtype: 'F32', shape, data_offsets: [0, 0] });
  let threw = false;
  try {
    inferConfigFromTensors(parseSafetensorsHeader(mkHeader({ 'foo.bar': T([2, 2]) })));
  } catch { threw = true; }
  check('unknown naming rejected with clear error', threw);
}

// ---- 3. 2D SVG carries nav attributes (FR-G2) ----
{
  const g = buildGraph(byId['qwen3-30b-a3b'].config, byId['qwen3-30b-a3b'].source);
  const svg = buildSVG(g);
  check('2D svg has seg nav', svg.includes('data-nav="seg:0:0:attn:'), '');
  check('2D svg has path nav', svg.includes('data-nav="path:embed_tokens"') && svg.includes('data-nav="path:lm_head"'));
  check('2D svg nav class', svg.includes('class="v2d-nav"'));
}

console.log(failures === 0 ? '\nALL TESTS PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
