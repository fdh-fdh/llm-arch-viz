// Local safetensors support (FR-A4): read ONLY the header via File.slice —
// weights never leave the user's machine and never enter memory.
// Also: heuristic config inference from tensor names/shapes (channel B of the
// dual-channel design), covering llama/qwen/qwen-moe/mixtral/gpt2 naming.

export function parseSafetensorsHeader(buf) {
  const dv = new DataView(buf);
  const len = Number(dv.getBigUint64(0, true));
  if (len <= 0 || len > buf.byteLength - 8) throw new Error('safetensors 头长度非法');
  const json = new TextDecoder().decode(new Uint8Array(buf, 8, len));
  const doc = JSON.parse(json);
  const tensors = [];
  for (const [name, v] of Object.entries(doc)) {
    if (name === '__metadata__') continue;
    tensors.push({ name, shape: v.shape, dtype: v.dtype });
  }
  return tensors;
}

export async function readSafetensorsFile(file) {
  const head8 = await file.slice(0, 8).arrayBuffer();
  const len = Number(new DataView(head8).getBigUint64(0, true));
  if (len <= 0 || len > 100e6) throw new Error(`safetensors 头长度异常(${len})——不是合法的 safetensors 文件?`);
  const buf = await file.slice(0, 8 + len).arrayBuffer();
  return parseSafetensorsHeader(buf);
}

const byName = (tensors) => {
  const map = new Map();
  for (const t of tensors) map.set(t.name, t);
  return map;
};

function findSuffix(tensors, suffix) {
  return tensors.find((t) => t.name.endsWith(suffix));
}

function maxIndex(tensors, re) {
  let max = -1;
  for (const t of tensors) {
    const m = t.name.match(re);
    if (m) max = Math.max(max, Number(m[1]));
  }
  return max;
}

function guessHeadDim(qRows, kvRows) {
  for (const hd of [128, 96, 80, 64, 48, 32]) {
    if (qRows % hd === 0 && kvRows % hd === 0 && qRows / hd >= kvRows / hd) return hd;
  }
  return 64;
}

// tensors -> { config, notes[] }  (throws when the naming scheme is unknown)
export function inferConfigFromTensors(tensors, fileName = '') {
  const notes = [];
  const map = byName(tensors);
  const shard = fileName.match(/-of-\d+/) || fileName.match(/model-\d{5}/);
  if (shard) notes.push('⚠ 这似乎是多分片之一:层数/词表可能不完整,建议同时拖入 config.json');

  // ---- GPT-2 naming (Conv1D: shapes stored [in, out]) ----
  if (map.has('wte.weight') || findSuffix(tensors, '.attn.c_attn.weight')) {
    const wte = map.get('wte.weight') || findSuffix(tensors, 'wte.weight');
    const wpe = map.get('wpe.weight') || findSuffix(tensors, 'wpe.weight');
    const layers = maxIndex(tensors, /(?:^|\.)h\.(\d+)\./) + 1;
    const cfc = findSuffix(tensors, '.mlp.c_fc.weight');
    const hidden = wte.shape[1];
    notes.push('按 GPT-2 命名识别(Conv1D 权重为 [in, out])');
    notes.push('n_head 无法从张量形状推断,按 hidden/64 估计');
    return {
      notes,
      config: {
        model_type: 'gpt2', vocab_size: wte.shape[0], n_embd: hidden,
        n_layer: layers, n_head: Math.max(1, Math.round(hidden / 64)),
        n_positions: wpe ? wpe.shape[0] : 1024,
        n_inner: cfc ? cfc.shape[1] : hidden * 4,
        torch_dtype: tensors[0].dtype?.toLowerCase() || 'unknown',
      },
    };
  }

  // ---- llama-family naming ----
  const embed = findSuffix(tensors, 'embed_tokens.weight');
  const q = findSuffix(tensors, 'layers.0.self_attn.q_proj.weight');
  if (!embed || !q) {
    throw new Error('无法识别该 safetensors 的命名规则(支持 llama/qwen/MoE/GPT-2 命名)。可改用 config.json 导入。');
  }
  const k = findSuffix(tensors, 'layers.0.self_attn.k_proj.weight');
  const hidden = embed.shape[1];
  const vocab = embed.shape[0];
  const layers = maxIndex(tensors, /layers\.(\d+)\./) + 1;
  const qRows = q.shape[0], kvRows = k ? k.shape[0] : qRows;
  const headDim = guessHeadDim(qRows, kvRows);
  const heads = qRows / headDim, kvHeads = kvRows / headDim;
  if (heads !== Math.round(heads) || kvHeads !== Math.round(kvHeads)) {
    notes.push('⚠ head_dim 推断可能不准(q/k 行数无法被常见 head_dim 整除)');
  } else {
    notes.push(`head_dim 按整除关系推断为 ${headDim}(${heads} Q 头 / ${kvHeads} KV 头)`);
  }
  const qkNorm = !!findSuffix(tensors, 'layers.0.self_attn.q_norm.weight');
  const tie = !findSuffix(tensors, 'lm_head.weight');
  if (tie) notes.push('未发现独立 lm_head → 判定为 tied embeddings');

  const cfgBase = {
    hidden_size: hidden, vocab_size: vocab, num_hidden_layers: layers,
    num_attention_heads: Math.round(heads), num_key_value_heads: Math.round(kvHeads),
    head_dim: headDim, tie_word_embeddings: tie, rms_norm_eps: 1e-6,
    hidden_act: 'silu', torch_dtype: tensors[0].dtype?.toLowerCase() || 'unknown',
    max_position_embeddings: null,
  };
  notes.push('ctx / rope_theta / eps 无法从权重推断,使用占位值');

  // MoE?
  const expertGate = findSuffix(tensors, 'layers.0.mlp.experts.0.gate_proj.weight');
  const mixtralW1 = findSuffix(tensors, 'layers.0.block_sparse_moe.experts.0.w1.weight');
  if (expertGate) {
    const experts = maxIndex(tensors, /experts\.(\d+)\./) + 1;
    notes.push(`识别为 qwen-MoE 命名:${experts} 个专家;top-k 无法从权重推断,按 8 估计`);
    return {
      notes,
      config: {
        ...cfgBase, model_type: 'qwen3_moe', num_experts: experts,
        num_experts_per_tok: Math.min(8, experts), moe_intermediate_size: expertGate.shape[0],
        intermediate_size: expertGate.shape[0] * 4, norm_topk_prob: true,
      },
    };
  }
  if (mixtralW1) {
    const experts = maxIndex(tensors, /experts\.(\d+)\./) + 1;
    notes.push(`识别为 Mixtral 命名:${experts} 个专家;top-k 按 2 估计`);
    return {
      notes,
      config: {
        ...cfgBase, model_type: 'mixtral', num_local_experts: experts,
        num_experts_per_tok: 2, intermediate_size: mixtralW1.shape[0],
      },
    };
  }
  const gate = findSuffix(tensors, 'layers.0.mlp.gate_proj.weight');
  return {
    notes,
    config: {
      ...cfgBase,
      model_type: qkNorm ? 'qwen3' : 'llama',
      intermediate_size: gate ? gate.shape[0] : hidden * 4,
    },
  };
}
