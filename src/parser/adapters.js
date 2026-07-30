// Config adapter registry: HF config.json -> ArchGraph IR.
// Channel A of the dual-channel design: semantic authority comes from config;
// shapes are fully derivable for known families (no weight download needed).

function prod(shape) { return shape.reduce((a, b) => a * b, 1); }

function T(role, name, shape, kind = 'matrix') {
  return { role, name, shape, kind, params: prod(shape) };
}

function normTensor(name, dim, withBias = false) {
  const t = [T('weight', name + '.weight', [dim], 'vector')];
  if (withBias) t.push(T('bias', name + '.bias', [dim], 'vector'));
  return t;
}

function get(cfg, keys, fallback = undefined) {
  for (const k of keys) if (cfg[k] !== undefined && cfg[k] !== null) return cfg[k];
  return fallback;
}

// ---------------------------------------------------------------------------
// Dense decoder family: llama / mistral / qwen2 / qwen3 / gemma(2,3) / phi3 / olmo…
// ---------------------------------------------------------------------------
function denseAttention(cfg, dims, opts = {}) {
  const { hidden, heads, kvHeads, headDim } = dims;
  const qkNorm = opts.qkNorm ?? ['qwen3', 'qwen3_moe', 'gemma3', 'gemma3_text', 'olmo2'].includes(cfg.model_type);
  const bias = get(cfg, ['attention_bias'], cfg.model_type === 'qwen2' ? true : false);
  const tensors = [
    T('q_proj', 'model.layers.{i}.self_attn.q_proj.weight', [heads * headDim, hidden]),
    T('k_proj', 'model.layers.{i}.self_attn.k_proj.weight', [kvHeads * headDim, hidden]),
    T('v_proj', 'model.layers.{i}.self_attn.v_proj.weight', [kvHeads * headDim, hidden]),
    T('o_proj', 'model.layers.{i}.self_attn.o_proj.weight', [hidden, heads * headDim]),
  ];
  if (bias) {
    tensors.push(T('q_bias', 'model.layers.{i}.self_attn.q_proj.bias', [heads * headDim], 'vector'));
    tensors.push(T('k_bias', 'model.layers.{i}.self_attn.k_proj.bias', [kvHeads * headDim], 'vector'));
    tensors.push(T('v_bias', 'model.layers.{i}.self_attn.v_proj.bias', [kvHeads * headDim], 'vector'));
  }
  if (qkNorm) {
    tensors.push(T('q_norm', 'model.layers.{i}.self_attn.q_norm.weight', [headDim], 'vector'));
    tensors.push(T('k_norm', 'model.layers.{i}.self_attn.k_norm.weight', [headDim], 'vector'));
  }
  const variant = kvHeads === heads ? 'MHA' : (kvHeads === 1 ? 'MQA' : 'GQA');
  return {
    kind: 'attn',
    label: `Self-Attention (${variant})`,
    meta: {
      variant, heads, kvHeads, headDim,
      ropeTheta: get(cfg, ['rope_theta']),
      sliding: get(cfg, ['sliding_window']) && get(cfg, ['use_sliding_window'], true) ? get(cfg, ['sliding_window']) : null,
      qkNorm,
    },
    tensors,
  };
}

function gatedMLP(hidden, inter, act, prefix = 'model.layers.{i}.mlp') {
  return {
    kind: 'mlp',
    label: 'MLP (SwiGLU)',
    meta: { inter, act, gated: true },
    tensors: [
      T('gate_proj', `${prefix}.gate_proj.weight`, [inter, hidden]),
      T('up_proj', `${prefix}.up_proj.weight`, [inter, hidden]),
      T('down_proj', `${prefix}.down_proj.weight`, [hidden, inter]),
    ],
  };
}

function denseAdapter(cfg) {
  const hidden = get(cfg, ['hidden_size', 'n_embd', 'd_model']);
  const layers = get(cfg, ['num_hidden_layers', 'n_layer', 'num_layers']);
  const heads = get(cfg, ['num_attention_heads', 'n_head']);
  const kvHeads = get(cfg, ['num_key_value_heads'], heads);
  const headDim = get(cfg, ['head_dim'], Math.floor(hidden / heads));
  const inter = get(cfg, ['intermediate_size', 'n_inner'], hidden * 4);
  const vocab = get(cfg, ['vocab_size']);
  const tie = get(cfg, ['tie_word_embeddings'], false);
  const act = get(cfg, ['hidden_act', 'hidden_activation'], 'silu');
  const eps = get(cfg, ['rms_norm_eps', 'layer_norm_epsilon'], 1e-6);
  const dims = { hidden, heads, kvHeads, headDim };

  const layerProto = {
    segments: [
      { kind: 'norm', label: 'input RMSNorm', meta: { eps }, tensors: normTensor('model.layers.{i}.input_layernorm', hidden) },
      denseAttention(cfg, dims),
      { kind: 'add', label: '+ residual' },
      { kind: 'norm', label: 'post-attn RMSNorm', meta: { eps }, tensors: normTensor('model.layers.{i}.post_attention_layernorm', hidden) },
      gatedMLP(hidden, inter, act),
      { kind: 'add', label: '+ residual' },
    ],
  };

  return {
    dims: { hidden, layers, vocab, ctx: get(cfg, ['max_position_embeddings', 'n_positions']) },
    embedding: { tensors: [T('embed', 'model.embed_tokens.weight', [vocab, hidden])] },
    stacks: [{ label: 'Decoder Layer', count: layers, layer: layerProto }],
    finalNorm: { tensors: normTensor('model.norm', hidden), meta: { eps } },
    lmHead: tie ? { tied: true, tensors: [] } : { tied: false, tensors: [T('lm_head', 'lm_head.weight', [vocab, hidden])] },
  };
}

// ---------------------------------------------------------------------------
// GPT-2 family (learned positions, fused qkv, LayerNorm with bias, dense MLP)
// ---------------------------------------------------------------------------
function gpt2Adapter(cfg) {
  const hidden = get(cfg, ['n_embd', 'hidden_size']);
  const layers = get(cfg, ['n_layer', 'num_hidden_layers']);
  const heads = get(cfg, ['n_head', 'num_attention_heads']);
  const vocab = get(cfg, ['vocab_size']);
  const ctx = get(cfg, ['n_positions', 'n_ctx', 'max_position_embeddings'], 1024);
  const inter = get(cfg, ['n_inner'], hidden * 4) || hidden * 4;

  const layerProto = {
    segments: [
      { kind: 'norm', label: 'LayerNorm 1', meta: { eps: get(cfg, ['layer_norm_epsilon'], 1e-5) }, tensors: normTensor('h.{i}.ln_1', hidden, true) },
      {
        kind: 'attn',
        label: 'Self-Attention (MHA, fused QKV)',
        meta: { variant: 'MHA', heads, kvHeads: heads, headDim: hidden / heads, learnedPos: true },
        tensors: [
          T('qkv', 'h.{i}.attn.c_attn.weight', [hidden, 3 * hidden]),
          T('qkv_bias', 'h.{i}.attn.c_attn.bias', [3 * hidden], 'vector'),
          T('o_proj', 'h.{i}.attn.c_proj.weight', [hidden, hidden]),
          T('o_bias', 'h.{i}.attn.c_proj.bias', [hidden], 'vector'),
        ],
      },
      { kind: 'add', label: '+ residual' },
      { kind: 'norm', label: 'LayerNorm 2', tensors: normTensor('h.{i}.ln_2', hidden, true) },
      {
        kind: 'mlp',
        label: 'MLP (GELU)',
        meta: { inter, act: 'gelu', gated: false },
        tensors: [
          T('c_fc', 'h.{i}.mlp.c_fc.weight', [hidden, inter]),
          T('c_fc_bias', 'h.{i}.mlp.c_fc.bias', [inter], 'vector'),
          T('c_proj', 'h.{i}.mlp.c_proj.weight', [inter, hidden]),
          T('c_proj_bias', 'h.{i}.mlp.c_proj.bias', [hidden], 'vector'),
        ],
      },
      { kind: 'add', label: '+ residual' },
    ],
  };

  return {
    dims: { hidden, layers, vocab, ctx },
    embedding: {
      tensors: [
        T('wte', 'wte.weight', [vocab, hidden]),
        T('wpe', 'wpe.weight', [ctx, hidden]),
      ],
    },
    stacks: [{ label: 'Transformer Block', count: layers, layer: layerProto }],
    finalNorm: { tensors: normTensor('ln_f', hidden, true) },
    lmHead: { tied: true, tensors: [] },
  };
}

// ---------------------------------------------------------------------------
// MoE families
// ---------------------------------------------------------------------------
function moeSegment(cfg, hidden, opts) {
  const { experts, topK, expertInter, sharedInter, sharedCount, prefix, normTopK } = opts;
  const expertProto = {
    tensors: [
      T('gate_proj', `${prefix}.experts.{j}.gate_proj.weight`, [expertInter, hidden]),
      T('up_proj', `${prefix}.experts.{j}.up_proj.weight`, [expertInter, hidden]),
      T('down_proj', `${prefix}.experts.{j}.down_proj.weight`, [hidden, expertInter]),
    ],
  };
  const seg = {
    kind: 'moe',
    label: `Sparse MoE (${experts} experts, top-${topK})`,
    meta: { experts, topK, expertInter, sharedExperts: sharedCount || 0, sharedInter: sharedInter || 0, normTopK: !!normTopK },
    routerTensors: [T('router', `${prefix}.gate.weight`, [experts, hidden])],
    expertProto,
  };
  if (sharedCount > 0) {
    seg.sharedProto = {
      count: sharedCount,
      tensors: [
        T('gate_proj', `${prefix}.shared_experts.gate_proj.weight`, [sharedInter * sharedCount, hidden]),
        T('up_proj', `${prefix}.shared_experts.up_proj.weight`, [sharedInter * sharedCount, hidden]),
        T('down_proj', `${prefix}.shared_experts.down_proj.weight`, [hidden, sharedInter * sharedCount]),
      ],
    };
  }
  return seg;
}

function qwenMoeAdapter(cfg) {
  const base = denseAdapter(cfg);
  const hidden = get(cfg, ['hidden_size']);
  const eps = get(cfg, ['rms_norm_eps'], 1e-6);
  const experts = get(cfg, ['num_experts', 'n_routed_experts']);
  const topK = get(cfg, ['num_experts_per_tok', 'top_k']);
  const expertInter = get(cfg, ['moe_intermediate_size']);
  const isQwen2Moe = cfg.model_type === 'qwen2_moe';
  const sharedInter = isQwen2Moe ? get(cfg, ['shared_expert_intermediate_size'], 0) : 0;

  const dims = {
    hidden,
    heads: get(cfg, ['num_attention_heads']),
    kvHeads: get(cfg, ['num_key_value_heads']),
    headDim: get(cfg, ['head_dim'], Math.floor(hidden / get(cfg, ['num_attention_heads']))),
  };
  const moe = moeSegment(cfg, hidden, {
    experts, topK, expertInter,
    sharedInter, sharedCount: sharedInter ? 1 : 0,
    prefix: 'model.layers.{i}.mlp',
    normTopK: get(cfg, ['norm_topk_prob'], false),
  });
  const layerProto = {
    segments: [
      { kind: 'norm', label: 'input RMSNorm', meta: { eps }, tensors: normTensor('model.layers.{i}.input_layernorm', hidden) },
      denseAttention(cfg, dims),
      { kind: 'add', label: '+ residual' },
      { kind: 'norm', label: 'post-attn RMSNorm', meta: { eps }, tensors: normTensor('model.layers.{i}.post_attention_layernorm', hidden) },
      moe,
      { kind: 'add', label: '+ residual' },
    ],
  };
  base.stacks = [{ label: 'MoE Decoder Layer', count: get(cfg, ['num_hidden_layers']), layer: layerProto }];
  return base;
}

function mixtralAdapter(cfg) {
  const base = denseAdapter(cfg);
  const hidden = get(cfg, ['hidden_size']);
  const eps = get(cfg, ['rms_norm_eps'], 1e-5);
  const experts = get(cfg, ['num_local_experts']);
  const topK = get(cfg, ['num_experts_per_tok']);
  const inter = get(cfg, ['intermediate_size']);
  const dims = {
    hidden,
    heads: get(cfg, ['num_attention_heads']),
    kvHeads: get(cfg, ['num_key_value_heads']),
    headDim: get(cfg, ['head_dim'], Math.floor(hidden / get(cfg, ['num_attention_heads']))),
  };
  const moe = {
    kind: 'moe',
    label: `Sparse MoE (${experts} experts, top-${topK})`,
    meta: { experts, topK, expertInter: inter, sharedExperts: 0, sharedInter: 0 },
    routerTensors: [T('router', 'model.layers.{i}.block_sparse_moe.gate.weight', [experts, hidden])],
    expertProto: {
      tensors: [
        T('w1', 'model.layers.{i}.block_sparse_moe.experts.{j}.w1.weight', [inter, hidden]),
        T('w3', 'model.layers.{i}.block_sparse_moe.experts.{j}.w3.weight', [inter, hidden]),
        T('w2', 'model.layers.{i}.block_sparse_moe.experts.{j}.w2.weight', [hidden, inter]),
      ],
    },
  };
  const layerProto = {
    segments: [
      { kind: 'norm', label: 'input RMSNorm', meta: { eps }, tensors: normTensor('model.layers.{i}.input_layernorm', hidden) },
      denseAttention(cfg, dims),
      { kind: 'add', label: '+ residual' },
      { kind: 'norm', label: 'post-attn RMSNorm', meta: { eps }, tensors: normTensor('model.layers.{i}.post_attention_layernorm', hidden) },
      moe,
      { kind: 'add', label: '+ residual' },
    ],
  };
  base.stacks = [{ label: 'MoE Decoder Layer', count: get(cfg, ['num_hidden_layers']), layer: layerProto }];
  return base;
}

// DeepSeek V2/V3: MLA attention + first_k dense layers + shared experts
function deepseekAdapter(cfg) {
  const hidden = get(cfg, ['hidden_size']);
  const layers = get(cfg, ['num_hidden_layers']);
  const heads = get(cfg, ['num_attention_heads']);
  const vocab = get(cfg, ['vocab_size']);
  const eps = get(cfg, ['rms_norm_eps'], 1e-6);
  const firstDense = get(cfg, ['first_k_dense_replace'], 0);
  const denseInter = get(cfg, ['intermediate_size']);
  const moeInter = get(cfg, ['moe_intermediate_size']);
  const experts = get(cfg, ['n_routed_experts']);
  const topK = get(cfg, ['num_experts_per_tok']);
  const nShared = get(cfg, ['n_shared_experts'], 0);
  const qLora = get(cfg, ['q_lora_rank']);
  const kvLora = get(cfg, ['kv_lora_rank']);
  const qkNope = get(cfg, ['qk_nope_head_dim'], 128);
  const qkRope = get(cfg, ['qk_rope_head_dim'], 64);
  const vDim = get(cfg, ['v_head_dim'], 128);
  const qDim = qkNope + qkRope;

  const attnTensors = [];
  if (qLora) {
    attnTensors.push(T('q_a_proj', 'model.layers.{i}.self_attn.q_a_proj.weight', [qLora, hidden]));
    attnTensors.push(T('q_a_norm', 'model.layers.{i}.self_attn.q_a_layernorm.weight', [qLora], 'vector'));
    attnTensors.push(T('q_b_proj', 'model.layers.{i}.self_attn.q_b_proj.weight', [heads * qDim, qLora]));
  } else {
    attnTensors.push(T('q_proj', 'model.layers.{i}.self_attn.q_proj.weight', [heads * qDim, hidden]));
  }
  attnTensors.push(T('kv_a_proj', 'model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight', [kvLora + qkRope, hidden]));
  attnTensors.push(T('kv_a_norm', 'model.layers.{i}.self_attn.kv_a_layernorm.weight', [kvLora], 'vector'));
  attnTensors.push(T('kv_b_proj', 'model.layers.{i}.self_attn.kv_b_proj.weight', [heads * (qkNope + vDim), kvLora]));
  attnTensors.push(T('o_proj', 'model.layers.{i}.self_attn.o_proj.weight', [hidden, heads * vDim]));

  const mlaSegment = {
    kind: 'attn',
    label: 'Multi-head Latent Attention (MLA)',
    meta: { variant: 'MLA', heads, kvHeads: heads, headDim: qDim, kvLora, qLora, ropeTheta: get(cfg, ['rope_theta']) },
    tensors: attnTensors,
  };

  const mkLayer = (ffnSegment) => ({
    segments: [
      { kind: 'norm', label: 'input RMSNorm', meta: { eps }, tensors: normTensor('model.layers.{i}.input_layernorm', hidden) },
      mlaSegment,
      { kind: 'add', label: '+ residual' },
      { kind: 'norm', label: 'post-attn RMSNorm', meta: { eps }, tensors: normTensor('model.layers.{i}.post_attention_layernorm', hidden) },
      ffnSegment,
      { kind: 'add', label: '+ residual' },
    ],
  });

  const stacks = [];
  if (firstDense > 0) {
    stacks.push({ label: 'Dense Decoder Layer', count: firstDense, layer: mkLayer(gatedMLP(hidden, denseInter, 'silu')) });
  }
  const moe = moeSegment(cfg, hidden, {
    experts, topK, expertInter: moeInter,
    sharedInter: moeInter, sharedCount: nShared,
    prefix: 'model.layers.{i}.mlp',
    normTopK: get(cfg, ['norm_topk_prob'], true),
  });
  stacks.push({ label: 'MoE Decoder Layer', count: layers - firstDense, layer: mkLayer(moe) });

  return {
    dims: { hidden, layers, vocab, ctx: get(cfg, ['max_position_embeddings']) },
    embedding: { tensors: [T('embed', 'model.embed_tokens.weight', [vocab, hidden])] },
    stacks,
    finalNorm: { tensors: normTensor('model.norm', hidden), meta: { eps } },
    lmHead: get(cfg, ['tie_word_embeddings'], false)
      ? { tied: true, tensors: [] }
      : { tied: false, tensors: [T('lm_head', 'lm_head.weight', [vocab, hidden])] },
  };
}

// ---------------------------------------------------------------------------
// Registry + generic fallback
// ---------------------------------------------------------------------------
const DENSE_TYPES = [
  'llama', 'mistral', 'qwen2', 'qwen3', 'gemma', 'gemma2', 'gemma3_text', 'gemma3',
  'phi', 'phi3', 'phi4', 'olmo', 'olmo2', 'granite', 'internlm2', 'exaone', 'glm', 'chatglm',
  'starcoder2', 'stablelm', 'cohere', 'smollm', 'smollm2', 'minicpm', 'yi',
];

const REGISTRY = new Map();
for (const t of DENSE_TYPES) REGISTRY.set(t, denseAdapter);
REGISTRY.set('gpt2', gpt2Adapter);
REGISTRY.set('gpt_neox', denseAdapter);
REGISTRY.set('qwen2_moe', qwenMoeAdapter);
REGISTRY.set('qwen3_moe', qwenMoeAdapter);
REGISTRY.set('mixtral', mixtralAdapter);
REGISTRY.set('deepseek_v2', deepseekAdapter);
REGISTRY.set('deepseek_v3', deepseekAdapter);

export function resolveAdapter(config) {
  // multimodal wrappers: descend into text_config
  let cfg = config;
  if (!cfg.model_type && cfg.text_config) cfg = { ...cfg.text_config };
  else if (cfg.text_config && !get(cfg, ['hidden_size', 'n_embd'])) cfg = { ...cfg.text_config, model_type: cfg.text_config.model_type || cfg.model_type };

  const type = cfg.model_type || 'unknown';
  if (REGISTRY.has(type)) {
    return { adapter: REGISTRY.get(type), cfg, matched: type, degraded: false };
  }
  // generic fallback: try dense-decoder field heuristics
  const hasCore = get(cfg, ['hidden_size', 'n_embd']) && get(cfg, ['num_hidden_layers', 'n_layer']) && get(cfg, ['num_attention_heads', 'n_head']);
  if (hasCore) {
    const isMoe = get(cfg, ['num_experts', 'num_local_experts', 'n_routed_experts']);
    return { adapter: isMoe ? qwenMoeAdapter : denseAdapter, cfg, matched: 'generic-' + (isMoe ? 'moe' : 'dense'), degraded: true };
  }
  return { adapter: null, cfg, matched: null, degraded: true };
}

export const SUPPORTED_TYPES = [...REGISTRY.keys()];
