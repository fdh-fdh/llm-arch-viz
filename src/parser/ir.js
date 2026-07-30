// IR builder: adapter output -> ArchGraph with parameter budget + capability tier.
// Prototype × repeat-count representation: one layer prototype per stack, never N copies.

import { resolveAdapter } from './adapters.js';

function sumTensors(tensors) {
  return (tensors || []).reduce((a, t) => a + t.params, 0);
}

function segmentParams(seg) {
  if (seg.kind === 'moe') {
    const router = sumTensors(seg.routerTensors);
    const expert = sumTensors(seg.expertProto.tensors);
    const shared = seg.sharedProto ? sumTensors(seg.sharedProto.tensors) : 0;
    return {
      total: router + expert * seg.meta.experts + shared,
      active: router + expert * seg.meta.topK + shared,
      router, expert, shared,
    };
  }
  const p = sumTensors(seg.tensors);
  return { total: p, active: p };
}

export function buildGraph(config, source = { kind: 'manual' }) {
  const { adapter, cfg, matched, degraded } = resolveAdapter(config);
  if (!adapter) {
    throw new Error('无法识别该配置:缺少 hidden_size / num_hidden_layers / num_attention_heads 等核心字段');
  }
  const model = adapter(cfg);
  const { hidden, layers, vocab, ctx } = model.dims;

  const embParams = sumTensors(model.embedding.tensors);
  const finalNormParams = sumTensors(model.finalNorm.tensors);
  const headParams = model.lmHead.tied ? 0 : sumTensors(model.lmHead.tensors);

  let layersTotal = 0;
  let layersActive = 0;
  let maxExperts = 0;
  const stacks = model.stacks.map((stack, si) => {
    let perLayer = 0, perLayerActive = 0;
    const segs = stack.layer.segments.map((seg) => {
      const p = segmentParams(seg);
      perLayer += p.total;
      perLayerActive += p.active;
      if (seg.kind === 'moe') maxExperts = Math.max(maxExperts, seg.meta.experts);
      return { ...seg, paramInfo: p };
    });
    layersTotal += perLayer * stack.count;
    layersActive += perLayerActive * stack.count;
    return { id: 'stack' + si, label: stack.label, count: stack.count, segments: segs, perLayer, perLayerActive };
  });

  const total = embParams + layersTotal + finalNormParams + headParams;
  const active = embParams + layersActive + finalNormParams + headParams;

  // Capability tier (animation gating etc. — see 内存优化 doc §3)
  let tier;
  if (total < 0.5e9 && layers <= 24) tier = 'T0';
  else if (total >= 100e9) tier = 'T3';
  else if (total >= 15e9 || maxExperts >= 64) tier = 'T2';
  else tier = 'T1';

  return {
    formatVersion: 1,
    meta: {
      name: source.name || source.repoId || cfg.model_type || 'model',
      modelType: cfg.model_type || 'unknown',
      matchedAdapter: matched,
      degraded,
      source,
      hidden, layers, vocab, ctx,
      heads: cfg.num_attention_heads || cfg.n_head,
      kvHeads: cfg.num_key_value_heads || cfg.num_attention_heads || cfg.n_head,
      dtype: cfg.torch_dtype || 'unknown',
      tier,
      maxExperts,
      params: {
        total, active,
        embedding: embParams,
        lmHead: headParams,
        tied: !!model.lmHead.tied,
        finalNorm: finalNormParams,
        layers: layersTotal,
      },
    },
    embedding: model.embedding,
    stacks,
    finalNorm: model.finalNorm,
    lmHead: model.lmHead,
    rawConfig: config,
  };
}

export function fmtParams(n) {
  if (n >= 1e12) return (n / 1e12).toFixed(2) + ' T';
  if (n >= 1e9) return (n / 1e9).toFixed(2) + ' B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + ' M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + ' K';
  return String(n);
}

export function fmtShape(shape) {
  return '[' + shape.map((d) => d.toLocaleString('en-US')).join(' × ') + ']';
}

// Expand a tensor name template for a concrete layer / expert index.
export function tensorName(tpl, layerIdx, expertIdx) {
  return tpl.replace('{i}', String(layerIdx)).replace('{j}', String(expertIdx ?? 0));
}

export const TIER_INFO = {
  T0: { label: 'T0 教学级', animation: 'full', maxExpand: Infinity, desc: '全功能:数据流动画 + 全展开' },
  T1: { label: 'T1 常规', animation: 'full', maxExpand: 5, desc: '动画默认开启,最多同时展开 5 层' },
  T2: { label: 'T2 大型', animation: 'focus', maxExpand: 3, desc: '仅单层聚焦模式提供动画' },
  T3: { label: 'T3 超大', animation: 'none', maxExpand: 2, desc: '完全静态,LOD 聚合渲染' },
};
