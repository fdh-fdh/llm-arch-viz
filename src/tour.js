// Guided tour engine (FR-F1): station sequence auto-generated from the IR,
// bbycroft-style "Continue" walkthrough. Each station = a matcher over layout
// items (+ which layer must be expanded) + a one-line narration.
// The controller (main.js) resolves the matcher AFTER expanding, pins the item
// (Inspector shows the full knowledge card) and flies the camera to it.

import { fmtParams } from './parser/ir.js';

// match: { pred(item) -> bool, nth?: number (0-based occurrence) }
function st(label, caption, pred, opts = {}) {
  return { label, caption, match: { pred, nth: opts.nth || 0 }, expand: opts.expand || null };
}

export function buildTour(graph) {
  const m = graph.meta;
  const stations = [];

  stations.push(st('输入 token ids',
    '文字先被 tokenizer 切成整数序列——模型看到的从来不是文字,而是这些 id。',
    (it) => it.type === 'io' && it.title === 'input_ids'));

  stations.push(st('Token Embedding',
    `查表:每个 id 取出一行 ${m.hidden} 维向量,词从此有了可计算的坐标。`,
    (it) => it.path && it.path[0] === 'embed_tokens'));

  // representative layer(s): layer 0 of the first stack, plus the first MoE
  // stack if the first stack is dense (e.g. DeepSeek first_k_dense).
  const stackIdxs = [0];
  if (!graph.stacks[0].segments.some((s) => s.kind === 'moe')) {
    const moeSi = graph.stacks.findIndex((s) => s.segments.some((g) => g.kind === 'moe'));
    if (moeSi > 0) stackIdxs.push(moeSi);
  }

  for (const si of stackIdxs) {
    const stack = graph.stacks[si];
    const li = 0;
    const expand = { si, li };
    const inLayer = (pred) => (it) => it.si === si && it.li === li && pred(it);
    let addSeen = 0;
    if (stackIdxs.length > 1) {
      stations.push(st(`进入 ${stack.label}`,
        si === 0 ? `前 ${stack.count} 层是 dense 层(MoE 模型常见的 warm-up 设计)。`
                 : `后 ${stack.count} 层是 MoE 层——真正的稀疏主体。`,
        inLayer((it) => it.role === 'γ' || it.type === 'tensor'), { expand }));
    }
    for (const seg of stack.segments) {
      if (seg.kind === 'norm') {
        stations.push(st(seg.label,
          '进子层前先把向量拉回统一尺度(Pre-Norm)——那根细条就是逐维缩放 γ。',
          inLayer((it) => it.segLabel === seg.label && it.role === 'γ'), { expand }));
      } else if (seg.kind === 'attn') {
        const v = seg.meta.variant;
        stations.push(st(seg.label,
          v === 'MLA'
            ? `Q/K/V 投影:MLA 先把 K/V 压进 ${seg.meta.kvLora} 维潜向量再解压,KV cache 因此极小。`
            : v === 'GQA'
              ? `Q/K/V 投影:${seg.meta.heads} 个 Q 头共享 ${seg.meta.kvHeads} 组 K/V(${seg.meta.heads / seg.meta.kvHeads}:1)——注意 K/V 矩阵明显更窄。`
              : `Q/K/V 投影:把 hidden 向量投到 ${seg.meta.heads} 个注意力头的子空间。`,
          inLayer((it) => it.segLabel === seg.label &&
            ['q_proj', 'qkv', 'q_a_proj'].includes(it.role)), { expand }));
        stations.push(st('输出投影 o_proj',
          '各 head 的注意力结果拼接后,由 o_proj 投回 hidden 维,写回残差流。',
          inLayer((it) => it.segLabel === seg.label &&
            ['o_proj'].includes(it.role)), { expand }));
      } else if (seg.kind === 'add') {
        stations.push(st('残差 ⊕',
          addSeen === 0
            ? '子层输出加回主干——留意旁边那条旁路轨道,信息和梯度沿它直通首尾。'
            : '第二次残差合并,这一层到此结束,主干继续流向下一层。',
          inLayer((it) => it.type === 'add' && it.title === '+ residual'),
          { expand, nth: addSeen++ }));
      } else if (seg.kind === 'mlp') {
        stations.push(st(seg.label,
          seg.meta.gated
            ? `SwiGLU:gate 决定放行多少、up 提供内容,${seg.meta.inter.toLocaleString('en-US')} 维加工后压回——模型的"知识仓库"。`
            : `两层前馈(${seg.meta.inter.toLocaleString('en-US')} 维 + GELU)——GPT-2 的知识仓库。`,
          inLayer((it) => it.segLabel === seg.label &&
            ['gate_proj', 'c_fc', 'w1', 'w3', 'up_proj'].includes(it.role)), { expand }));
      } else if (seg.kind === 'moe') {
        stations.push(st('MoE Router',
          `小线性层给 ${seg.meta.experts} 个专家打分,每个 token 只选 top-${seg.meta.topK} 干活——${fmtParams(graph.meta.params.total)} 总参、${fmtParams(graph.meta.params.active)} 激活的秘密。`,
          inLayer((it) => it.role === 'router'), { expand }));
        stations.push(st(`专家阵列 ×${seg.meta.experts}`,
          `每个专家是一个窄版 SwiGLU;它们训练中自发分工(代码/多语言/…)。点击可展开专家网格。`,
          inLayer((it) => it.type === 'expertAgg' || it.type === 'expert'), { expand }));
        if (seg.sharedProto) {
          stations.push(st('共享专家',
            '所有 token 必经的"通识专家",与路由专家的输出相加。',
            inLayer((it) => (it.title || '').startsWith('共享专家')), { expand }));
        }
        stations.push(st('Σ 加权合并',
          `top-${seg.meta.topK} 个专家的输出按路由权重加权求和,写回残差流。`,
          inLayer((it) => it.type === 'add' && (it.title || '').startsWith('Σ')), { expand }));
      }
    }
  }

  const repeated = graph.stacks.reduce((a, s) => a + s.count, 0);
  stations.push(st(`× ${repeated} 层堆叠`,
    `同样的结构重复 ${repeated} 次,残差流一路加工——这就是"深度"的含义。`,
    (it) => it.type === 'stackAgg'));

  stations.push(st('Final Norm',
    '出塔前最后一次归一化,为输出投影准备干净的向量。',
    (it) => it.path && it.path[0] === 'final_norm'));

  stations.push(st('LM Head',
    m.params.tied
      ? '投回词表空间得到每个候选词的分数——这里与 Embedding 共享同一份权重(tied)。'
      : `把 ${m.hidden} 维隐向量投到 ${m.vocab.toLocaleString('en-US')} 个词的分数上。`,
    (it) => it.path && it.path[0] === 'lm_head'));

  stations.push(st('Logits → 采样',
    '分数过 softmax 变成概率,采出下一个 token,拼回输入,循环往复——自回归生成。',
    (it) => it.type === 'io' && it.title === 'logits'));

  return stations;
}

// Resolve a station's matcher against the current layout items.
export function resolveStation(station, layout) {
  let seen = 0;
  for (let i = 0; i < layout.soa.count; i++) {
    const it = layout.items[i];
    if (it && station.match.pred(it)) {
      if (seen === station.match.nth) return it;
      seen++;
    }
  }
  return null;
}
