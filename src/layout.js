// Layout engine: ArchGraph IR + fold state -> instance SoA (pos/scale/color/flag)
// plus a parallel `items` metadata array (tooltips / click actions) and overlay labels.
// Collapsed runs of identical layers render as ONE aggregate slab ("×N") — the
// core "exploit the repetition" strategy from the memory-optimization doc.

import { tensorName, fmtParams, fmtShape } from './parser/ir.js';

const S = 0.55;            // log2(dim) -> world units
const GAP = 1.4;           // vertical gap between segments
const LAYER_GAP = 2.2;

function L(d) { return Math.max(0.6, Math.log2((d || 1) + 1) * S); }

const COLORS = {
  io:        [0.55, 0.60, 0.70],
  embedding: [0.36, 0.43, 0.88],
  norm:      [0.93, 0.68, 0.25],
  q:         [0.62, 0.40, 0.93],
  k:         [0.52, 0.34, 0.85],
  v:         [0.45, 0.30, 0.78],
  o:         [0.70, 0.48, 0.95],
  attnMisc:  [0.58, 0.42, 0.88],
  mlp:       [0.13, 0.62, 0.55],
  router:    [0.05, 0.48, 0.36],
  expert:    [0.18, 0.68, 0.52],
  expertAgg: [0.10, 0.55, 0.44],
  shared:    [0.10, 0.65, 0.68],
  stackAgg:  [0.42, 0.52, 0.72],
  lmHead:    [0.30, 0.36, 0.80],
  connector: [0.72, 0.76, 0.82],
  add:       [0.60, 0.64, 0.72],
};

function roleColor(role) {
  if (role.startsWith('q')) return COLORS.q;
  if (role.startsWith('k')) return COLORS.k;
  if (role.startsWith('v')) return COLORS.v;
  if (role.startsWith('o')) return COLORS.o;
  if (role === 'qkv') return COLORS.attnMisc;
  return COLORS.attnMisc;
}

export class LayoutBuilder {
  constructor() {
    this.capacity = 4096;
    this._alloc(this.capacity);
  }

  _alloc(cap) {
    this.pos = new Float32Array(cap * 3);
    this.scale = new Float32Array(cap * 3);
    this.color = new Float32Array(cap * 3);
    this.flag = new Float32Array(cap);
    this.items = new Array(cap);
  }

  _ensure(n) {
    if (n <= this.capacity) return;
    while (this.capacity < n) this.capacity *= 2;
    const old = { pos: this.pos, scale: this.scale, color: this.color, flag: this.flag, items: this.items, count: this.count };
    this._alloc(this.capacity);
    this.pos.set(old.pos.subarray(0, old.count * 3));
    this.scale.set(old.scale.subarray(0, old.count * 3));
    this.color.set(old.color.subarray(0, old.count * 3));
    this.flag.set(old.flag.subarray(0, old.count));
    for (let i = 0; i < old.count; i++) this.items[i] = old.items[i];
  }

  box(x, y, z, sx, sy, sz, rgb, flag, item) {
    this._ensure(this.count + 1);
    const i = this.count++;
    const i3 = i * 3;
    this.pos[i3] = x; this.pos[i3 + 1] = y; this.pos[i3 + 2] = z;
    this.scale[i3] = sx; this.scale[i3 + 1] = sy; this.scale[i3 + 2] = sz;
    this.color[i3] = rgb[0]; this.color[i3 + 1] = rgb[1]; this.color[i3 + 2] = rgb[2];
    this.flag[i] = flag;
    this.items[i] = item || null;
    if (item) { item.x = x; item.y = y; item.z = z; item.sy = sy; }
    return i;
  }

  connector(yTop, yBottom, x = 0, z = 0) {
    const h = yTop - yBottom;
    if (h <= 0.01) return;
    this.box(x, (yTop + yBottom) / 2, z, 0.16, h, 0.16, COLORS.connector, 1, null);
  }

  build(graph, state) {
    this.count = 0;
    this.labels = [];
    this.expandedRanges = [];
    let y = 0;
    const bounds = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
    const grow = (x, yy, z, sx, sy, sz) => {
      bounds.min[0] = Math.min(bounds.min[0], x - sx / 2); bounds.max[0] = Math.max(bounds.max[0], x + sx / 2);
      bounds.min[1] = Math.min(bounds.min[1], yy - sy / 2); bounds.max[1] = Math.max(bounds.max[1], yy + sy / 2);
      bounds.min[2] = Math.min(bounds.min[2], z - sz / 2); bounds.max[2] = Math.max(bounds.max[2], z + sz / 2);
    };
    const boxG = (x, yy, z, sx, sy, sz, rgb, flag, item) => {
      grow(x, yy, z, sx, sy, sz);
      return this.box(x, yy, z, sx, sy, sz, rgb, flag, item);
    };
    const meta = graph.meta;

    // ---- input ----
    boxG(0, y, 0, 3.2, 0.5, 1.6, COLORS.io, 0, {
      type: 'io', title: 'input_ids', lines: [`[B, T] · token ids < ${meta.vocab.toLocaleString('en-US')}`],
    });
    this.labels.push({ x: 0, y: y + 0.9, z: 0, text: 'input_ids', kind: 'minor' });
    let prevBottom = y - 0.25;
    y -= 0.5 + GAP;

    // ---- embedding ----
    {
      const t = graph.embedding.tensors[0];
      const sx = L(meta.hidden), sz = L(meta.vocab);
      this.connector(prevBottom, y + 0.4);
      boxG(0, y, 0, sx, 0.8, sz, COLORS.embedding, 0, {
        type: 'tensor', title: 'Token Embedding',
        lines: graph.embedding.tensors.map((tt) => `${tt.name}  ${fmtShape(tt.shape)}  ${fmtParams(tt.params)}`),
        params: meta.params.embedding,
      });
      this.labels.push({ x: sx / 2 + 1.2, y, z: 0, text: `Embedding ${fmtParams(meta.params.embedding)}`, kind: 'block' });
      prevBottom = y - 0.4;
      y -= 0.8 + GAP;
    }

    // ---- stacks ----
    let globalLayer = 0;
    for (let si = 0; si < graph.stacks.length; si++) {
      const stack = graph.stacks[si];
      let li = 0;
      while (li < stack.count) {
        const key = si + ':' + li;
        if (state.expanded.has(key)) {
          const topY = y;
          const r = this._layer(graph, si, li, globalLayer + li, y, prevBottom, state, boxG, grow);
          this.expandedRanges.push({ key, top: topY, bottom: r.bottom });
          prevBottom = r.bottom;
          y = r.nextY;
          li++;
        } else {
          // collapsed run -> one aggregate slab
          let run = 0;
          while (li + run < stack.count && !state.expanded.has(si + ':' + (li + run))) run++;
          const h = Math.min(1.1 + run * 0.16, 14);
          const sx = L(meta.hidden) * 1.6, sz = L(meta.hidden) * 1.15;
          this.connector(prevBottom, y + h / 2);
          boxG(0, y, 0, sx, h, sz, COLORS.stackAgg, 2, {
            type: 'stackAgg', si, liStart: li, run,
            title: `${stack.label} × ${run}`,
            lines: [
              `层 ${globalLayer + li} – ${globalLayer + li + run - 1}`,
              `每层 ${fmtParams(stack.perLayer)}(激活 ${fmtParams(stack.perLayerActive)})`,
              '点击展开第一层',
            ],
            action: { kind: 'expandLayer', si, li },
          });
          this.labels.push({ x: sx / 2 + 1.2, y, z: 0, text: `${stack.label} ×${run}`, kind: 'block' });
          prevBottom = y - h / 2;
          y -= h + LAYER_GAP;
          li += run;
        }
      }
      globalLayer += stack.count;
    }

    // ---- final norm ----
    {
      const sx = L(meta.hidden);
      this.connector(prevBottom, y + 0.15);
      boxG(0, y, 0, sx, 0.3, 1.4, COLORS.norm, 0, {
        type: 'tensor', title: 'Final RMSNorm',
        lines: graph.finalNorm.tensors.map((t) => `${t.name}  ${fmtShape(t.shape)}`),
      });
      this.labels.push({ x: sx / 2 + 1.2, y, z: 0, text: 'Final Norm', kind: 'minor' });
      prevBottom = y - 0.15;
      y -= 0.3 + GAP;
    }

    // ---- lm head ----
    {
      const tied = graph.lmHead.tied;
      const sx = L(meta.hidden), sz = L(meta.vocab);
      this.connector(prevBottom, y + 0.4);
      boxG(0, y, 0, sx, 0.8, sz, COLORS.lmHead, 0, {
        type: 'tensor', title: tied ? 'LM Head(与 Embedding 共享权重)' : 'LM Head',
        lines: tied
          ? [`tied → wte 转置  [${meta.hidden.toLocaleString('en-US')} × ${meta.vocab.toLocaleString('en-US')}]`]
          : graph.lmHead.tensors.map((t) => `${t.name}  ${fmtShape(t.shape)}  ${fmtParams(t.params)}`),
      });
      this.labels.push({ x: sx / 2 + 1.2, y, z: 0, text: tied ? 'LM Head (tied)' : `LM Head ${fmtParams(meta.params.lmHead)}`, kind: 'block' });
      prevBottom = y - 0.4;
      y -= 0.8 + GAP;
    }

    // ---- logits ----
    boxG(0, y, 0, 3.2, 0.5, 1.6, COLORS.io, 0, {
      type: 'io', title: 'logits', lines: [`[B, T, ${meta.vocab.toLocaleString('en-US')}]`],
    });
    this.connector(prevBottom, y + 0.25);
    this.labels.push({ x: 0, y: y - 0.9, z: 0, text: 'logits', kind: 'minor' });

    return {
      soa: { pos: this.pos, scale: this.scale, color: this.color, flag: this.flag, count: this.count },
      items: this.items,
      labels: this.labels,
      bounds,
      spineTop: 0.5,
      spineBottom: y - 0.5,
    };
  }

  // Expanded layer layout. Returns {bottom, nextY}.
  _layer(graph, si, li, globalIdx, y, prevBottom, state, boxG) {
    const stack = graph.stacks[si];
    const meta = graph.meta;
    this.labels.push({ x: -L(meta.hidden) * 1.4 - 1.5, y: y - 2, z: 0, text: `Layer ${globalIdx}`, kind: 'layer', action: { kind: 'collapseLayer', si, li } });

    for (const seg of stack.segments) {
      if (seg.kind === 'add') {
        this.connector(prevBottom, y + 0.3);
        boxG(0, y, 0, 0.7, 0.6, 0.7, COLORS.add, 0, {
          type: 'add', title: '+ residual', lines: ['残差连接'],
        });
        prevBottom = y - 0.3;
        y -= 0.6 + GAP * 0.7;
      } else if (seg.kind === 'norm') {
        const sx = L(meta.hidden);
        this.connector(prevBottom, y + 0.14);
        boxG(0, y, 0, sx, 0.28, 1.3, COLORS.norm, 0, {
          type: 'tensor', title: seg.label,
          lines: seg.tensors.map((t) => `${tensorName(t.name, globalIdx)}  ${fmtShape(t.shape)}`),
        });
        prevBottom = y - 0.14;
        y -= 0.28 + GAP * 0.7;
      } else if (seg.kind === 'attn') {
        const mats = seg.tensors.filter((t) => t.kind === 'matrix');
        const rowT = mats.filter((t) => ['q_proj', 'k_proj', 'v_proj', 'qkv', 'q_a_proj', 'q_b_proj', 'kv_a_proj', 'kv_b_proj'].includes(t.role));
        const below = mats.filter((t) => !rowT.includes(t));
        const boxes = rowT.map((t) => ({ t, sx: L(t.shape[1]), sz: L(t.shape[0]) }));
        const totalW = boxes.reduce((a, b) => a + b.sx, 0) + (boxes.length - 1) * 1.0;
        let cx = -totalW / 2;
        const rowH = 0.55;
        this.connector(prevBottom, y + rowH / 2);
        this.labels.push({ x: totalW / 2 + 1.2, y, z: 0, text: seg.label, kind: 'seg' });
        for (const b of boxes) {
          boxG(cx + b.sx / 2, y, 0, b.sx, rowH, b.sz, roleColor(b.t.role), 0, {
            type: 'tensor', title: `${seg.label} · ${b.t.role}`,
            lines: [
              `${tensorName(b.t.name, globalIdx)}`,
              `${fmtShape(b.t.shape)}  ${fmtParams(b.t.params)}`,
              seg.meta.variant === 'MLA'
                ? `MLA: kv_lora ${seg.meta.kvLora}${seg.meta.qLora ? ' · q_lora ' + seg.meta.qLora : ''}`
                : `${seg.meta.variant}: ${seg.meta.heads} heads / ${seg.meta.kvHeads} KV · head_dim ${seg.meta.headDim}`,
            ],
          });
          cx += b.sx + 1.0;
        }
        prevBottom = y - rowH / 2;
        y -= rowH + GAP * 0.7;
        for (const t of below) {
          const sx = L(t.shape[1]), sz = L(t.shape[0]);
          this.connector(prevBottom, y + 0.275);
          boxG(0, y, 0, sx, 0.55, sz, roleColor(t.role), 0, {
            type: 'tensor', title: `${seg.label} · ${t.role}`,
            lines: [`${tensorName(t.name, globalIdx)}`, `${fmtShape(t.shape)}  ${fmtParams(t.params)}`],
          });
          prevBottom = y - 0.275;
          y -= 0.55 + GAP * 0.7;
        }
      } else if (seg.kind === 'mlp') {
        const mats = seg.tensors.filter((t) => t.kind === 'matrix');
        const up = mats.filter((t) => t.shape[0] >= t.shape[1] || ['gate_proj', 'up_proj', 'c_fc', 'w1', 'w3'].includes(t.role));
        const down = mats.filter((t) => !up.includes(t));
        const boxes = up.map((t) => ({ t, sx: L(t.shape[1]), sz: L(t.shape[0]) }));
        const totalW = boxes.reduce((a, b) => a + b.sx, 0) + (boxes.length - 1) * 1.0;
        let cx = -totalW / 2;
        this.connector(prevBottom, y + 0.275);
        this.labels.push({ x: totalW / 2 + 1.2, y, z: 0, text: seg.label, kind: 'seg' });
        for (const b of boxes) {
          boxG(cx + b.sx / 2, y, 0, b.sx, 0.55, b.sz, COLORS.mlp, 0, {
            type: 'tensor', title: `${seg.label} · ${b.t.role}`,
            lines: [`${tensorName(b.t.name, globalIdx)}`, `${fmtShape(b.t.shape)}  ${fmtParams(b.t.params)}`],
          });
          cx += b.sx + 1.0;
        }
        prevBottom = y - 0.275;
        y -= 0.55 + GAP * 0.7;
        for (const t of down) {
          this.connector(prevBottom, y + 0.275);
          boxG(0, y, 0, L(t.shape[1]), 0.55, L(t.shape[0]), COLORS.mlp, 0, {
            type: 'tensor', title: `${seg.label} · ${t.role}`,
            lines: [`${tensorName(t.name, globalIdx)}`, `${fmtShape(t.shape)}  ${fmtParams(t.params)}`],
          });
          prevBottom = y - 0.275;
          y -= 0.55 + GAP * 0.7;
        }
      } else if (seg.kind === 'moe') {
        y = this._moe(graph, seg, si, li, globalIdx, y, prevBottom, state, boxG);
        prevBottom = this._lastBottom;
      }
    }
    return { bottom: prevBottom, nextY: y - LAYER_GAP * 0.5 };
  }

  _moe(graph, seg, si, li, globalIdx, y, prevBottom, state, boxG) {
    const meta = graph.meta;
    const m = seg.meta;
    // router
    const rt = seg.routerTensors[0];
    this.connector(prevBottom, y + 0.25);
    boxG(0, y, 0, L(rt.shape[1]) * 0.8, 0.5, L(rt.shape[0]) * 0.8, COLORS.router, 0, {
      type: 'tensor', title: `Router · top-${m.topK} / ${m.experts}`,
      lines: [
        `${tensorName(rt.name, globalIdx)}  ${fmtShape(rt.shape)}`,
        `softmax → top-${m.topK}${m.normTopK ? ' → renormalize' : ''}`,
      ],
    });
    this.labels.push({ x: L(rt.shape[1]) * 0.4 + 1.2, y, z: 0, text: `Router top-${m.topK}/${m.experts}`, kind: 'seg' });
    prevBottom = y - 0.25;
    y -= 0.5 + GAP * 0.7;

    const expParams = seg.expertProto.tensors.reduce((a, t) => a + t.params, 0);
    const key = si + ':' + li;
    const expanded = state.expandedExperts.has(key);

    if (!expanded) {
      // aggregate slab
      const sx = L(meta.hidden) * 1.5, sz = L(m.expertInter) * 1.35;
      const h = Math.min(1.0 + m.experts * 0.012, 4.5);
      this.connector(prevBottom, y + h / 2);
      boxG(0, y, 0, sx, h, sz, COLORS.expertAgg, 2, {
        type: 'expertAgg', si, li,
        title: `专家阵列 × ${m.experts}`,
        lines: [
          `每专家 SwiGLU:${fmtParams(expParams)}`,
          `合计 ${fmtParams(expParams * m.experts)} · 每 token 激活 ${m.topK} 个(${(m.topK / m.experts * 100).toFixed(1)}%)`,
          '点击展开专家网格',
        ],
        action: { kind: 'expandExperts', si, li },
      });
      this.labels.push({ x: sx / 2 + 1.2, y, z: 0, text: `Experts ×${m.experts}`, kind: 'block' });
      prevBottom = y - h / 2;
      y -= h + GAP * 0.7;
    } else {
      // expert grid
      const E = m.experts;
      const cols = Math.ceil(Math.sqrt(E * 1.6));
      const rows = Math.ceil(E / cols);
      const ew = Math.max(0.9, L(meta.hidden) * 0.22);
      const ed = Math.max(0.7, L(m.expertInter) * 0.28);
      const gx = ew * 1.35, gz = ed * 1.6;
      const y0 = y - 0.4;
      this.connector(prevBottom, y0 + 0.5);
      this.labels.push({ x: (cols * gx) / 2 + 1.4, y: y0, z: 0, text: `Experts ×${E}(点击标签折叠)`, kind: 'layer', action: { kind: 'collapseExperts', si, li } });
      for (let e = 0; e < E; e++) {
        const r = Math.floor(e / cols), c = e % cols;
        const x = (c - (cols - 1) / 2) * gx;
        const z = (r - (rows - 1) / 2) * gz;
        boxG(x, y0, z, ew, 0.5, ed, COLORS.expert, 0, {
          type: 'expert', title: `Expert ${e}`,
          lines: [
            ...seg.expertProto.tensors.map((t) => `${tensorName(t.name, globalIdx, e)}  ${fmtShape(t.shape)}`),
            `参数 ${fmtParams(expParams)}`,
          ],
        });
      }
      prevBottom = y0 - 0.25;
      y = y0 - 0.5 - GAP * 0.7;
    }

    if (seg.sharedProto) {
      const sp = seg.sharedProto;
      const spParams = sp.tensors.reduce((a, t) => a + t.params, 0);
      const sx = L(meta.hidden) * 0.8, sz = L(m.sharedInter * (sp.count || 1)) * 0.8;
      this.connector(prevBottom, y + 0.25);
      boxG(0, y, 0, sx, 0.5, sz, COLORS.shared, 0, {
        type: 'tensor', title: `共享专家 × ${sp.count}`,
        lines: sp.tensors.map((t) => `${tensorName(t.name, globalIdx)}  ${fmtShape(t.shape)}`).concat([`参数 ${fmtParams(spParams)}(每 token 恒定激活)`]),
      });
      prevBottom = y - 0.25;
      y -= 0.5 + GAP * 0.7;
    }

    // combine node
    this.connector(prevBottom, y + 0.25);
    boxG(0, y, 0, 1.6, 0.5, 1.0, COLORS.router, 0, {
      type: 'add', title: 'Σ 加权合并', lines: [`top-${m.topK} 专家输出加权求和`],
    });
    this._lastBottom = y - 0.25;
    return y - 0.5 - GAP * 0.7;
  }
}
