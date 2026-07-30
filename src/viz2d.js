// 2D SVG renderer — the memory-friendly fallback view (mobile default) and the
// vector export path. Same IR, different renderer.

import { fmtParams, fmtShape, tensorName, TIER_INFO } from './parser/ir.js';

const C = {
  io: '#8892a6', embedding: '#5c6ee0', norm: '#eead40', attn: '#9e69ed',
  mlp: '#21a08c', moe: '#0f7a5c', expert: '#2eae85', shared: '#1aa6ae',
  stack: '#6b84b8', lmHead: '#4d5ccc', text: '#1a2233', dim: '#5c6a82',
};

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

export function buildSVG(graph, { width = 760 } = {}) {
  const meta = graph.meta;
  const cx = 270;
  const boxW = 300;
  const rows = [];

  rows.push({ kind: 'io', label: 'input_ids', sub: `[B, T] · vocab ${meta.vocab.toLocaleString('en-US')}` });
  rows.push({ kind: 'embedding', label: 'Token Embedding', sub: `${fmtShape(graph.embedding.tensors[0].shape)} · ${fmtParams(meta.params.embedding)}` });

  let globalIdx = 0;
  for (const stack of graph.stacks) {
    const segRows = [];
    for (const seg of stack.segments) {
      if (seg.kind === 'norm') segRows.push({ kind: 'norm', label: seg.label, sub: `RMSNorm(${meta.hidden})` });
      else if (seg.kind === 'attn') {
        const mats = seg.tensors.filter((t) => t.kind === 'matrix');
        segRows.push({
          kind: 'attn', label: seg.label,
          sub: seg.meta.variant === 'MLA'
            ? `kv_lora ${seg.meta.kvLora} · ${seg.meta.heads} heads`
            : `${seg.meta.heads} heads / ${seg.meta.kvHeads} KV · head_dim ${seg.meta.headDim}`,
          detail: mats.map((t) => `${t.role} ${fmtShape(t.shape)}`),
        });
      } else if (seg.kind === 'mlp') {
        segRows.push({ kind: 'mlp', label: seg.label, sub: `intermediate ${seg.meta.inter.toLocaleString('en-US')} · ${seg.meta.act}` });
      } else if (seg.kind === 'moe') {
        const ep = seg.expertProto.tensors.reduce((a, t) => a + t.params, 0);
        segRows.push({
          kind: 'moe', label: seg.label,
          sub: `router ${meta.hidden}→${seg.meta.experts} · 每专家 ${fmtParams(ep)}${seg.meta.sharedExperts ? ` · 共享×${seg.meta.sharedExperts}` : ''}`,
          moe: seg.meta,
        });
      } else if (seg.kind === 'add') {
        segRows.push({ kind: 'add', label: '⊕ residual' });
      }
    }
    rows.push({ kind: 'stack', label: `${stack.label} × ${stack.count}`, sub: `每层 ${fmtParams(stack.perLayer)} · 激活 ${fmtParams(stack.perLayerActive)}`, segRows, count: stack.count });
    globalIdx += stack.count;
  }

  rows.push({ kind: 'norm', label: 'Final RMSNorm', sub: `(${meta.hidden})` });
  rows.push({
    kind: 'lmHead',
    label: graph.lmHead.tied ? 'LM Head(tied)' : 'LM Head',
    sub: graph.lmHead.tied ? '与 embedding 共享权重' : `${fmtShape(graph.lmHead.tensors[0].shape)} · ${fmtParams(meta.params.lmHead)}`,
  });
  rows.push({ kind: 'io', label: 'logits', sub: `[B, T, ${meta.vocab.toLocaleString('en-US')}]` });

  // ---- render ----
  let y = 84;
  const parts = [];
  const arrow = (yy, h = 16) => parts.push(`<line x1="${cx}" y1="${yy}" x2="${cx}" y2="${yy + h}" stroke="#9aa5b8" stroke-width="1.6" marker-end="url(#arr)"/>`);

  const block = (row, x, w, indent) => {
    const isAdd = row.kind === 'add';
    const h = isAdd ? 24 : (row.detail ? 44 + row.detail.length * 13 : 44);
    const col = C[row.kind] || '#888';
    if (isAdd) {
      parts.push(`<circle cx="${cx}" cy="${y + 12}" r="10" fill="#fff" stroke="#5c6a82" stroke-width="1.4"/><text x="${cx}" y="${y + 16}" text-anchor="middle" font-size="13" fill="#334">+</text>`);
    } else {
      parts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${col}18" stroke="${col}" stroke-width="1.6"/>`);
      parts.push(`<text x="${x + 12}" y="${y + 19}" font-size="13" font-weight="600" fill="${C.text}">${esc(row.label)}</text>`);
      if (row.sub) parts.push(`<text x="${x + 12}" y="${y + 35}" font-size="11" fill="${C.dim}">${esc(row.sub)}</text>`);
      if (row.detail) row.detail.forEach((d, i) => {
        parts.push(`<text x="${x + 12}" y="${y + 48 + i * 13}" font-size="10" font-family="monospace" fill="${C.dim}">${esc(d)}</text>`);
      });
      if (row.moe) {
        // mini expert dots
        const n = Math.min(row.moe.experts, 24);
        for (let i = 0; i < n; i++) {
          const dx = x + w - 14 - (i % 8) * 10;
          const dy = y + h - 12 - Math.floor(i / 8) * 9;
          parts.push(`<rect x="${dx}" y="${dy}" width="7" height="6" rx="1.5" fill="${i < row.moe.topK ? C.expert : '#c3d4cd'}"/>`);
        }
      }
    }
    y += h;
    return h;
  };

  for (const row of rows) {
    if (row.kind === 'stack') {
      const inner = row.segRows;
      const innerH = inner.reduce((a, r) => a + (r.kind === 'add' ? 24 : (r.detail ? 44 + r.detail.length * 13 : 44)) + 14, 0);
      const frameH = innerH + 46;
      parts.push(`<rect x="70" y="${y}" width="400" height="${frameH}" rx="12" fill="#5c6ee008" stroke="${C.stack}" stroke-width="2"/>`);
      parts.push(`<rect x="86" y="${y - 10}" width="${Math.min(240, row.label.length * 9 + 20)}" height="22" rx="6" fill="${C.stack}"/>`);
      parts.push(`<text x="96" y="${y + 5}" font-size="12" font-weight="700" fill="#fff">${esc(row.label)}</text>`);
      parts.push(`<text x="${462}" y="${y + 20}" font-size="10.5" text-anchor="end" fill="${C.dim}">${esc(row.sub)}</text>`);
      y += 34;
      for (const r of inner) {
        block(r, 100, 340, true);
        arrow(y, 14); y += 14;
      }
      y -= 14; // no trailing arrow inside frame
      y = Math.max(y, 0);
      y += 24;
      arrow(y, 16); y += 16;
    } else {
      block(row, cx - 150, boxW);
      arrow(y, 16); y += 16;
    }
  }
  y -= 16;

  const H = y + 40;
  const W = width;
  const tier = TIER_INFO[meta.tier];
  const header = `
    <text x="24" y="34" font-size="19" font-weight="700" fill="${C.text}">${esc(meta.name)}</text>
    <text x="24" y="56" font-size="12" fill="${C.dim}">model_type: ${esc(meta.modelType)} · ${meta.layers} 层 · hidden ${meta.hidden} · 总参 ${fmtParams(meta.params.total)} · 激活 ${fmtParams(meta.params.active)} · ${esc(tier.label)}${meta.degraded ? ' · ⚠ 通用兜底解析' : ''}</text>`;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="-apple-system,'Segoe UI',Roboto,'PingFang SC',sans-serif">
  <defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#9aa5b8"/></marker></defs>
  <rect width="${W}" height="${H}" fill="#f7f9fc"/>
  ${header}
  ${parts.join('\n')}
</svg>`;
  return svg;
}
