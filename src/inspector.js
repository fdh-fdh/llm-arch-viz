// Inspector (FR-D): real-time info panel for whatever the pointer touches.
// renderInspector(sel, graph) -> HTML. Interaction is delegated in main.js via
// data-nav / data-kv attributes.

import { fmtParams, fmtShape, kvCacheBytesPerToken, flopsPerToken, TIER_INFO } from './parser/ir.js';
import { getKnowledge } from './knowledge.js';
import { formulaHTML } from './formula.js';

function esc(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function fmtBytes(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + ' GB';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + ' MB';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + ' KB';
  return n + ' B';
}

function knowledgeBlocks(keys, { compact = false } = {}) {
  const out = [];
  for (const key of keys || []) {
    const k = getKnowledge(key);
    if (!k) continue;
    out.push(`
      <div class="kn">
        <div class="kn-t">${esc(k.title)}</div>
        ${k.formula ? formulaHTML(k.formula) : ''}
        <div class="kn-i">${esc(k.intuition)}</div>
        ${compact ? '' : `<details class="kn-d"><summary>展开细节</summary><div>${esc(k.details)}</div>
          ${(k.refs || []).map((r) => `<a href="${esc(r.u)}" target="_blank" rel="noopener">${esc(r.t)}</a>`).join(' · ')}
        </details>`}
      </div>`);
  }
  return out.join('');
}

function breadcrumb(parts) {
  return `<div class="crumb">${parts.map((p, i) =>
    p.nav
      ? `<button class="crumb-b" data-nav="${esc(p.nav)}">${esc(p.label)}</button>`
      : `<span class="crumb-s${i === parts.length - 1 ? ' cur' : ''}">${esc(p.label)}</span>`
  ).join('<span class="crumb-sep">›</span>')}</div>`;
}

function layerNav(item, graph) {
  const total = graph.stacks[item.si]?.count ?? 0;
  const prev = item.li > 0 ? `${item.si}:${item.li - 1}` : (item.si > 0 ? `${item.si - 1}:${graph.stacks[item.si - 1].count - 1}` : null);
  const next = item.li < total - 1 ? `${item.si}:${item.li + 1}` : (item.si < graph.stacks.length - 1 ? `${item.si + 1}:0` : null);
  return `<div class="ins-nav">
    ${prev ? `<button data-nav="layer:${prev}">◂ 上一层</button>` : '<span></span>'}
    ${next ? `<button data-nav="layer:${next}">下一层 ▸</button>` : '<span></span>'}
  </div>`;
}

function kvCacheWidget(graph, ctx) {
  const perTok = kvCacheBytesPerToken(graph);
  const maxCtx = graph.meta.ctx || 32768;
  const cur = Math.min(ctx || Math.min(8192, maxCtx), maxCtx);
  return `<div class="ins-kv">
    <div class="ins-kv-row"><span>KV cache @ ctx <b id="kvCtxVal">${cur.toLocaleString('en-US')}</b></span>
      <b id="kvBytesVal">${fmtBytes(perTok * cur)}</b></div>
    <input type="range" min="128" max="${maxCtx}" step="128" value="${cur}" data-kv="${perTok}">
    <div class="ins-note">bf16 · 每 token ${fmtBytes(perTok)} · 全部层合计(估算)</div>
  </div>`;
}

// sel: null (model overview) | layout item | {element: {item, row, col}}
export function renderInspector(sel, graph, opts = {}) {
  if (!graph) return '<div class="ins-empty">加载一个模型开始</div>';
  const m = graph.meta;

  // ---------- element sub-selection ----------
  if (sel && sel.element) {
    const { item, row, col } = sel.element;
    const e = item.elem;
    const parts = [
      { label: m.name, nav: 'model' },
      item.globalIdx !== undefined ? { label: `Layer ${item.globalIdx}`, nav: `layer:${item.si}:${item.li}` } : null,
      item.segLabel ? { label: item.segLabel } : null,
      { label: `W[${row}, ${col}]` },
    ].filter(Boolean);
    return `
      ${breadcrumb(parts)}
      <div class="ins-h">元素 <span class="mono">W[${row}, ${col}]</span></div>
      <div class="ins-meta mono">${esc(e.name)}</div>
      <div class="ins-body">
        <div class="kn"><div class="kn-i">
          第 <b>${row}</b> ${esc(e.rowSem || '行')} × 第 <b>${col}</b> ${esc(e.colSem || '列')} 的连接权重。
          ${e.rows > 1 ? `参与计算:输出的第 ${row} 维 = Σ<sub>k</sub> 输入[k] · W[${row}, k],此元素贡献 k=${col} 那一项。` : `对输入第 ${col} 维做逐元素缩放。`}
        </div></div>
        ${knowledgeBlocks((sel.element.item.kb || []).slice(0, 1), { compact: true })}
      </div>`;
  }

  // ---------- model overview (default) ----------
  if (!sel) {
    const tier = TIER_INFO[m.tier];
    const segs = graph.stacks[0]?.segments.filter((s) => s.kind !== 'add') || [];
    return `
      ${breadcrumb([{ label: m.name }])}
      <div class="ins-h">架构总览</div>
      <div class="ins-body">
        <div class="kn"><div class="kn-i">
          ${esc(m.name)} 是 <b>${esc(m.modelType)}</b> 家族的 decoder-only Transformer:
          ${m.layers} 层 · hidden ${m.hidden} · ${m.maxExperts ? `MoE ${m.maxExperts} 专家(top-${graph.stacks.find(s=>s.segments.some(g=>g.kind==='moe'))?.segments.find(g=>g.kind==='moe')?.meta.topK ?? '?'})` : `${m.heads} heads / ${m.kvHeads} KV`}
          · 总参 ${fmtParams(m.params.total)},每 token 激活 ${fmtParams(m.params.active)}。${esc(tier.label)}:${esc(tier.desc)}。
        </div></div>
        <div class="ins-sub">每层管线(悬停 3D 对象或点击下列组件)</div>
        <div class="ins-list">
          ${segs.map((s) => `<button class="ins-li" data-nav="seg:0:0:${esc(s.kind)}:${esc(s.label)}">
            <span>${esc(s.label)}</span><span class="mono">${fmtParams(s.paramInfo?.total ?? 0)}</span>
          </button>`).join('')}
        </div>
        <div class="ins-kpis">
          <div><span>每 token FLOPs</span><b>${fmtParams(flopsPerToken(graph))}(估算)</b></div>
          <div><span>KV cache / token</span><b>${fmtBytes(kvCacheBytesPerToken(graph))}</b></div>
        </div>
        ${kvCacheWidget(graph, opts.kvCtx)}
      </div>`;
  }

  // ---------- layout item ----------
  const parts = [{ label: m.name, nav: 'model' }];
  if (sel.globalIdx !== undefined && sel.si !== undefined) {
    parts.push({ label: `Layer ${sel.globalIdx}`, nav: `layer:${sel.si}:${sel.li}` });
  }
  if (sel.segLabel) parts.push({ label: sel.segLabel });
  if (sel.role) parts.push({ label: sel.role });
  else if (!sel.segLabel) parts.push({ label: sel.title || sel.type });

  let metaLine = '';
  if (sel.elem && sel.elem.rows > 1) {
    const params = sel.elem.rows * sel.elem.cols;
    metaLine = `<div class="ins-meta mono">${esc(sel.elem.name)}<br>[${sel.elem.rows.toLocaleString('en-US')} × ${sel.elem.cols.toLocaleString('en-US')}] · ${fmtParams(params)} · ${esc(m.dtype)} ≈ ${fmtBytes(params * 2)}</div>`;
  } else if (sel.lines?.length) {
    metaLine = `<div class="ins-meta mono">${sel.lines.map(esc).join('<br>')}</div>`;
  }

  let extra = '';
  if (sel.segMeta && (sel.kb || []).some((k) => ['gqa', 'mqa', 'mla', 'attention_core'].includes(k))) {
    const sm = sel.segMeta;
    if (sm.heads) {
      extra += `<div class="ins-gqa">${gqaDiagram(sm.heads, sm.kvHeads)}</div>`;
    }
    extra += kvCacheWidget(graph, opts.kvCtx);
  }
  if (sel.type === 'stackAgg' && sel.stackRef) {
    extra += `<div class="ins-sub">层内管线</div><div class="ins-list">${
      sel.stackRef.segments.filter((s) => s.kind !== 'add').map((s) =>
        `<button class="ins-li" data-nav="seg:${sel.si}:${sel.liStart}:${esc(s.kind)}:${esc(s.label)}">
          <span>${esc(s.label)}</span><span class="mono">${fmtParams(s.paramInfo?.total ?? 0)}</span></button>`).join('')
    }</div>`;
  }

  const hint = sel.elem && sel.elem.rows > 1
    ? '<div class="ins-note">💡 贴近这块矩阵可以看到元素格;悬停任意格子查看 W[row, col]</div>' : '';

  return `
    ${breadcrumb(parts)}
    <div class="ins-h">${esc(sel.title || sel.type)}</div>
    ${metaLine}
    <div class="ins-body">
      ${knowledgeBlocks(sel.kb)}
      ${extra}
      ${hint}
      ${sel.globalIdx !== undefined && sel.si !== undefined ? layerNav(sel, graph) : ''}
    </div>`;
}

// Mini GQA grouping diagram (FR-D4): Q heads on top, shared KV groups below.
function gqaDiagram(heads, kvHeads) {
  const show = Math.min(heads, 16);
  const ratio = heads / kvHeads;
  const shownKV = Math.max(1, Math.round(show / ratio));
  const w = 232, qw = w / show - 2, kw = w / shownKV - 3;
  let svg = `<svg viewBox="0 0 ${w} 46" width="100%" height="46">`;
  for (let i = 0; i < show; i++) {
    const g = Math.floor(i / ratio) % 6;
    svg += `<rect x="${i * (qw + 2)}" y="0" width="${qw}" height="12" rx="2" fill="hsl(${262 - g * 24},62%,${62 - g * 3}%)"/>`;
  }
  for (let j = 0; j < shownKV; j++) {
    svg += `<rect x="${j * (kw + 3)}" y="30" width="${kw}" height="12" rx="2" fill="hsl(${262 - (j % 6) * 24},45%,72%)"/>`;
    const x1 = j * (kw + 3) + kw / 2;
    svg += `<line x1="${x1}" y1="30" x2="${x1}" y2="16" stroke="#9aa5b8" stroke-width="1"/>`;
  }
  svg += `</svg><div class="ins-note">${heads} 个 Q 头${heads !== kvHeads ? ` → 每 ${ratio} 个共享一组 K/V(共 ${kvHeads} 组)` : ',每头独享 K/V (MHA)'}${heads > 16 ? ' · 图示前 16 头' : ''}</div>`;
  return svg;
}
