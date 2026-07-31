// llm-arch-viz — app bootstrap & UI wiring (zero-dependency vanilla ES modules).
// v2: Inspector real-time panel, focus mode, element-cell interaction, fly-to camera.

import { Renderer } from './gl/renderer.js';
import { OrbitCamera } from './gl/camera.js';
import { pick } from './gl/pick.js';
import { mat4ProjectPoint } from './gl/mat4.js';
import { buildGraph, fmtParams, TIER_INFO } from './parser/ir.js';
import { LayoutBuilder } from './layout.js';
import { buildSVG } from './viz2d.js';
import { renderInspector } from './inspector.js';
import { exportLlmarch, parseLlmarch } from './llmarch.js';
import { fetchModel } from './hf.js';
import { downloadBlob, downloadText, exportPoster, exportGLB } from './export.js';
import { SAMPLES } from './samples.js';
import { buildTour, resolveStation } from './tour.js';
import { readSafetensorsFile, inferConfigFromTensors } from './safetensors.js';

const $ = (id) => document.getElementById(id);
const canvas = $('gl');
const renderer = new Renderer(canvas);
const camera = new OrbitCamera(canvas, () => requestRender());
const builder = new LayoutBuilder();

const state = {
  graph: null,
  layout: null,
  viewMode: window.matchMedia('(max-width: 760px)').matches ? '2d' : '3d',
  expanded: new Set(),
  expandedExperts: new Set(),
  lru: [],
  hoverId: -1,
  hoverCell: null,          // [col,row] in shader-grid coords
  anim: { on: false, raf: 0, t: 0, last: 0 },
  labelPool: [],
  // selection (FR-D)
  hoverSel: null,           // transient item under pointer
  pinned: null,             // resolved pinned item (re-resolved after rebuild)
  pinnedDesc: null,         // identity descriptor surviving rebuilds
  focus: false,
  kvCtx: null,
  cellMode: 1,              // FR-C5: 0 off / 1 auto / 2 boost
};

// ---------------------------------------------------------------------------
// Demand rendering
// ---------------------------------------------------------------------------
let renderQueued = false;
function requestRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    drawFrame();
  });
}

function pulseRange() {
  const lay = state.layout;
  if (!lay) return null;
  const tier = state.graph.meta.tier;
  if (tier === 'T2') {
    const last = state.lru[state.lru.length - 1];
    const r = (lay.expandedRanges || []).find((x) => x.key === last);
    return r ? [r.top, r.bottom] : null;
  }
  return [lay.spineTop, lay.spineBottom];
}

function drawFrame() {
  if (state.viewMode !== '3d') return;
  let pulseY = -1e9;
  if (state.anim.on) {
    const range = pulseRange();
    if (range) pulseY = range[0] + (range[1] - range[0]) * state.anim.t;
  }
  renderer.render(camera, {
    hoverId: state.hoverId, pulseY, focus: state.focus, hoverCell: state.hoverCell,
    cellMode: state.cellMode,
  });
  positionLabels();
}

function animLoop(ts) {
  if (!state.anim.on) return;
  if (!state.anim.last) state.anim.last = ts;
  const dt = (ts - state.anim.last) / 1000;
  state.anim.last = ts;
  state.anim.t = (state.anim.t + dt / 3.2) % 1;
  drawFrame();
  state.anim.raf = requestAnimationFrame(animLoop);
}

// ---------------------------------------------------------------------------
// Labels overlay
// ---------------------------------------------------------------------------
function positionLabels() {
  const lay = state.layout;
  const wrap = $('labels');
  if (!lay) return;
  const W = wrap.clientWidth, H = wrap.clientHeight;
  const crowded = lay.labels.length > 80;
  lay.labels.forEach((lb, i) => {
    if (crowded && (lb.kind === 'seg' || lb.kind === 'minor')) {
      const el0 = state.labelPool[i];
      if (el0) el0.style.display = 'none';
      return;
    }
    let el = state.labelPool[i];
    if (!el) {
      el = document.createElement('div');
      wrap.appendChild(el);
      state.labelPool[i] = el;
    }
    const [x, y, , w] = mat4ProjectPoint(camera.viewProj, lb.x, lb.y, lb.z);
    if (w <= 0 || x < -1.1 || x > 1.1 || y < -1.1 || y > 1.1) {
      el.style.display = 'none';
      return;
    }
    el.style.display = 'block';
    el.className = 'lbl ' + (lb.kind || '');
    el.textContent = lb.text;
    el.style.left = ((x + 1) / 2 * W) + 'px';
    el.style.top = ((1 - (y + 1) / 2) * H) + 'px';
    el.onclick = lb.action ? (() => { applyAction(lb.action); }) : null;
  });
  for (let i = lay.labels.length; i < state.labelPool.length; i++) {
    state.labelPool[i].style.display = 'none';
  }
}

// ---------------------------------------------------------------------------
// Selection / Inspector (FR-D)
// ---------------------------------------------------------------------------
function descOf(item) {
  if (!item) return null;
  return {
    type: item.type, title: item.title, si: item.si, li: item.li,
    segLabel: item.segLabel, role: item.role, expertIdx: item.expertIdx,
  };
}

function resolveDesc(desc) {
  if (!desc || !state.layout) return null;
  const items = state.layout.items;
  for (let i = 0; i < state.layout.soa.count; i++) {
    const it = items[i];
    if (!it) continue;
    if (it.type === desc.type && it.si === desc.si && it.li === desc.li &&
        it.segLabel === desc.segLabel && it.role === desc.role &&
        it.expertIdx === desc.expertIdx && it.title === desc.title) return it;
  }
  // fallback: same layer, same segment
  for (let i = 0; i < state.layout.soa.count; i++) {
    const it = items[i];
    if (it && it.si === desc.si && it.li === desc.li && it.segLabel === desc.segLabel) return it;
  }
  return null;
}

let lastInspectorKey = null;
function updateInspector() {
  const sel = state.hoverSel || state.pinned;
  const key = sel ? JSON.stringify(descOf(sel.element ? sel.element.item : sel)) + (sel.element ? `:${sel.element.row}:${sel.element.col}` : '') : 'model';
  if (key === lastInspectorKey) return;
  lastInspectorKey = key;
  $('inspector').innerHTML = renderInspector(sel, state.graph, { kvCtx: state.kvCtx });
  $('insPin').textContent = state.pinned ? '📌' : '';
}

function pinItem(item, element = null) {
  state.pinned = element ? { element } : item;
  state.pinnedDesc = descOf(element ? element.item : item);
  state.pinnedElement = element ? { row: element.row, col: element.col } : null;
  applyFocusDim(element ? element.item : item);
  lastInspectorKey = null;
  updateInspector();
}

function clearPin() {
  state.pinned = null;
  state.pinnedDesc = null;
  state.pinnedElement = null;
  state.focus = false;
  if (state.layout) {
    state.layout.soa.dim.fill(0);
    renderer.setDim(state.layout.soa.dim);
  }
  lastInspectorKey = null;
  updateInspector();
  requestRender();
}

// Focus mode: dim everything outside the selected item's layer (FR-C6).
function applyFocusDim(item) {
  const lay = state.layout;
  if (!lay || !item) return;
  const dim = lay.soa.dim;
  const sameLayer = item.si !== undefined && item.li !== undefined;
  for (let i = 0; i < lay.soa.count; i++) {
    const it = lay.items[i];
    if (lay.soa.flag[i] === 1) { dim[i] = 0; continue; }        // keep spine
    if (!it) { dim[i] = 1; continue; }
    if (sameLayer) dim[i] = (it.si === item.si && it.li === item.li) ? 0 : 1;
    else dim[i] = (it === item) ? 0 : 1;
  }
  state.focus = true;
  renderer.setDim(dim);
  requestRender();
}

function flyToItem(item) {
  if (!item) return;
  const size = Math.max(item.sx || 4, item.sz || 4, (item.sy || 1));
  camera.flyTo({ target: [item.x, item.y, item.z], dist: Math.max(8, size * 2.4 + 5) });
}

// Shared nav handler (Inspector breadcrumb/lists + 2D SVG blocks, FR-G2).
function findItem(pred) {
  for (let i = 0; i < state.layout.soa.count; i++) {
    const it = state.layout.items[i];
    if (it && pred(it)) return it;
  }
  return null;
}
function handleNav(nav) {
  if (!nav || !state.layout) return;
  const is3d = state.viewMode === '3d';
  if (nav === 'model') { clearPin(); return; }
  if (nav.startsWith('path:')) {
    const key = nav.slice(5);
    const target = findItem((it) => (it.path && it.path[0] === key) || (it.type === 'io' && it.title === key));
    if (target) { pinItem(target); if (is3d) flyToItem(target); }
    return;
  }
  if (nav.startsWith('layer:')) {
    const [, si, li] = nav.split(':').map(Number);
    expandLayer(si, li);
    const target = findItem((it) => it.si === si && it.li === li && it.type === 'tensor');
    if (target) { pinItem(target); if (is3d) flyToLayer(si, li); }
    return;
  }
  if (nav.startsWith('seg:')) {
    const [, siS, liS, , ...labelParts] = nav.split(':');
    const si = Number(siS), li = Number(liS);
    const label = labelParts.join(':');
    expandLayer(si, li);
    const target = findItem((it) => it.si === si && it.li === li && it.segLabel === label);
    if (target) { pinItem(target); if (is3d) flyToItem(target); }
  }
}
$('inspector').addEventListener('click', (e) => {
  handleNav(e.target.closest('[data-nav]')?.dataset.nav);
});
$('view2d').addEventListener('click', (e) => {
  handleNav(e.target.closest('[data-nav]')?.dataset.nav);
});

// ---------------------------------------------------------------------------
// Guided tour (FR-F1)
// ---------------------------------------------------------------------------
const tourState = { active: false, stations: [], i: 0 };
function startTour() {
  if (!state.graph) return;
  tourState.stations = buildTour(state.graph);
  tourState.active = true;
  tourState.i = -1;
  $('tourBar').hidden = false;
  $('btnTour').textContent = '🎬 游览中';
  gotoStation(0);
}
function endTour() {
  tourState.active = false;
  $('tourBar').hidden = true;
  $('btnTour').textContent = '🎬 游览';
  clearPin();
}
function gotoStation(i) {
  const N = tourState.stations.length;
  if (i < 0) return;
  if (i >= N) { endTour(); status('游览结束 🎉 — 自由探索吧,悬停任何组件都有讲解'); return; }
  tourState.i = i;
  const st = tourState.stations[i];
  if (st.expand) expandLayer(st.expand.si, st.expand.li);
  const item = resolveStation(st, state.layout);
  if (item) {
    pinItem(item);
    if (state.viewMode === '3d') flyToItem(item);
  }
  $('tourStep').textContent = `${i + 1}/${N}`;
  $('tourLabel').textContent = st.label;
  $('tourCaption').textContent = st.caption;
  $('tourPrev').disabled = i === 0;
  $('tourNext').textContent = i === N - 1 ? '完成 ✓' : '继续 ▸';
}
$('btnTour').onclick = () => (tourState.active ? endTour() : startTour());
$('tourNext').onclick = () => gotoStation(tourState.i + 1);
$('tourPrev').onclick = () => gotoStation(tourState.i - 1);
$('tourExit').onclick = endTour;

// ---------------------------------------------------------------------------
// Tensor search (FR-G3)
// ---------------------------------------------------------------------------
function searchModel(query) {
  const g = state.graph;
  if (!g || !query.trim()) return [];
  const tokens = query.trim().toLowerCase().split(/[\s./_]+/).filter(Boolean);
  const numTok = tokens.find((t) => /^\d+$/.test(t));
  const textToks = tokens.filter((t) => !/^\d+$/.test(t));
  const gIdx = numTok ? Number(numTok) : null;
  const results = [];
  let base = 0;
  g.stacks.forEach((stack, si) => {
    let li = 0;
    if (gIdx !== null && gIdx >= base && gIdx < base + stack.count) li = gIdx - base;
    else if (gIdx !== null && (gIdx < base || gIdx >= base + stack.count)) { base += stack.count; return; }
    const layerIdx = base + li;
    for (const seg of stack.segments) {
      const cands = [
        ...(seg.tensors || []),
        ...(seg.routerTensors || []),
        ...(seg.expertProto ? seg.expertProto.tensors : []),
      ];
      for (const t of cands) {
        const name = t.name.replace('{i}', String(layerIdx)).replace('{j}', '0');
        const lower = name.toLowerCase();
        if (textToks.every((tok) => lower.includes(tok))) {
          results.push({ name, si, li, segLabel: seg.label, role: t.role, meta: `Layer ${layerIdx} · ${seg.label}` });
        }
      }
    }
    base += stack.count;
  });
  // global tensors
  const globals = [
    { name: g.embedding.tensors[0].name, path: 'embed_tokens', meta: 'Embedding' },
    { name: 'lm_head.weight', path: 'lm_head', meta: 'LM Head' },
  ];
  for (const t of globals) {
    if (textToks.length && textToks.every((tok) => t.name.toLowerCase().includes(tok))) {
      results.push({ name: t.name, path: t.path, meta: t.meta });
    }
  }
  return results.slice(0, 8);
}
function jumpToResult(r) {
  $('searchResults').hidden = true;
  $('searchInput').value = '';
  if (r.path) { handleNav('path:' + r.path); return; }
  expandLayer(r.si, r.li);
  const target = findItem((it) => it.si === r.si && it.li === r.li && it.segLabel === r.segLabel &&
    (it.role === r.role || it.role === undefined)) ||
    findItem((it) => it.si === r.si && it.li === r.li && it.segLabel === r.segLabel);
  if (target) { pinItem(target); if (state.viewMode === '3d') flyToItem(target); }
}
$('searchInput').addEventListener('input', () => {
  const res = searchModel($('searchInput').value);
  const box = $('searchResults');
  if (!res.length) { box.hidden = true; return; }
  box.hidden = false;
  box.innerHTML = res.map((r, i) =>
    `<button class="sr-item" data-sr="${i}">${r.name}<br><span class="sr-meta">${r.meta}</span></button>`).join('');
  box.querySelectorAll('[data-sr]').forEach((el) => {
    el.onclick = () => jumpToResult(res[Number(el.dataset.sr)]);
  });
});
$('searchInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    const res = searchModel($('searchInput').value);
    if (res.length) jumpToResult(res[0]);
  }
  if (e.key === 'Escape') { $('searchResults').hidden = true; $('searchInput').blur(); }
});
document.addEventListener('click', (e) => {
  if (!e.target.closest('.search-box')) $('searchResults').hidden = true;
});
$('inspector').addEventListener('input', (e) => {
  const t = e.target;
  if (t.dataset?.kv) {
    const perTok = Number(t.dataset.kv);
    const ctx = Number(t.value);
    state.kvCtx = ctx;
    const b = perTok * ctx;
    const fmtB = b >= 1e9 ? (b / 1e9).toFixed(2) + ' GB' : (b / 1e6).toFixed(1) + ' MB';
    const cv = $('kvCtxVal'), bv = $('kvBytesVal');
    if (cv) cv.textContent = ctx.toLocaleString('en-US');
    if (bv) bv.textContent = fmtB;
  }
});

window.addEventListener('keydown', (e) => {
  const typing = ['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName);
  if (tourState.active && !typing) {
    if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); gotoStation(tourState.i + 1); return; }
    if (e.key === 'ArrowLeft') { e.preventDefault(); gotoStation(tourState.i - 1); return; }
    if (e.key === 'Escape') { endTour(); return; }
  }
  if (e.key === 'Escape') {
    if (state.pinned) clearPin();
    $('modal').classList.remove('open');
  }
  if (e.key === '/' && !typing) { e.preventDefault(); $('searchInput').focus(); }
});

// ---------------------------------------------------------------------------
// Layout / state rebuild
// ---------------------------------------------------------------------------
function rebuild(fit = false) {
  if (!state.graph) return;
  state.layout = builder.build(state.graph, state);
  state.layout.expandedRanges = builder.expandedRanges || [];
  renderer.setInstances(state.layout.soa);
  state.hoverId = -1;
  state.hoverCell = null;
  state.hoverSel = null;
  // re-resolve pinned selection on the new layout
  if (state.pinnedDesc) {
    const it = resolveDesc(state.pinnedDesc);
    if (it) {
      state.pinned = state.pinnedElement ? { element: { item: it, ...state.pinnedElement } } : it;
      applyFocusDim(it);
    } else {
      clearPin();
    }
  }
  if (fit) camera.fit(state.layout.bounds);
  if (state.viewMode === '2d') render2D();
  lastInspectorKey = null;
  updateInspector();
  requestRender();
}

function tierMax() {
  return TIER_INFO[state.graph.meta.tier].maxExpand;
}

function expandLayer(si, li) {
  const key = si + ':' + li;
  if (state.expanded.has(key)) return;
  const max = tierMax();
  while (state.expanded.size >= max && state.lru.length) {
    const old = state.lru.shift();
    state.expanded.delete(old);
    state.expandedExperts.delete(old);
  }
  state.expanded.add(key);
  state.lru.push(key);
  rebuild();
}

function collapseLayer(si, li) {
  const key = si + ':' + li;
  state.expanded.delete(key);
  state.expandedExperts.delete(key);
  state.lru = state.lru.filter((k) => k !== key);
  rebuild();
}

function flyToLayer(si, li) {
  const r = (state.layout.expandedRanges || []).find((x) => x.key === si + ':' + li);
  if (r) camera.flyTo({ target: [0, (r.top + r.bottom) / 2, 0], dist: Math.max(14, (r.top - r.bottom) * 1.1) });
}

function applyAction(action) {
  if (!action) return;
  if (action.kind === 'expandLayer') { expandLayer(action.si, action.li); flyToLayer(action.si, action.li); }
  else if (action.kind === 'collapseLayer') collapseLayer(action.si, action.li);
  else if (action.kind === 'expandExperts') { state.expandedExperts.add(action.si + ':' + action.li); rebuild(); }
  else if (action.kind === 'collapseExperts') { state.expandedExperts.delete(action.si + ':' + action.li); rebuild(); }
}

// ---------------------------------------------------------------------------
// Hover / click picking (element-aware, FR-C4)
// ---------------------------------------------------------------------------
let lastPickTime = 0;
let lastHoverUV = null;
canvas.addEventListener('pointermove', (e) => {
  if (!state.layout || e.buttons) return;
  const now = performance.now();
  if (now - lastPickTime < 33) return;
  lastPickTime = now;
  const hit = pick(camera, canvas, e.clientX, e.clientY, state.layout.soa);
  const idx = hit.index;
  const item = idx >= 0 ? state.layout.items[idx] : null;
  const tip = $('tooltip');
  if (item) {
    let changed = state.hoverId !== idx;
    state.hoverId = idx;
    state.hoverSel = item;
    lastHoverUV = { u: hit.u, v: hit.v };
    // element cell under cursor
    let elemLine = '';
    if (item.elem && item.elem.rows > 1) {
      const row = Math.min(item.elem.rows - 1, Math.floor(hit.v * item.elem.rows));
      const col = Math.min(item.elem.cols - 1, Math.floor(hit.u * item.elem.cols));
      const gc = [Math.floor(hit.u * Math.min(item.elem.cols, 512)), Math.floor(hit.v * Math.min(item.elem.rows, 512))];
      if (!state.hoverCell || state.hoverCell[0] !== gc[0] || state.hoverCell[1] !== gc[1]) changed = true;
      state.hoverCell = gc;
      state.hoverElem = { row, col };
      elemLine = `<div class="tl hl">W[${row}, ${col}] · 第${row} ${item.elem.rowSem || '行'} × 第${col} ${item.elem.colSem || '列'}</div>`;
    } else {
      if (state.hoverCell) changed = true;
      state.hoverCell = null;
      state.hoverElem = null;
    }
    if (changed) requestRender();
    tip.style.display = 'block';
    tip.innerHTML = `<div class="tt">${escapeHtml(item.title)}</div>` +
      (item.lines || []).map((l) => `<div class="tl">${escapeHtml(l)}</div>`).join('') + elemLine;
    const rect = canvas.getBoundingClientRect();
    const tx = e.clientX - rect.left + 16, ty = e.clientY - rect.top + 14;
    tip.style.left = Math.min(tx, rect.width - 440) + 'px';
    tip.style.top = Math.min(ty, rect.height - 180) + 'px';
    canvas.style.cursor = 'pointer';
    updateInspector();
  } else {
    if (state.hoverId !== -1) { state.hoverId = -1; state.hoverCell = null; requestRender(); }
    state.hoverSel = null;
    tip.style.display = 'none';
    canvas.style.cursor = 'default';
    updateInspector();
  }
});
canvas.addEventListener('pointerleave', () => {
  state.hoverId = -1;
  state.hoverCell = null;
  state.hoverSel = null;
  $('tooltip').style.display = 'none';
  updateInspector();
  requestRender();
});

let downPos = null;
canvas.addEventListener('pointerdown', (e) => { downPos = [e.clientX, e.clientY]; });
canvas.addEventListener('pointerup', (e) => {
  if (!downPos || !state.layout) return;
  const moved = Math.hypot(e.clientX - downPos[0], e.clientY - downPos[1]);
  downPos = null;
  if (moved > 5) return;
  const hit = pick(camera, canvas, e.clientX, e.clientY, state.layout.soa);
  const item = hit.index >= 0 ? state.layout.items[hit.index] : null;
  if (!item) { if (state.pinned) clearPin(); return; }
  if (item.action) { applyAction(item.action); return; }
  // pin: shift+click on a matrix pins the element, click pins the component
  if (e.shiftKey && item.elem && item.elem.rows > 1) {
    const row = Math.min(item.elem.rows - 1, Math.floor(hit.v * item.elem.rows));
    const col = Math.min(item.elem.cols - 1, Math.floor(hit.u * item.elem.cols));
    pinItem(item, { item, row, col });
  } else {
    pinItem(item);
    flyToItem(item);
  }
});

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ---------------------------------------------------------------------------
// Sidebar (model card)
// ---------------------------------------------------------------------------
function updateSidebar() {
  const g = state.graph;
  if (!g) return;
  const m = g.meta;
  $('mName').textContent = m.name;
  $('mSub').textContent = `model_type: ${m.modelType} · 适配器: ${m.matchedAdapter}`;
  const tier = TIER_INFO[m.tier];
  const badges = [
    `<span class="badge">${m.layers} 层</span>`,
    `<span class="badge">hidden ${m.hidden}</span>`,
    m.maxExperts ? `<span class="badge">${m.maxExperts} 专家</span>` : `<span class="badge">${m.heads}H/${m.kvHeads}KV</span>`,
    m.ctx ? `<span class="badge">ctx ${m.ctx.toLocaleString('en-US')}</span>` : '',
    `<span class="badge tier" title="${tier.desc}">${tier.label}</span>`,
    m.degraded ? `<span class="badge warn" title="未知 model_type,使用通用兜底解析">⚠ 兜底解析</span>` : '',
    m.source?.kind === 'estimated' ? `<span class="badge warn">UNVERIFIED 估算</span>` : '',
  ];
  $('mBadges').innerHTML = badges.join('');

  const p = m.params;
  const rows = [
    ['Embedding', p.embedding],
    ...g.stacks.map((s) => [`${s.label} ×${s.count}`, s.perLayer * s.count]),
    ['Final Norm', p.finalNorm],
    ['LM Head' + (p.tied ? '(tied)' : ''), p.lmHead],
  ];
  $('mParams').innerHTML =
    rows.map(([k, v]) => `<tr><td>${k}</td><td>${fmtParams(v)}</td></tr>`).join('') +
    `<tr class="total"><td>总参数</td><td>${fmtParams(p.total)}</td></tr>` +
    `<tr class="total"><td>激活 / token</td><td>${fmtParams(p.active)}</td></tr>`;

  $('cfgPre').textContent = JSON.stringify(g.rawConfig, null, 2);

  const totalLayers = g.stacks.reduce((a, s) => a + s.count, 0);
  const canExpandAll = totalLayers <= tierMax();
  $('btnExpandAll').disabled = !canExpandAll;
  $('btnExpandAll').title = canExpandAll ? '' : `${TIER_INFO[m.tier].label}:最多同时展开 ${tierMax()} 层(${tier.desc})`;
  const animMode = tier.animation;
  $('btnAnim').disabled = animMode === 'none';
  $('btnAnim').title = animMode === 'none' ? 'T3 超大模型为保证流畅采用完全静态渲染'
    : animMode === 'focus' ? 'T2:动画仅作用于最近展开的一层(单层聚焦)' : '数据流动画';
}

// ---------------------------------------------------------------------------
// Load pipeline
// ---------------------------------------------------------------------------
function loadConfig(config, source) {
  const graph = buildGraph(config, source);
  state.graph = graph;
  state.expanded = new Set();
  state.expandedExperts = new Set();
  state.lru = [];
  state.pinned = null;
  state.pinnedDesc = null;
  state.pinnedElement = null;
  state.focus = false;
  state.kvCtx = null;
  stopAnim();
  if (typeof tourState !== 'undefined' && tourState.active) {
    tourState.active = false;
    $('tourBar').hidden = true;
    $('btnTour').textContent = '🎬 游览';
  }
  state.cellMode = graph.meta.tier === 'T3' ? 0 : 1;
  const lodSel = document.getElementById('lodSelect');
  if (lodSel) lodSel.value = String(state.cellMode);
  expandDefault(graph);
  updateSidebar();
  rebuild(true);
  status(`已加载 ${graph.meta.name} — 总参 ${fmtParams(graph.meta.params.total)},激活 ${fmtParams(graph.meta.params.active)}`);
}

function expandDefault(graph) {
  state.expanded.add('0:0');
  state.lru.push('0:0');
  const seg = graph.stacks[0]?.segments.find((s) => s.kind === 'moe');
  if (seg && seg.meta.experts <= 64) state.expandedExperts.add('0:0');
}

function status(msg, ms = 3400) {
  const el = $('status');
  el.textContent = msg;
  el.style.display = 'block';
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.style.display = 'none'; }, ms);
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
$('btnLoad').onclick = async () => {
  const v = $('repoInput').value.trim();
  if (!v) return;
  status('正在从 HuggingFace 获取 config.json …', 20000);
  try {
    const token = $('tokenInput').value.trim() || null;
    const { config, source } = await fetchModel(v, { token });
    loadConfig(config, source);
    history.replaceState(null, '', '?repo=' + encodeURIComponent(source.repoId));
  } catch (e) {
    status('加载失败:' + e.message, 8000);
  }
};
$('repoInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('btnLoad').click(); });

{
  const sel = $('sampleSelect');
  sel.innerHTML = '<option value="">内置示例…</option>' +
    SAMPLES.map((s) => `<option value="${s.id}">${s.name}</option>`).join('');
  sel.onchange = () => {
    const s = SAMPLES.find((x) => x.id === sel.value);
    if (s) {
      loadConfig(s.config, s.source);
      history.replaceState(null, '', '?sample=' + s.id);
    }
  };
}

$('btnExpandAll').onclick = () => {
  if (!state.graph) return;
  state.expanded.clear(); state.lru = [];
  state.graph.stacks.forEach((s, si) => {
    for (let li = 0; li < s.count; li++) { state.expanded.add(si + ':' + li); state.lru.push(si + ':' + li); }
  });
  rebuild(true);
};
$('btnCollapseAll').onclick = () => {
  state.expanded.clear(); state.expandedExperts.clear(); state.lru = [];
  clearPin();
  rebuild(true);
};
$('btnResetCam').onclick = () => { if (state.layout) camera.fit(state.layout.bounds); };
$('lodSelect').onchange = () => {
  state.cellMode = Number($('lodSelect').value);
  requestRender();
};

$('btnView').onclick = () => {
  state.viewMode = state.viewMode === '3d' ? '2d' : '3d';
  applyViewMode();
};
function applyViewMode() {
  const is2d = state.viewMode === '2d';
  $('view2d').classList.toggle('active', is2d);
  canvas.style.visibility = is2d ? 'hidden' : 'visible';
  $('labels').style.display = is2d ? 'none' : 'block';
  $('btnView').textContent = is2d ? '3D 视图' : '2D 视图';
  if (is2d) { stopAnim(); render2D(); } else requestRender();
}
function render2D() {
  if (!state.graph) return;
  $('view2d').innerHTML = buildSVG(state.graph);
}

function stopAnim() {
  state.anim.on = false;
  state.anim.last = 0;
  cancelAnimationFrame(state.anim.raf);
  $('btnAnim').textContent = '▶ 动画';
  requestRender();
}
$('btnAnim').onclick = () => {
  if (!state.graph) return;
  if (state.anim.on) { stopAnim(); return; }
  if (state.graph.meta.tier === 'T2' && !state.lru.length) {
    status('T2 大模型:请先展开一层,动画将聚焦该层');
    return;
  }
  state.anim.on = true;
  $('btnAnim').textContent = '⏸ 动画';
  state.anim.raf = requestAnimationFrame(animLoop);
};

// export menu
$('btnExport').onclick = (e) => {
  e.stopPropagation();
  $('exportMenu').classList.toggle('open');
};
document.addEventListener('click', () => $('exportMenu').classList.remove('open'));
$('exportMenu').addEventListener('click', async (e) => {
  const x = e.target?.dataset?.x;
  if (!x || !state.graph) return;
  $('exportMenu').classList.remove('open');
  const name = (state.graph.meta.name || 'model').replace(/[\/\s]+/g, '_');
  try {
    if (x === 'png') {
      status('渲染 2× 截图…');
      downloadBlob(await renderer.exportPNG(camera, { focus: state.focus, cellMode: state.cellMode }, 2), `${name}.png`);
    } else if (x === 'poster') {
      status('合成海报…');
      const blob = await renderer.exportPNG(camera, { focus: state.focus, cellMode: state.cellMode }, 2);
      downloadBlob(await exportPoster(blob, state.graph), `${name}_poster.png`);
    } else if (x === 'svg') {
      downloadText(buildSVG(state.graph), `${name}.svg`, 'image/svg+xml');
    } else if (x === 'glb') {
      status('烘焙 GLB…');
      downloadBlob(exportGLB(state.layout.soa), `${name}.glb`);
    } else if (x === 'llmarch') {
      downloadText(exportLlmarch({
        config: state.graph.rawConfig,
        source: state.graph.meta.source,
        name: state.graph.meta.name,
        view: {
          camera: { yaw: camera.yaw, pitch: camera.pitch, dist: camera.dist, target: camera.target },
          expanded: [...state.expanded],
          expandedExperts: [...state.expandedExperts],
        },
      }), `${name}.llmarch`);
    }
  } catch (err) {
    status('导出失败:' + err.message, 7000);
  }
});

// import: file picker + drag&drop + paste modal
$('btnImport').onclick = () => $('fileInput').click();
$('fileInput').onchange = async (e) => {
  const f = e.target.files[0];
  if (f) importFile(f);
  e.target.value = '';
};
async function importFile(file) {
  try {
    if (/\.safetensors$/i.test(file.name)) {
      status('读取 safetensors 文件头(仅头部,权重不加载)…', 10000);
      const tensors = await readSafetensorsFile(file);
      const { config, notes } = inferConfigFromTensors(tensors, file.name);
      loadConfig(config, { kind: 'manual', name: file.name + '(本地推断)', verified: true });
      status(`已从 ${tensors.length} 个张量推断架构 · ` + notes[0], 9000);
      console.info('safetensors 推断说明:\n- ' + notes.join('\n- '));
      return;
    }
    const doc = parseLlmarch(await file.text());
    loadConfig(doc.config, doc.source);
    if (doc.view) restoreView(doc.view);
  } catch (err) {
    status('导入失败:' + err.message, 7000);
  }
}
function restoreView(view) {
  if (view.expanded) { state.expanded = new Set(view.expanded); state.lru = [...view.expanded]; }
  if (view.expandedExperts) state.expandedExperts = new Set(view.expandedExperts);
  rebuild(true);
  if (view.camera) {
    camera.yaw = view.camera.yaw; camera.pitch = view.camera.pitch;
    camera.dist = view.camera.dist; camera.target = view.camera.target;
    requestRender();
  }
}
window.addEventListener('dragover', (e) => { e.preventDefault(); $('drop-overlay').classList.add('on'); });
window.addEventListener('dragleave', (e) => { if (!e.relatedTarget) $('drop-overlay').classList.remove('on'); });
window.addEventListener('drop', (e) => {
  e.preventDefault();
  $('drop-overlay').classList.remove('on');
  const f = e.dataTransfer.files?.[0];
  if (f) importFile(f);
});

// paste/manual modal
{
  const tpl = $('tplSelect');
  tpl.innerHTML = SAMPLES.map((s) => `<option value="${s.id}">${s.name}</option>`).join('');
  $('tplApply').onclick = () => {
    const s = SAMPLES.find((x) => x.id === tpl.value);
    if (s) $('cfgText').value = JSON.stringify(s.config, null, 2);
  };
}
$('btnPaste').onclick = () => { $('modal').classList.add('open'); $('modalErr').textContent = ''; };
$('modalCancel').onclick = () => $('modal').classList.remove('open');
$('modal').addEventListener('click', (e) => { if (e.target === $('modal')) $('modal').classList.remove('open'); });
$('modalOk').onclick = () => {
  try {
    const doc = parseLlmarch($('cfgText').value);
    loadConfig(doc.config, { ...doc.source, kind: 'manual' });
    $('modal').classList.remove('open');
  } catch (err) {
    $('modalErr').textContent = err.message;
  }
};

// token persistence (best-effort)
try {
  $('tokenInput').value = localStorage.getItem('hfToken') || '';
  $('tokenInput').addEventListener('change', () => localStorage.setItem('hfToken', $('tokenInput').value.trim()));
} catch { /* private mode */ }

window.addEventListener('resize', () => requestRender());

// debug/test hook
window.__viz = { state, applyAction, rebuild, camera, pinItem, clearPin, expandLayer, handleNav, startTour, gotoStation, endTour, tourState, searchModel, jumpToResult };

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
applyViewMode();
{
  const q = new URLSearchParams(location.search);
  const repo = q.get('repo');
  const sample = q.get('sample');
  if (repo) {
    $('repoInput').value = repo;
    $('btnLoad').click();
  } else {
    const s = SAMPLES.find((x) => x.id === (sample || 'qwen3-30b-a3b')) || SAMPLES[0];
    $('sampleSelect').value = s.id;
    loadConfig(s.config, s.source);
  }
}
