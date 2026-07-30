// llm-arch-viz — app bootstrap & UI wiring (zero-dependency vanilla ES modules).

import { Renderer } from './gl/renderer.js';
import { OrbitCamera } from './gl/camera.js';
import { pick } from './gl/pick.js';
import { mat4ProjectPoint } from './gl/mat4.js';
import { buildGraph, fmtParams, TIER_INFO } from './parser/ir.js';
import { LayoutBuilder } from './layout.js';
import { buildSVG } from './viz2d.js';
import { exportLlmarch, parseLlmarch } from './llmarch.js';
import { fetchModel } from './hf.js';
import { downloadBlob, downloadText, exportPoster, exportGLB } from './export.js';
import { SAMPLES } from './samples.js';

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
  anim: { on: false, raf: 0, t: 0, last: 0 },
  labelPool: [],
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
  renderer.render(camera, { hoverId: state.hoverId, pulseY });
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
  const crowded = lay.labels.length > 80; // many layers expanded: keep only major labels
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
// Layout / state rebuild
// ---------------------------------------------------------------------------
function rebuild(fit = false) {
  if (!state.graph) return;
  state.layout = builder.build(state.graph, state);
  state.layout.expandedRanges = builder.expandedRanges || [];
  renderer.setInstances(state.layout.soa);
  state.hoverId = -1;
  if (fit) camera.fit(state.layout.bounds);
  if (state.viewMode === '2d') render2D();
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

function applyAction(action) {
  if (!action) return;
  if (action.kind === 'expandLayer') expandLayer(action.si, action.li);
  else if (action.kind === 'collapseLayer') collapseLayer(action.si, action.li);
  else if (action.kind === 'expandExperts') { state.expandedExperts.add(action.si + ':' + action.li); rebuild(); }
  else if (action.kind === 'collapseExperts') { state.expandedExperts.delete(action.si + ':' + action.li); rebuild(); }
}

// ---------------------------------------------------------------------------
// Hover / click picking
// ---------------------------------------------------------------------------
let lastPickTime = 0;
canvas.addEventListener('pointermove', (e) => {
  if (!state.layout || e.buttons) return;
  const now = performance.now();
  if (now - lastPickTime < 33) return;    // ~30 Hz throttle
  lastPickTime = now;
  const idx = pick(camera, canvas, e.clientX, e.clientY, state.layout.soa);
  const item = idx >= 0 ? state.layout.items[idx] : null;
  const tip = $('tooltip');
  if (item) {
    if (state.hoverId !== idx) { state.hoverId = idx; requestRender(); }
    tip.style.display = 'block';
    tip.innerHTML = `<div class="tt">${escapeHtml(item.title)}</div>` +
      (item.lines || []).map((l) => `<div class="tl">${escapeHtml(l)}</div>`).join('');
    const rect = canvas.getBoundingClientRect();
    let tx = e.clientX - rect.left + 16, ty = e.clientY - rect.top + 14;
    tip.style.left = Math.min(tx, rect.width - 440) + 'px';
    tip.style.top = Math.min(ty, rect.height - 160) + 'px';
    canvas.style.cursor = item.action ? 'pointer' : 'default';
  } else {
    if (state.hoverId !== -1) { state.hoverId = -1; requestRender(); }
    tip.style.display = 'none';
    canvas.style.cursor = 'default';
  }
});
canvas.addEventListener('pointerleave', () => {
  state.hoverId = -1;
  $('tooltip').style.display = 'none';
  requestRender();
});

let downPos = null;
canvas.addEventListener('pointerdown', (e) => { downPos = [e.clientX, e.clientY]; });
canvas.addEventListener('pointerup', (e) => {
  if (!downPos || !state.layout) return;
  const moved = Math.hypot(e.clientX - downPos[0], e.clientY - downPos[1]);
  downPos = null;
  if (moved > 5) return; // was a drag
  const idx = pick(camera, canvas, e.clientX, e.clientY, state.layout.soa);
  if (idx >= 0) applyAction(state.layout.items[idx]?.action);
});

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ---------------------------------------------------------------------------
// Sidebar
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

  // tier-driven controls
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
  stopAnim();
  // default: expand the first layer of the first stack (bbycroft-style sample layer)
  expandDefault(graph);
  updateSidebar();
  rebuild(true);
  status(`已加载 ${graph.meta.name} — 总参 ${fmtParams(graph.meta.params.total)},激活 ${fmtParams(graph.meta.params.active)}`);
}

function expandDefault(graph) {
  state.expanded.add('0:0');
  state.lru.push('0:0');
  // MoE 小专家数(≤64)默认直接展开专家网格
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

// samples
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
  rebuild(true);
};
$('btnResetCam').onclick = () => { if (state.layout) camera.fit(state.layout.bounds); };

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
      downloadBlob(await renderer.exportPNG(camera, {}, 2), `${name}.png`);
    } else if (x === 'poster') {
      status('合成海报…');
      const blob = await renderer.exportPNG(camera, {}, 2);
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

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
// debug/test hook
window.__viz = { state, applyAction, rebuild, camera };

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
