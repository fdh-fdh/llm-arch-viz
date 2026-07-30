// Export: PNG screenshot / poster PNG / GLB (baked merged geometry) / SVG / .llmarch
// GLB note: three.js-style instancing extensions are avoided on purpose — we bake
// instances into one merged mesh so every viewer (Blender, PowerPoint, <model-viewer>) opens it.

import { fmtParams } from './parser/ir.js';

export function downloadBlob(blob, filename) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 800);
}

export function downloadText(text, filename, mime = 'application/json') {
  downloadBlob(new Blob([text], { type: mime }), filename);
}

// ---------------------------------------------------------------------------
// Poster PNG: 3D capture + title + param table + watermark, composed on 2D canvas.
// ---------------------------------------------------------------------------
export async function exportPoster(renderBlob, graph) {
  const img = await createImageBitmap(renderBlob);
  const W = 1600;
  const imgH = Math.round(img.height * (W - 160) / img.width);
  const H = 340 + imgH + 210;
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  const meta = graph.meta;

  ctx.fillStyle = '#f4f6fa';
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = '#151c2c';
  ctx.font = '700 54px -apple-system, "Segoe UI", "PingFang SC", sans-serif';
  ctx.fillText(meta.name, 80, 110);
  ctx.font = '400 26px -apple-system, "Segoe UI", "PingFang SC", sans-serif';
  ctx.fillStyle = '#5c6a82';
  ctx.fillText(`model_type: ${meta.modelType} · ${meta.layers} layers · hidden ${meta.hidden} · ctx ${meta.ctx ? meta.ctx.toLocaleString('en-US') : '?'}`, 80, 158);

  // stats chips
  const chips = [
    `总参数 ${fmtParams(meta.params.total)}`,
    `激活/Token ${fmtParams(meta.params.active)}`,
    `Embedding ${fmtParams(meta.params.embedding)}`,
    meta.maxExperts ? `MoE ${meta.maxExperts} 专家` : `${meta.heads} heads / ${meta.kvHeads} KV`,
    meta.tier,
  ];
  let x = 80;
  ctx.font = '600 24px -apple-system, "Segoe UI", "PingFang SC", sans-serif';
  for (const chip of chips) {
    const w = ctx.measureText(chip).width + 44;
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#c9d2e0';
    ctx.lineWidth = 2;
    roundRect(ctx, x, 190, w, 52, 26);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#33415e';
    ctx.fillText(chip, x + 22, 224);
    x += w + 18;
  }

  ctx.save();
  ctx.shadowColor = 'rgba(30,40,70,0.18)';
  ctx.shadowBlur = 30;
  ctx.shadowOffsetY = 10;
  ctx.fillStyle = '#fff';
  roundRect(ctx, 80, 290, W - 160, imgH, 18);
  ctx.fill();
  ctx.restore();
  ctx.save();
  roundRect(ctx, 80, 290, W - 160, imgH, 18);
  ctx.clip();
  ctx.drawImage(img, 80, 290, W - 160, imgH);
  ctx.restore();

  // watermark / provenance
  ctx.font = '400 22px -apple-system, "Segoe UI", "PingFang SC", sans-serif';
  ctx.fillStyle = '#8892a6';
  const src = meta.source?.kind === 'huggingface'
    ? `huggingface.co/${meta.source.repoId}${meta.source.revision ? ' @ ' + meta.source.revision.slice(0, 8) : ''}`
    : meta.source?.kind === 'estimated' ? '社区估算配置(UNVERIFIED)' : '手动配置';
  ctx.fillText(`来源:${src}`, 80, H - 120);
  ctx.fillText('llm-arch-viz · inspired by bbycroft.net/llm', 80, H - 80);
  if (meta.source?.kind === 'estimated' || meta.degraded) {
    ctx.font = '700 30px -apple-system, sans-serif';
    ctx.fillStyle = '#c2452d';
    ctx.fillText(meta.source?.kind === 'estimated' ? '⚠ UNVERIFIED — 社区估算,非官方数据' : '⚠ 通用兜底解析,结构可能不完整', 80, H - 36);
  }
  return new Promise((res) => c.toBlob(res, 'image/png'));
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// ---------------------------------------------------------------------------
// GLB export: bake instance SoA -> single merged mesh (POSITION/NORMAL/COLOR_0).
// ---------------------------------------------------------------------------
const CUBE_FACES = [
  { n: [0, 0, 1],  v: [[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]] },
  { n: [0, 0, -1], v: [[1,-1,-1],[-1,-1,-1],[-1,1,-1],[1,1,-1]] },
  { n: [1, 0, 0],  v: [[1,-1,1],[1,-1,-1],[1,1,-1],[1,1,1]] },
  { n: [-1, 0, 0], v: [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1]] },
  { n: [0, 1, 0],  v: [[-1,1,1],[1,1,1],[1,1,-1],[-1,1,-1]] },
  { n: [0, -1, 0], v: [[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]] },
];

export function exportGLB(soa, { maxInstances = 120000 } = {}) {
  const n = soa.count;
  if (n > maxInstances) {
    throw new Error(`当前场景 ${n.toLocaleString('en-US')} 个实例超过 GLB 导出上限(${maxInstances.toLocaleString('en-US')})。请先折叠部分层再导出。`);
  }
  const vertsPer = 24, idxPer = 36;
  const positions = new Float32Array(n * vertsPer * 3);
  const normals = new Float32Array(n * vertsPer * 3);
  const colors = new Float32Array(n * vertsPer * 3);
  const indices = new Uint32Array(n * idxPer);
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];

  let vp = 0, ip = 0, vbase = 0;
  for (let i = 0; i < n; i++) {
    const i3 = i * 3;
    const px = soa.pos[i3], py = soa.pos[i3 + 1], pz = soa.pos[i3 + 2];
    const hx = soa.scale[i3] / 2, hy = soa.scale[i3 + 1] / 2, hz = soa.scale[i3 + 2] / 2;
    const r = soa.color[i3], g = soa.color[i3 + 1], b = soa.color[i3 + 2];
    for (const f of CUBE_FACES) {
      for (const v of f.v) {
        const x = px + v[0] * hx, y = py + v[1] * hy, z = pz + v[2] * hz;
        positions[vp] = x; normals[vp] = f.n[0]; colors[vp] = r; vp++;
        positions[vp] = y; normals[vp] = f.n[1]; colors[vp] = g; vp++;
        positions[vp] = z; normals[vp] = f.n[2]; colors[vp] = b; vp++;
        if (x < min[0]) min[0] = x; if (x > max[0]) max[0] = x;
        if (y < min[1]) min[1] = y; if (y > max[1]) max[1] = y;
        if (z < min[2]) min[2] = z; if (z > max[2]) max[2] = z;
      }
    }
    for (let f = 0; f < 6; f++) {
      const b0 = vbase + f * 4;
      indices[ip++] = b0; indices[ip++] = b0 + 1; indices[ip++] = b0 + 2;
      indices[ip++] = b0; indices[ip++] = b0 + 2; indices[ip++] = b0 + 3;
    }
    vbase += vertsPer;
  }

  const pad4 = (len) => (len + 3) & ~3;
  const binParts = [positions.buffer, normals.buffer, colors.buffer, indices.buffer];
  const byteLens = binParts.map((bf) => bf.byteLength);
  const offsets = [];
  let off = 0;
  for (const l of byteLens) { offsets.push(off); off = pad4(off + l); }
  const binLen = off;
  const bin = new Uint8Array(binLen);
  binParts.forEach((bf, i) => bin.set(new Uint8Array(bf), offsets[i]));

  const json = {
    asset: { version: '2.0', generator: 'llm-arch-viz' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, name: 'llm-architecture' }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1, COLOR_0: 2 }, indices: 3, material: 0 }] }],
    materials: [{ name: 'boxes', pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 0.9 } }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: n * vertsPer, type: 'VEC3', min, max },
      { bufferView: 1, componentType: 5126, count: n * vertsPer, type: 'VEC3' },
      { bufferView: 2, componentType: 5126, count: n * vertsPer, type: 'VEC3' },
      { bufferView: 3, componentType: 5125, count: n * idxPer, type: 'SCALAR' },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: offsets[0], byteLength: byteLens[0], target: 34962 },
      { buffer: 0, byteOffset: offsets[1], byteLength: byteLens[1], target: 34962 },
      { buffer: 0, byteOffset: offsets[2], byteLength: byteLens[2], target: 34962 },
      { buffer: 0, byteOffset: offsets[3], byteLength: byteLens[3], target: 34963 },
    ],
    buffers: [{ byteLength: binLen }],
  };

  const enc = new TextEncoder();
  let jsonBytes = enc.encode(JSON.stringify(json));
  const jsonPad = pad4(jsonBytes.length);
  const padded = new Uint8Array(jsonPad);
  padded.set(jsonBytes);
  padded.fill(0x20, jsonBytes.length); // pad with spaces
  const total = 12 + 8 + jsonPad + 8 + binLen;
  const out = new ArrayBuffer(total);
  const dv = new DataView(out);
  const u8 = new Uint8Array(out);
  dv.setUint32(0, 0x46546c67, true);  // 'glTF'
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  dv.setUint32(12, jsonPad, true);
  dv.setUint32(16, 0x4e4f534a, true); // 'JSON'
  u8.set(padded, 20);
  dv.setUint32(20 + jsonPad, binLen, true);
  dv.setUint32(24 + jsonPad, 0x004e4942, true); // 'BIN'
  u8.set(bin, 28 + jsonPad);
  return new Blob([out], { type: 'model/gltf-binary' });
}
