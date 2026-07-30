// CPU ray vs AABB picking over the instance SoA (a few thousand visible boxes:
// brute force at ~30 Hz is faster than building any acceleration structure).

import { mat4Invert, mat4Identity } from './mat4.js';

const invVP = mat4Identity();

function unproject(m, x, y, z) {
  const px = m[0] * x + m[4] * y + m[8] * z + m[12];
  const py = m[1] * x + m[5] * y + m[9] * z + m[13];
  const pz = m[2] * x + m[6] * y + m[10] * z + m[14];
  const pw = m[3] * x + m[7] * y + m[11] * z + m[15];
  return [px / pw, py / pw, pz / pw];
}

// Returns instance index or -1. soa: {pos, scale, flag, count}; pickableFlagMax: skip flags > max.
export function pick(camera, canvas, clientX, clientY, soa, pickableFlagMax = 2.5) {
  const rect = canvas.getBoundingClientRect();
  const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
  const ndcY = -(((clientY - rect.top) / rect.height) * 2 - 1);
  mat4Invert(invVP, camera.viewProj);
  const o = unproject(invVP, ndcX, ndcY, -1);
  const f = unproject(invVP, ndcX, ndcY, 1);
  const d = [f[0] - o[0], f[1] - o[1], f[2] - o[2]];
  const len = Math.hypot(d[0], d[1], d[2]) || 1;
  d[0] /= len; d[1] /= len; d[2] /= len;

  let best = -1;
  let bestT = Infinity;
  const inv = [1 / (d[0] || 1e-12), 1 / (d[1] || 1e-12), 1 / (d[2] || 1e-12)];
  for (let i = 0; i < soa.count; i++) {
    if (soa.flag[i] > pickableFlagMax) continue;
    const i3 = i * 3;
    const hx = soa.scale[i3] / 2, hy = soa.scale[i3 + 1] / 2, hz = soa.scale[i3 + 2] / 2;
    const minX = soa.pos[i3] - hx, maxX = soa.pos[i3] + hx;
    const minY = soa.pos[i3 + 1] - hy, maxY = soa.pos[i3 + 1] + hy;
    const minZ = soa.pos[i3 + 2] - hz, maxZ = soa.pos[i3 + 2] + hz;
    let t1 = (minX - o[0]) * inv[0], t2 = (maxX - o[0]) * inv[0];
    let tmin = Math.min(t1, t2), tmax = Math.max(t1, t2);
    t1 = (minY - o[1]) * inv[1]; t2 = (maxY - o[1]) * inv[1];
    tmin = Math.max(tmin, Math.min(t1, t2)); tmax = Math.min(tmax, Math.max(t1, t2));
    t1 = (minZ - o[2]) * inv[2]; t2 = (maxZ - o[2]) * inv[2];
    tmin = Math.max(tmin, Math.min(t1, t2)); tmax = Math.min(tmax, Math.max(t1, t2));
    if (tmax >= Math.max(tmin, 0) && tmin < bestT) {
      bestT = tmin;
      best = i;
    }
  }
  return best;
}
