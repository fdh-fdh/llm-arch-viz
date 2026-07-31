// Orbit camera: yaw/pitch around a target, wheel zoom, drag rotate, shift/right-drag pan.

import { mat4Perspective, mat4LookAt, mat4Multiply, mat4Identity } from './mat4.js';

export class OrbitCamera {
  constructor(canvas, onChange) {
    this.canvas = canvas;
    this.onChange = onChange;
    this.target = [0, 0, 0];
    this.yaw = 0.5;          // radians
    this.pitch = 0.25;
    this.dist = 60;
    this.fovy = 45 * Math.PI / 180;
    this.near = 0.1;
    this.far = 5000;
    this.view = mat4Identity();
    this.proj = mat4Identity();
    this.viewProj = mat4Identity();
    this._dragging = null;
    this._bind();
  }

  eye() {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    return [
      this.target[0] + this.dist * cp * sy,
      this.target[1] + this.dist * sp,
      this.target[2] + this.dist * cp * cy,
    ];
  }

  update(aspect) {
    mat4Perspective(this.proj, this.fovy, aspect, this.near, this.far);
    mat4LookAt(this.view, this.eye(), this.target, [0, 1, 0]);
    mat4Multiply(this.viewProj, this.proj, this.view);
  }

  // Smooth tween to a target view (FR-C6). Compatible with demand rendering:
  // drives onChange every frame while animating, then stops.
  flyTo({ target, dist, yaw, pitch }, ms = 650) {
    cancelAnimationFrame(this._flyRaf);
    const from = {
      target: [...this.target], dist: this.dist, yaw: this.yaw, pitch: this.pitch,
    };
    const to = {
      target: target || from.target,
      dist: dist ?? from.dist,
      yaw: yaw ?? from.yaw,
      pitch: pitch ?? from.pitch,
    };
    const t0 = performance.now();
    const ease = (t) => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    const step = (now) => {
      const t = Math.min(1, (now - t0) / ms);
      const e = ease(t);
      this.target = from.target.map((a, i) => a + (to.target[i] - a) * e);
      this.dist = from.dist + (to.dist - from.dist) * e;
      this.yaw = from.yaw + (to.yaw - from.yaw) * e;
      this.pitch = from.pitch + (to.pitch - from.pitch) * e;
      this.onChange && this.onChange();
      if (t < 1) this._flyRaf = requestAnimationFrame(step);
    };
    this._flyRaf = requestAnimationFrame(step);
  }

  fit(bounds) {
    // bounds: {min:[x,y,z], max:[x,y,z]}
    const cx = (bounds.min[0] + bounds.max[0]) / 2;
    const cy = (bounds.min[1] + bounds.max[1]) / 2;
    const cz = (bounds.min[2] + bounds.max[2]) / 2;
    const dx = bounds.max[0] - bounds.min[0];
    const dy = bounds.max[1] - bounds.min[1];
    const dz = bounds.max[2] - bounds.min[2];
    const radius = Math.max(1, Math.hypot(dx, dy, dz) / 2);
    this.yaw = 0.5;          // reset to the default pleasant angle
    this.pitch = 0.25;
    this.target = [cx, cy, cz];
    this.dist = radius / Math.tan(this.fovy / 2) * 1.15;
    this.far = Math.max(5000, this.dist * 10);
    this.onChange && this.onChange();
  }

  _bind() {
    const el = this.canvas;
    el.addEventListener('pointerdown', (e) => {
      el.setPointerCapture(e.pointerId);
      this._dragging = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
    });
    el.addEventListener('pointermove', (e) => {
      if (!this._dragging) return;
      const dx = e.clientX - this._dragging.x;
      const dy = e.clientY - this._dragging.y;
      this._dragging.x = e.clientX;
      this._dragging.y = e.clientY;
      if (this._dragging.pan) {
        const scale = this.dist * 0.0016;
        const cy2 = Math.cos(this.yaw), sy2 = Math.sin(this.yaw);
        // right vector (x-z plane), up approximated by world Y
        this.target[0] -= (dx * cy2) * scale;
        this.target[2] -= (-dx * sy2) * scale;
        this.target[1] += dy * scale;
      } else {
        this.yaw -= dx * 0.005;
        this.pitch += dy * 0.005;
        const lim = Math.PI / 2 - 0.02;
        this.pitch = Math.max(-lim, Math.min(lim, this.pitch));
      }
      this.onChange && this.onChange();
    });
    const endDrag = () => { this._dragging = null; };
    el.addEventListener('pointerup', endDrag);
    el.addEventListener('pointercancel', endDrag);
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      const f = Math.exp(e.deltaY * 0.0012);
      this.dist = Math.max(1.5, Math.min(4000, this.dist * f));
      this.onChange && this.onChange();
    }, { passive: false });
  }
}
