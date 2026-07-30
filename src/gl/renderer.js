// Zero-dependency WebGL2 instanced-cube renderer.
// One shared cube geometry + per-instance SoA buffers, a single draw call.
// v2: procedural element-cell grid in the fragment shader (LOD4, zero memory),
//     head/KV-group bands, hover-cell highlight, focus dimming.

const VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aVert;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aPos;       // instance center
layout(location=3) in vec3 aScale;
layout(location=4) in vec3 aColor;
layout(location=5) in float aFlag;     // 0 solid, 1 connector, 2 aggregate
layout(location=6) in vec2 aUV;        // per-face uv
layout(location=7) in vec2 aGrid;      // element grid [cols, rows] (0 = no grid)
layout(location=8) in vec2 aSub;       // subdiv bands [bands, groupSize] (0 = none)
layout(location=9) in float aDim;      // 1 = dimmed in focus mode
uniform mat4 uViewProj;
uniform float uHoverId;
uniform float uPulseY;
uniform float uInstanceBase;
out vec3 vColor;
out vec3 vNormal;
out float vFlag;
out float vHover;
out vec2 vUV;
out vec2 vGrid;
out vec2 vSub;
out float vDim;
void main() {
  vec3 world = aPos + aVert * aScale;
  gl_Position = uViewProj * vec4(world, 1.0);
  vNormal = aNormal;
  vFlag = aFlag;
  vUV = aUV;
  vGrid = aGrid;
  vSub = aSub;
  vDim = aDim;
  float id = uInstanceBase + float(gl_InstanceID);
  vHover = (abs(id - uHoverId) < 0.5) ? 1.0 : 0.0;
  vec3 c = aColor;
  if (uPulseY > -1e8) {
    float d = abs(aPos.y - uPulseY);
    float glow = smoothstep(3.5, 0.0, d);
    c = mix(c, vec3(1.0, 0.95, 0.55), glow * 0.75);
  }
  vColor = c;
}`;

const FS = `#version 300 es
precision highp float;
in vec3 vColor;
in vec3 vNormal;
in float vFlag;
in float vHover;
in vec2 vUV;
in vec2 vGrid;
in vec2 vSub;
in float vDim;
uniform float uFocus;        // 1 = focus mode active (dim others)
uniform vec2 uHoverCell;     // hovered element cell (col,row)
uniform float uHoverCellOn;
out vec4 outColor;

float hash21(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
  vec3 n = normalize(vNormal);
  float l1 = max(dot(n, normalize(vec3(0.5, 0.8, 0.6))), 0.0);
  float l2 = max(dot(n, normalize(vec3(-0.6, 0.2, -0.5))), 0.0);
  float light = 0.52 + 0.42 * l1 + 0.18 * l2;
  vec3 c = vColor;

  // ---- LOD4 procedural element cells (zero-memory bbycroft look) ----
  if (vGrid.x > 0.5 && vFlag < 0.5) {
    vec2 g = vUV * vGrid;
    vec2 fw = fwidth(g);
    float fade = clamp(1.6 - max(fw.x, fw.y), 0.0, 1.0);  // cells < ~1px fade out
    if (fade > 0.01) {
      vec2 cell = floor(g);
      float h = hash21(cell);
      c *= 1.0 + (h - 0.5) * 0.16 * fade;                  // checkerboard-ish shimmer
      vec2 fr = fract(g);
      vec2 lw = clamp(fw * 1.4, 0.03, 0.45);
      float line = max(1.0 - step(lw.x, fr.x), 1.0 - step(lw.y, fr.y));
      c *= 1.0 - 0.20 * line * fade;                       // grid lines
      if (vSub.x > 0.5) {                                  // head / KV-group bands
        float band = floor(vUV.y * vSub.x);
        float grp = (vSub.y > 0.5) ? floor(band / vSub.y) : band;
        float gh = fract(grp * 0.61803398875);
        vec3 tint = vec3(0.85 + 0.3 * gh, 1.0 - 0.18 * gh, 0.85 + 0.3 * (1.0 - gh));
        c = mix(c, c * tint, 0.30 * fade);
      }
      if (vHover > 0.5 && uHoverCellOn > 0.5) {            // hovered element highlight
        if (all(lessThan(abs(cell - floor(uHoverCell)), vec2(0.5)))) {
          c = mix(c, vec3(1.0, 0.92, 0.35), 0.85);
        }
      }
    }
  }

  c *= light;
  if (vFlag > 0.5 && vFlag < 1.5) c *= 0.9;
  if (vHover > 0.5) c = mix(c, vec3(1.0), 0.30);
  if (uFocus > 0.5 && vDim > 0.5) c = mix(c, vec3(0.90, 0.92, 0.95), 0.85);
  outColor = vec4(c, 1.0);
}`;

// Unit cube: 24 vertices (per-face normals + per-face uv), 36 indices.
// Top/bottom faces map uv = (x, z) so matrix grids read cols→x, rows→z.
function buildCube() {
  const p = 0.5;
  const faces = [
    { n: [0, 0, 1],  v: [[-p,-p,p],[p,-p,p],[p,p,p],[-p,p,p]],   uv: [[0,0],[1,0],[1,1],[0,1]] },
    { n: [0, 0, -1], v: [[p,-p,-p],[-p,-p,-p],[-p,p,-p],[p,p,-p]], uv: [[1,0],[0,0],[0,1],[1,1]] },
    { n: [1, 0, 0],  v: [[p,-p,p],[p,-p,-p],[p,p,-p],[p,p,p]],   uv: [[0,1],[0,0],[1,0],[1,1]] },
    { n: [-1, 0, 0], v: [[-p,-p,-p],[-p,-p,p],[-p,p,p],[-p,p,-p]], uv: [[0,0],[0,1],[1,1],[1,0]] },
    { n: [0, 1, 0],  v: [[-p,p,p],[p,p,p],[p,p,-p],[-p,p,-p]],   uv: [[0,1],[1,1],[1,0],[0,0]] },
    { n: [0, -1, 0], v: [[-p,-p,-p],[p,-p,-p],[p,-p,p],[-p,-p,p]], uv: [[0,0],[1,0],[1,1],[0,1]] },
  ];
  const verts = [], normals = [], uvs = [], idx = [];
  let base = 0;
  for (const f of faces) {
    for (let i = 0; i < 4; i++) { verts.push(...f.v[i]); normals.push(...f.n); uvs.push(...f.uv[i]); }
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
    base += 4;
  }
  return {
    verts: new Float32Array(verts),
    normals: new Float32Array(normals),
    uvs: new Float32Array(uvs),
    idx: new Uint16Array(idx),
  };
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      antialias: true, alpha: false, preserveDrawingBuffer: false, powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 not available');
    this.gl = gl;
    this.count = 0;
    this.dprCap = 2;
    this.clearColor = [0.965, 0.972, 0.98];
    this._initProgram();
    this._initGeometry();
    this._initInstanceBuffers();
    canvas.addEventListener('webglcontextlost', (e) => e.preventDefault());
  }

  _compile(type, src) {
    const gl = this.gl;
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error('Shader compile error: ' + gl.getShaderInfoLog(s));
    }
    return s;
  }

  _initProgram() {
    const gl = this.gl;
    const prog = gl.createProgram();
    gl.attachShader(prog, this._compile(gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, this._compile(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('Program link error: ' + gl.getProgramInfoLog(prog));
    }
    this.prog = prog;
    const u = (n) => gl.getUniformLocation(prog, n);
    this.uViewProj = u('uViewProj');
    this.uHoverId = u('uHoverId');
    this.uPulseY = u('uPulseY');
    this.uInstanceBase = u('uInstanceBase');
    this.uFocus = u('uFocus');
    this.uHoverCell = u('uHoverCell');
    this.uHoverCellOn = u('uHoverCellOn');
  }

  _initGeometry() {
    const gl = this.gl;
    const cube = buildCube();
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const attach = (data, loc, size) => {
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    };
    attach(cube.verts, 0, 3);
    attach(cube.normals, 1, 3);
    attach(cube.uvs, 6, 2);
    const ib = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, cube.idx, gl.STATIC_DRAW);
    this.indexCount = cube.idx.length;
    gl.bindVertexArray(null);
  }

  _initInstanceBuffers() {
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    const mk = (loc, size) => {
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(loc, 1);
      return b;
    };
    this.bufPos = mk(2, 3);
    this.bufScale = mk(3, 3);
    this.bufColor = mk(4, 3);
    this.bufFlag = mk(5, 1);
    this.bufGrid = mk(7, 2);
    this.bufSub = mk(8, 2);
    this.bufDim = mk(9, 1);
    gl.bindVertexArray(null);
  }

  // soa: {pos, scale, color, flag, grid, sub, count}
  setInstances(soa) {
    const gl = this.gl;
    this.count = soa.count;
    const upload = (buf, arr, comps) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, arr.subarray(0, soa.count * comps), gl.DYNAMIC_DRAW);
    };
    upload(this.bufPos, soa.pos, 3);
    upload(this.bufScale, soa.scale, 3);
    upload(this.bufColor, soa.color, 3);
    upload(this.bufFlag, soa.flag, 1);
    upload(this.bufGrid, soa.grid, 2);
    upload(this.bufSub, soa.sub, 2);
    this.setDim(soa.dim || new Float32Array(soa.count));
  }

  // Focus-mode dimming: update only the dim buffer (cheap, no rebuild).
  setDim(dimArr) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufDim);
    gl.bufferData(gl.ARRAY_BUFFER, dimArr.subarray(0, this.count), gl.DYNAMIC_DRAW);
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, this.dprCap);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    return [w, h];
  }

  _draw(camera, opts) {
    const gl = this.gl;
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    const [r, g, b] = this.clearColor;
    gl.clearColor(r, g, b, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.count) return;
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.uViewProj, false, camera.viewProj);
    gl.uniform1f(this.uHoverId, opts.hoverId ?? -1);
    gl.uniform1f(this.uPulseY, opts.pulseY ?? -1e9);
    gl.uniform1f(this.uInstanceBase, 0);
    gl.uniform1f(this.uFocus, opts.focus ? 1 : 0);
    gl.uniform2f(this.uHoverCell, opts.hoverCell?.[0] ?? -1, opts.hoverCell?.[1] ?? -1);
    gl.uniform1f(this.uHoverCellOn, opts.hoverCell ? 1 : 0);
    gl.bindVertexArray(this.vao);
    gl.drawElementsInstanced(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0, this.count);
    gl.bindVertexArray(null);
  }

  render(camera, opts = {}) {
    const [w, h] = this.resize();
    camera.update(w / h);
    this.gl.viewport(0, 0, w, h);
    this._draw(camera, opts);
  }

  async exportPNG(camera, opts = {}, scale = 2) {
    const canvas = this.canvas;
    const prevW = canvas.width, prevH = canvas.height;
    const dpr = Math.min(window.devicePixelRatio || 1, this.dprCap);
    canvas.width = Math.floor(canvas.clientWidth * dpr * scale);
    canvas.height = Math.floor(canvas.clientHeight * dpr * scale);
    this.gl.viewport(0, 0, canvas.width, canvas.height);
    camera.update(canvas.width / canvas.height);
    this._draw(camera, { ...opts, hoverId: -1 });
    const blob = await new Promise((res) => canvas.toBlob(res, 'image/png'));
    canvas.width = prevW;
    canvas.height = prevH;
    return blob;
  }
}
