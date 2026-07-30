// Zero-dependency WebGL2 instanced-cube renderer.
// One shared cube geometry + one interleaved-per-attribute instance buffer set,
// a single draw call for the whole scene (the memory design from the plan docs).

const VS = `#version 300 es
precision highp float;
layout(location=0) in vec3 aVert;      // cube vertex (unit cube, centered)
layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aPos;       // instance center
layout(location=3) in vec3 aScale;     // instance size
layout(location=4) in vec3 aColor;
layout(location=5) in float aFlag;     // 0 solid, 1 connector, 2 aggregate
uniform mat4 uViewProj;
uniform float uHoverId;
uniform float uPulseY;                 // world-space y of the data-flow pulse (< -1e8 disables)
out vec3 vColor;
out vec3 vNormal;
out float vFlag;
out float vHover;
out vec3 vWorld;
uniform float uInstanceBase;
void main() {
  vec3 world = aPos + aVert * aScale;
  gl_Position = uViewProj * vec4(world, 1.0);
  vNormal = aNormal;
  vFlag = aFlag;
  vWorld = world;
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
in vec3 vWorld;
out vec4 outColor;
void main() {
  vec3 n = normalize(vNormal);
  float l1 = max(dot(n, normalize(vec3(0.5, 0.8, 0.6))), 0.0);
  float l2 = max(dot(n, normalize(vec3(-0.6, 0.2, -0.5))), 0.0);
  float light = 0.52 + 0.42 * l1 + 0.18 * l2;
  vec3 c = vColor * light;
  if (vFlag > 0.5 && vFlag < 1.5) c *= 0.9;          // connector: slightly dimmer
  if (vHover > 0.5) c = mix(c, vec3(1.0), 0.35);      // hover highlight
  outColor = vec4(c, 1.0);
}`;

// Unit cube: 24 vertices (per-face normals), 36 indices.
function buildCube() {
  const p = 0.5;
  const faces = [
    { n: [0, 0, 1],  v: [[-p,-p,p],[p,-p,p],[p,p,p],[-p,p,p]] },
    { n: [0, 0, -1], v: [[p,-p,-p],[-p,-p,-p],[-p,p,-p],[p,p,-p]] },
    { n: [1, 0, 0],  v: [[p,-p,p],[p,-p,-p],[p,p,-p],[p,p,p]] },
    { n: [-1, 0, 0], v: [[-p,-p,-p],[-p,-p,p],[-p,p,p],[-p,p,-p]] },
    { n: [0, 1, 0],  v: [[-p,p,p],[p,p,p],[p,p,-p],[-p,p,-p]] },
    { n: [0, -1, 0], v: [[-p,-p,-p],[p,-p,-p],[p,-p,p],[-p,-p,p]] },
  ];
  const verts = [], normals = [], idx = [];
  let base = 0;
  for (const f of faces) {
    for (const v of f.v) { verts.push(...v); normals.push(...f.n); }
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
    base += 4;
  }
  return {
    verts: new Float32Array(verts),
    normals: new Float32Array(normals),
    idx: new Uint16Array(idx),
  };
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
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
    this.uViewProj = gl.getUniformLocation(prog, 'uViewProj');
    this.uHoverId = gl.getUniformLocation(prog, 'uHoverId');
    this.uPulseY = gl.getUniformLocation(prog, 'uPulseY');
    this.uInstanceBase = gl.getUniformLocation(prog, 'uInstanceBase');
  }

  _initGeometry() {
    const gl = this.gl;
    const cube = buildCube();
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    const vb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    gl.bufferData(gl.ARRAY_BUFFER, cube.verts, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    const nb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, nb);
    gl.bufferData(gl.ARRAY_BUFFER, cube.normals, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

    const ib = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, cube.idx, gl.STATIC_DRAW);
    this.indexCount = cube.idx.length;
    gl.bindVertexArray(null);
  }

  _initInstanceBuffers() {
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    this.bufPos = gl.createBuffer();
    this.bufScale = gl.createBuffer();
    this.bufColor = gl.createBuffer();
    this.bufFlag = gl.createBuffer();
    const setup = (buf, loc, size) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(loc, 1);
    };
    setup(this.bufPos, 2, 3);
    setup(this.bufScale, 3, 3);
    setup(this.bufColor, 4, 3);
    setup(this.bufFlag, 5, 1);
    gl.bindVertexArray(null);
  }

  // soa: {pos: Float32Array, scale: Float32Array, color: Float32Array, flag: Float32Array, count}
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

  render(camera, { hoverId = -1, pulseY = -1e9 } = {}) {
    const gl = this.gl;
    const [w, h] = this.resize();
    camera.update(w / h);
    gl.viewport(0, 0, w, h);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    const [r, g, b] = this.clearColor;
    gl.clearColor(r, g, b, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.count) return;
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.uViewProj, false, camera.viewProj);
    gl.uniform1f(this.uHoverId, hoverId);
    gl.uniform1f(this.uPulseY, pulseY);
    gl.uniform1f(this.uInstanceBase, 0);
    gl.bindVertexArray(this.vao);
    gl.drawElementsInstanced(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0, this.count);
    gl.bindVertexArray(null);
  }

  // Off-screen style high-res capture: temporarily scale the backing store, render, toBlob, restore.
  async exportPNG(camera, opts = {}, scale = 2) {
    const canvas = this.canvas;
    const prevW = canvas.width, prevH = canvas.height;
    const dpr = Math.min(window.devicePixelRatio || 1, this.dprCap);
    canvas.width = Math.floor(canvas.clientWidth * dpr * scale);
    canvas.height = Math.floor(canvas.clientHeight * dpr * scale);
    const gl = this.gl;
    gl.viewport(0, 0, canvas.width, canvas.height);
    camera.update(canvas.width / canvas.height);
    // render directly (same path as render() but without resize())
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    const [r, g, b] = this.clearColor;
    gl.clearColor(r, g, b, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (this.count) {
      gl.useProgram(this.prog);
      gl.uniformMatrix4fv(this.uViewProj, false, camera.viewProj);
      gl.uniform1f(this.uHoverId, -1);
      gl.uniform1f(this.uPulseY, opts.pulseY ?? -1e9);
      gl.uniform1f(this.uInstanceBase, 0);
      gl.bindVertexArray(this.vao);
      gl.drawElementsInstanced(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0, this.count);
      gl.bindVertexArray(null);
    }
    const blob = await new Promise((res) => canvas.toBlob(res, 'image/png'));
    canvas.width = prevW;
    canvas.height = prevH;
    return blob;
  }
}
