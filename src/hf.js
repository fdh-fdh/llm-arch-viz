// HuggingFace Hub connection (Channel A of the data plan).
// config.json / index.json are plain git files served by huggingface.co with CORS.
// Optional proxy fallback for environments where direct access fails.

const HF = 'https://huggingface.co';

function normalizeRepoId(input) {
  let s = input.trim();
  s = s.replace(/^https?:\/\/(www\.)?huggingface\.co\//, '');
  s = s.replace(/\/(tree|blob|resolve)\/.*$/, '');
  s = s.replace(/\/+$/, '');
  return s;
}

async function fetchWithTimeout(url, opts = {}, ms = 15000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), ms);
  try {
    return await fetch(url, { ...opts, signal: ctrl.signal });
  } finally {
    clearTimeout(t);
  }
}

export async function fetchModel(repoInput, { token = null, proxyBase = null } = {}) {
  const repoId = normalizeRepoId(repoInput);
  if (!/^[\w.-]+\/[\w.-]+$/.test(repoId)) {
    throw new Error(`repo id 格式不对:"${repoId}"(应为 owner/name,如 Qwen/Qwen3-30B-A3B)`);
  }
  const headers = token ? { Authorization: 'Bearer ' + token } : {};
  const base = proxyBase ? proxyBase.replace(/\/$/, '') + '/' + HF.replace('https://', '') : HF;

  // model info (revision sha + gated flag) — best effort
  let revision = null, gated = false;
  try {
    const r = await fetchWithTimeout(`${base}/api/models/${repoId}`, { headers });
    if (r.ok) {
      const info = await r.json();
      revision = info.sha || null;
      gated = !!info.gated;
    } else if (r.status === 401 || r.status === 403) {
      gated = true;
    }
  } catch { /* info is optional */ }

  const cfgUrl = `${base}/${repoId}/resolve/main/config.json`;
  let resp;
  try {
    resp = await fetchWithTimeout(cfgUrl, { headers });
  } catch (e) {
    throw new Error(`无法连接 huggingface.co(网络/CORS)。可改用"粘贴 config.json"通道。原始错误:${e.message}`);
  }
  if (!resp.ok) {
    if (resp.status === 401 || resp.status === 403) {
      throw new Error(`该模型是 gated/私有(HTTP ${resp.status})。在设置中填入你的 HF token(fine-grained read-only),或改用本地 config.json 导入。`);
    }
    if (resp.status === 404) throw new Error(`找不到 ${repoId} 的 config.json(HTTP 404)——可能是 GGUF-only 或 diffusers 仓库`);
    throw new Error(`获取 config.json 失败:HTTP ${resp.status}`);
  }
  const config = await resp.json();
  return {
    config,
    source: { kind: 'huggingface', repoId, revision, verified: false, gated },
  };
}
