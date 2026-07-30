// .llmarch v1 file format — see《导出功能与llmarch文件格式规范》.
// Superset of HF config.json: config embedded verbatim, provenance declared.

export const LLMARCH_VERSION = 1;

export function exportLlmarch({ config, source, view, annotations, name, notes }) {
  return JSON.stringify({
    format: 'llmarch',
    version: LLMARCH_VERSION,
    meta: {
      name: name || source?.repoId || 'model',
      author: '',
      created: new Date().toISOString(),
      notes: notes || '',
      generator: 'llm-arch-viz',
    },
    source: {
      kind: source?.kind || 'manual',
      repoId: source?.repoId || null,
      revision: source?.revision || null,
      verified: !!source?.verified,
    },
    config,
    extensions: {},
    view: view || null,
    annotations: annotations || [],
  }, null, 2);
}

export function parseLlmarch(text) {
  let doc;
  try { doc = JSON.parse(text); } catch { throw new Error('不是合法的 JSON 文件'); }

  // A bare HF config.json is also accepted (the "配置导入通道").
  if (doc.format !== 'llmarch') {
    if (doc.model_type || doc.architectures || doc.hidden_size || doc.n_embd || doc.text_config) {
      return {
        config: doc,
        source: { kind: 'manual', name: doc.model_type ? doc.model_type + ' (config.json)' : 'config.json' },
        view: null, annotations: [],
      };
    }
    throw new Error('既不是 .llmarch 文件也不是 HF config.json');
  }
  if (typeof doc.version !== 'number' || doc.version > LLMARCH_VERSION) {
    throw new Error(`不支持的 .llmarch 版本:${doc.version}(本应用支持 ≤ ${LLMARCH_VERSION})`);
  }
  if (!doc.config || typeof doc.config !== 'object') throw new Error('.llmarch 缺少 config 字段');
  return {
    config: doc.config,
    source: { ...(doc.source || { kind: 'manual' }), name: doc.meta?.name },
    view: doc.view || null,
    annotations: doc.annotations || [],
    meta: doc.meta || {},
  };
}
