# 导出功能与 .llmarch 文件格式规范 — LLM 架构可视化网站(计划补充篇 II)

> 主题:① 截图与 3D/2D 模型文件下载;② 定义自有文件格式 `.llmarch`,保留"导入模型 configuration"的通道;③ 给非开源/私有模型一条可视化通道。
>
> 编制日期:2026-07-30

---

## 一、导出功能设计(截图 + 3D/2D 下载)

### 1.1 截图(PNG)

- **实现**:不用 `preserveDrawingBuffer: true`(常驻性能/内存代价)。点击导出时**离屏重渲染一帧**到指定分辨率的 render target,再 `canvas.toBlob()` —— 支持 1×/2×/4× 超采样(海报级),透明背景可选。这与 `frameloop="demand"` 天然兼容:导出本身就是一次按需渲染。
- **两种模式**:
  - **视图截图**:所见即所得(当前相机、展开状态、高亮)。
  - **海报模式**:自动取"全模型正交侧视 + 标题 + 参数预算表 + 图例 + 水印(repo id / 生成网址)"合成一张分享图 —— 这是社交传播的主力格式,列 MVP。
- **2D 视图**:原生就是 SVG/Canvas,导出 **SVG(矢量,可印刷/进论文)+ PNG** 两种,成本极低。

### 1.2 3D 模型下载(GLB)

- **格式选 glTF/GLB**(行业通用,Blender/Windows 3D 查看器/`<model-viewer>`/PowerPoint 都能开)。
- **技术要点(已核实)**:three.js 官方 GLTFExporter **不支持直接导出 InstancedMesh**([issue #23916](https://github.com/mrdoob/three.js/issues/23916));两条路:
  1. **默认路径(兼容优先)**:导出时把实例**烘焙成单个合并 BufferGeometry**(顶点色携带配色),任何查看器都能开;可选 Draco/meshopt 压缩。
  2. **进阶路径(体积优先)**:挂 [takahirox 的 EXT_mesh_gpu_instancing 导出插件](https://github.com/takahirox/three-gltf-extensions),保留实例化(文件极小),但部分查看器不认该扩展 —— 作为"高级选项"。
- **导出范围联动能力分级**(T0–T3):`当前视图(按现折叠状态)/ 选中的单层 / 全展开`;T3 超大模型禁用"全展开导出"(百万顶点级),默认导当前 LOD。
- **V2 彩蛋**:STL 导出 → 3D 打印你的模型架构(合并几何顺手就能出 STL,几乎零成本,传播性强)。

### 1.3 会话/工程导出

- 导出 `.llmarch` 文件(见第二节)= 完整可复现的可视化工程:别人拖进网站就还原你的视角、展开状态和标注。分享链接(URL 参数 / 短链)与之同构。

---

## 二、`.llmarch` 文件格式 v1(核心:配置导入通道)

### 2.1 设计原则

1. **HF config.json 的超集**:原样内嵌 config.json(不改一个字段),已有的适配器管线零改动直接消费 —— "导入 configuration 的通道"因此天然保留:任何人手里只要有一份 config.json(或者手写一份),就能可视化。
2. **纯 JSON、版本化、向前兼容**:扩展名 `.llmarch`(本质 JSON,允许 `.llmarch.json`);`version` 字段 + "未知字段必须忽略"规则;发布 JSON Schema 供校验;解析器随 `@you/hf-arch-parser` npm 包发布,网站之外(CLI/CI/别人的工具)也能读写。
3. **来源与可信度显式声明**:`source.kind` 区分 `huggingface / manual / estimated`,配 `verified` 标记 —— 这是第三节"非开源通道"不误导用户的关键。

### 2.2 格式定义(示例即规范)

```jsonc
{
  "format": "llmarch",
  "version": 1,
  "meta": {
    "name": "Qwen3-MoE 2层实验配置",
    "author": "donghao",
    "created": "2026-07-30T20:00:00Z",
    "notes": "自由文本备注"
  },
  "source": {
    "kind": "huggingface",            // huggingface | manual | estimated
    "repoId": "Qwen/Qwen3-30B-A3B",   // manual/estimated 时可空
    "revision": "abc123…",            // commit hash,保证可复现
    "verified": true                   // 是否经 safetensors/GGUF 头部交叉校验
  },
  "config": { /* HF config.json 原样内嵌,一字不改 */ },
  "extensions": {
    "tensorOverrides": [               // 可选:手工张量表(修正/补充 shape、dtype)
      { "name": "lm_head.weight", "shape": [151936, 2048], "dtype": "bf16" }
    ],
    "unknownArch": {                   // 可选:未知架构兜底通道的输入
      "tensorNames": ["model.layers.0.self_attn.q_proj.weight", "…"]
    },
    "quantization": null               // 可选:gptq | awq | fp8 | gguf:Q4_K_M …
  },
  "view": {                            // 可选:还原可视化状态
    "camera": { "position": [0,5,20], "target": [0,0,0] },
    "expandedLayers": [0, 37],
    "tierOverride": null,              // 手动改能力分级(如强开动画)
    "colorScheme": "default"
  },
  "annotations": [                     // 可选:用户标注,随文件分享
    { "target": "layers.3.self_attn", "text": "这里是 GQA 8:1" }
  ]
}
```

### 2.3 导入通道(统一入口,五种来源)

| 来源 | 方式 | 说明 |
|---|---|---|
| HF repo id | 输入框 | 主路径,自动拉 config |
| `.llmarch` 文件 | 拖拽/打开 | 完整还原工程(含视角/标注) |
| **裸 config.json** | 拖拽/粘贴 JSON | 最低门槛的"配置通道":从任何地方拿到的 config 都能画 |
| 本地 safetensors / GGUF | 拖拽本地文件 | **只用 File API 读文件头几 MB**(safetensors 前 8 字节 + header;GGUF 头自带架构元数据),**权重字节永不上传、永不全量读入内存** |
| 手动编辑器 | 表单/模板 | 见第三节 |

所有入口汇入同一个 IR 构建管线;导入即校验(JSON Schema + 适配器 sanity check),错误给行级提示。

---

## 三、非开源/私有模型的可视化通道

三类"非开源"场景,三条对应通道,共同原则:**一切解析在本地完成(浏览器/Tauri),配置内容零上传、零遥测**。

### 通道 A:手动配置编辑器(闭源模型 / 纸面架构)

- 表单式编辑器 + **模板库**:从最近的开源家族模板起步(llama 系 / qwen 系 / MoE 系 / MLA 系),改 `hidden_size / num_hidden_layers / num_experts…` 等参数,右侧 3D 实时预览,参数预算表实时重算 —— 相当于一个"架构计算器",本身就是独立卖点(教学、面试、论文复现、"如果 Llama 有 1T 参数长什么样"类玩法)。
- 产出即 `.llmarch(source.kind = manual)`,可导出分享。

### 通道 B:本地权重文件(公司内网私有模型)

- 企业里最常见的"非开源"是**自家训练的模型**:有 config.json 和 safetensors,但绝不能上传。拖拽本地文件 → File API 切片读 header → 本地构建 IR → 可视化。Tauri 桌面版更进一步:**完全离线可用**,适合内网合规环境;这也反过来强化了桌面形态的存在理由。

### 通道 C:社区估算库(著名闭源模型)

- 对 GPT-4/Gemini/Claude 这类只有公开传闻的模型,维护一个**社区估算配置库**(`source.kind = estimated, verified = false`),让用户能看"业界推测的 GPT-4 MoE 长什么样"。
- **诚实性硬约束**(不可妥协):估算模型在 3D 视图上常驻"UNVERIFIED / 社区估算"水印,参数预算表标注来源链接;绝不与验证过的 HF 模型混排在同一列表里,避免以讹传讹。

---

## 四、对主计划的修订项

| 里程碑 | 新增内容 | 增量成本 |
|---|---|---|
| **MVP** | PNG 视图截图 + 海报模式;`.llmarch` 定义与导入/导出;裸 config.json 拖拽/粘贴通道 | ~1 周(格式定义要趁 IR 还小时定,越晚越贵) |
| **V1** | GLB 导出(烘焙路径 + Draco);2D SVG/PNG 导出;手动配置编辑器 + 模板库;本地 safetensors/GGUF 拖拽;JSON Schema 发布进 npm 包 | ~2–3 周 |
| **V2** | EXT_mesh_gpu_instancing 进阶导出;STL/3D 打印;社区估算库(含水印机制) | ~1–2 周 |

风险提示:估算库有被断章截图传播的舆论风险 → 水印烧进导出的 PNG/GLB 里(不只是 UI 层);GLB 全展开导出在 T3 禁用;`.llmarch` 一旦发布 v1 就要守向前兼容承诺,extensions 命名空间预留自定义字段。

---

## 五、来源

[GLTFExporter 不支持 InstancedMesh(three.js #23916)](https://github.com/mrdoob/three.js/issues/23916) · [EXT_mesh_gpu_instancing 导出插件(takahirox/three-gltf-extensions)](https://github.com/takahirox/three-gltf-extensions/tree/main/exporters/EXT_mesh_gpu_instancing) · [GLTFLoader 侧的 EXT_mesh_gpu_instancing 支持(#21937)](https://github.com/mrdoob/three.js/issues/21937) · [GLTFExporter 官方文档](https://threejs.org/docs/pages/GLTFExporter.html) · [three.js 论坛:InstancedMesh 与 glTF 导入导出](https://discourse.threejs.org/t/blender-export-instancedmesh-directly/73769) · safetensors/GGUF 头部结构见前篇《制作计划》来源
