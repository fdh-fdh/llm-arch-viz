# LLM 架构 3D 可视化网站 — 完整制作计划

> 目标:做一个 bbycroft.net/llm 风格的交互式 3D 网站,输入任意 HuggingFace repo id(如 `Qwen/Qwen3-30B-A3B`),自动解析模型架构并渲染可交互的 3D 可视化,最终打包托管、长期复用。
>
> 编制日期:2026-07-30 · 含两轮独立 agent 工作:竞品调研 + 对抗性可行性审查

---

## 一、结论先行

**可以做,而且市场上确实没有人做过这个组合。** 但两个原始假设需要修正:

1. **"纯前端、无后端"不完全成立** —— HF 正在把大文件存储迁移到 Xet(`cas-bridge.xethub.hf.co`),该端点对浏览器跨域 Range 请求的 CORS 支持目前是坏的/不稳定的。好消息:`config.json` 和 `model.safetensors.index.json` 是普通 git 文件,浏览器直连没问题,且对已知架构家族,**光靠 config 就能推导出全部张量 shape**。方案:主路径纯前端,Day 1 部署一个 Cloudflare Worker 代理做兜底(免费额度足够)。
2. **"任意模型"应改为"主流开源 LLM(safetensors)"** —— transformers 已有 400+ 架构,还有 GGUF-only、trust_remote_code、量化仓库等长尾。诚实的工程目标:12~15 个适配器家族覆盖热门模型绝大多数流量,未知架构降级为"模块树"视图。

独立审查裁决:按原范围可行性 **6/10**,按下述收缩后范围 **8/10**。

---

## 二、竞品调研结论(独立 agent,2026-07-30)

| 产品 | 形式 | 任意 HF 模型? | 3D? | 开源 | 状态 |
|---|---|---|---|---|---|
| [bbycroft llm-viz](https://github.com/bbycroft/llm-viz) | 权重块+推理动画+叙事讲解 | ❌ 硬编码 GPT | ✅ | ⚠️ **无 LICENSE** | 低活跃,5.5k★ |
| [hfviewer.com](https://hfviewer.com)(Embedl)★最接近 | repo id → 分层架构图 | ✅ | ❌ 2D | ❌ 闭源 | 活跃,HF 官博推广 |
| [Netron](https://github.com/lutzroeder/netron) | 计算图查看器 | 半(需下载文件) | ❌ 2D | MIT | 非常活跃 |
| [Model Explorer](https://github.com/google-ai-edge/model-explorer)(Google) | 大图分层可视化 | ❌ 需本地导出 | ❌ 2D | Apache-2.0 | 活跃 |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | 教学式 GPT-2 实时推理 | ❌ 硬编码 | ❌ 2D | MIT | 7.9k★ |
| [TensorSpace](https://github.com/tensorspace-team/tensorspace) | 3D 分层(仅 CNN) | ❌ | ✅ | Apache-2.0 | 2019 后停滞 |

**市场空白**:"任意 HF 模型"阵营全是 2D(hfviewer/Netron/Model Explorer);"3D"阵营全不支持任意模型(llm-viz/TensorSpace/Zetane)。**"repo id → bbycroft 式 3D"这个交叉点没有人占。** 最大威胁是 hfviewer(商业公司,已被 HF 官方博客背书),差异化必须押在 3D 沉浸感 + 数据流动画 + 教学叙事上。

**法律红线**:llm-viz 仓库无 LICENSE = 默认保留所有权利,**不能 fork 或参考其源码**,必须 clean-room 自研(视觉风格/交互范式属思想范畴不受版权保护,风险低;但不读其源码、不复制一字文案、页面标注 "inspired by bbycroft.net/llm")。

---

## 三、总体架构

```
┌────────────────────────── 浏览器(静态 SPA)──────────────────────────┐
│                                                                      │
│  输入 repo id ──► ① 数据获取层 ──► ② 架构 IR 构建 ──► ③ 3D 渲染层     │
│                    @huggingface/hub    适配器注册表      Three.js      │
│                    (config.json /      + 张量名兜底     InstancedMesh │
│                     index.json /       + 参数量计算     + LOD/折叠    │
│                     safetensors头)         │                │         │
│                        │                   ▼                ▼         │
│                        │              IndexedDB 缓存    交互层(拾取/  │
│                        │              (按 commit hash)  标签/游览/动画)│
└────────────────────────┼─────────────────────────────────────────────┘
                         ▼ 直连失败时降级
              Cloudflare Worker 代理(补 CORS / 聚合分片头 / 缓存)
                         ▼
                 huggingface.co + cas-bridge.xethub.hf.co
```

技术选型:**Vite + TypeScript + React + react-three-fiber(Three.js)**;拾取用 three-mesh-bvh 或 GPU 颜色拾取;标签用 troika-three-text(SDF 文字);解析放 Web Worker。不用任何需要服务器的框架 —— 产物是纯静态文件。

---

## 四、HuggingFace 连接的完整方法(核心管线)

### 4.1 数据获取(全部经浏览器 fetch,带代理降级)

| 数据 | 端点 | 说明 |
|---|---|---|
| 模型元信息 | `GET huggingface.co/api/models/{repo}` | model_type、tags、gated 状态、文件列表 |
| 架构配置 | `GET huggingface.co/{repo}/resolve/main/config.json` | 普通 git 文件,CORS ✅,~2KB |
| 分片索引 | `.../resolve/main/model.safetensors.index.json` | 张量名→分片映射(DeepSeek-V3 实测 8.9MB / 163 分片,放 Worker 解析) |
| 张量 shape/dtype(可选校验) | 各分片前 8 字节 + header 的 HTTP Range 请求(`parseSafetensorsMetadata`) | ⚠️ Xet CDN 的 CORS 目前不稳定 → 走代理 |
| GGUF 仓库(V1) | `@huggingface/gguf` 读文件头 | 头部自带架构名/张量表/量化类型 |

**关键设计:config 是 shape 的第一来源。** 对已知家族,`hidden_size / num_attention_heads / num_key_value_heads / head_dim / intermediate_size / num_experts...` 足以推导每个矩阵的形状(本次 Qwen3-MoE 的手工推导就是证明:1.869B 总参 / 736M 激活全部可由 config 算出)。分片头只用于:未知架构兜底 + 参数量交叉校验。这一手把 CORS 风险从 P0 降为 P1。

### 4.2 架构中间表示(IR)——双通道

```ts
interface ArchGraph {
  meta: { repoId, modelType, totalParams, activeParams, dtype, contextLen }
  blocks: Block[]        // embedding / decoder_layer×N / final_norm / lm_head
}
interface Block { kind, repeat, children: TensorNode[] | Block[] }
interface TensorNode { name, shape, dtype, params, role }  // role: q_proj/gate/expert...
```

- **通道 A(语义权威)— config 适配器注册表**:每个 model_type 家族一个适配器,输出带语义的 IR(GQA 比例、MoE 路由 top-k、滑窗、MLA 等)。首批 8 个家族:`llama`(覆盖 llama/mistral/很多衍生)、`qwen2/qwen3`、`qwen2_moe/qwen3_moe`、`mixtral`、`gemma/2/3`、`gpt2/gpt_neox`、`deepseek_v2/v3`(MLA)、`phi/phi3`。V1 扩到 ~15 家族(+mamba/jamba/glm/olmo)。
- **通道 B(存在性权威)— 张量名反推**:从 `model.layers.0.self_attn.q_proj.weight` 之类的命名重建模块树,对未知 model_type 输出"降级模块树"视图并在 UI 明示降级。必须内置量化格式识别(GPTQ 的 `qweight/qzeros/scales`、FP8 的 `weight_scale_inv` 等),否则参数量会算错。
- **冲突仲裁规则**:语义、shape 以通道 A 为准;张量是否存在(如 tied embeddings 时没有独立 lm_head)以通道 B 为准。

### 4.3 3D 布局与渲染

- 张量 = 长方体,尺寸按维度对数缩放;层垂直堆叠;残差流用贯穿的"主干光带"表示。
- **重复层折叠**:默认显示 1 个展开层 + "×N" 堆叠体,点击展开任意层(bbycroft 对 GPT-3 也是这么做的)。
- **MoE 默认聚合**:128/256 个专家渲染成一块"专家阵列"体素(标注 top-k 路由),点击才展开成网格 —— 既是 UX 正确也顺带消灭性能问题。
- 量级验证(审查 agent 实测估算):DeepSeek-V3 全展开 ≈ 5 万实例 / 60 万三角形 / 1-2 个 draw call,桌面 60fps 现实。**瓶颈不在渲染,在拾取和文字标签**(方案已选 three-mesh-bvh + troika SDF + 视野内标签裁剪)。
- 数据流动画:符号化(发光粒子沿残差流/注意力/路由路径流动),**不做真实推理**。浏览器真实推理(transformers.js)砍掉 —— ONNX Runtime Web 不暴露中间激活;V2 再为 2~3 个策划的教学小模型手写 forward(bbycroft 的实际做法)。

### 4.4 交互与叙事

悬停 → 张量名/shape/dtype/参数量;点击 → 展开/聚焦;侧栏 → 参数预算表(像本次 Qwen3-MoE 分析那样自动生成);相机引导游览 + 每家族一套讲解文案(中英)。讲解内容是工作量隐藏大头,也是对 hfviewer 的核心差异化。

---

## 五、里程碑(单人,全职人周;业余时间 ×2.5)

| 阶段 | 交付物 | 工作量 |
|---|---|---|
| **M0 原型(先跑通)** | 硬编码 2 个家族(llama + qwen3_moe),config→IR→3D 静态渲染 + 悬停,桌面端 | 2-3 周 |
| **MVP** | 8 个适配器家族;×N 折叠/展开;MoE 聚合;参数预算面板;Worker 代理兜底 + 直连埋点;IndexedDB 按 commit 缓存;部署上线 | 累计 7-10 周 |
| **V1** | 分片头校验双路径;GGUF;量化识别;~15 家族 + 未知架构降级树;相机游览 + 数据流动画;每家族讲解文案;gated token(引导 fine-grained read-only);移动端降级 | 累计 20-26 周 |
| **V2** | 策划教学模型的真实激活可视化;架构对比视图(两模型并排);可嵌入卡片/分享链接 | +10-15 周 |

> 审查提醒:布局引擎、标签、拾取这三样最容易被低估(MVP 里占 3-4 周);bbycroft 单个模型的叙事打磨了数月。

---

## 六、打包、托管与复用

- **托管**:纯静态产物 → **Cloudflare Pages**(推荐,和 Worker 代理同生态)或 Vercel/GitHub Pages,免费档即可;代理 = 一个 ~100 行的 Cloudflare Worker(转发 + 补 CORS 头 + 聚合分片头 + 按 revision 缓存)。
- **自托管/复用**:提供 Dockerfile(nginx 托管静态文件 + 可选代理容器),`docker run -p 8080:80 llm-arch-viz` 一条命令起;IR schema 版本化,解析器发布为独立 npm 包 `@you/hf-arch-parser`,网站之外可复用(CLI、Jupyter、别人的项目)。
- **缓存复用**:同一 `repo@commit` 的 IR 在代理层 CDN 缓存,全球只解析一次;浏览器端 IndexedDB 二级缓存。
- **域名/成本**:域名 ~$10/年,其余 $0 起步;HF 匿名限流 3000 次/5min/IP,代理层缓存天然规避。

---

## 七、风险清单(独立审查 agent 结论)

| 风险 | 等级 | 缓解 |
|---|---|---|
| Xet CDN 对浏览器 Range 的 CORS 不稳定([issue](https://github.com/huggingface/datasets/issues/7931)) | 🔴→🟡 | config 推 shape 为主路径;Worker 代理 Day 1 上线;埋点直连成功率 |
| "任意模型"覆盖不了 400+ 架构长尾 | 🔴→🟡 | 范围改口径;适配器家族按下载量覆盖;未知架构降级树 + UI 明示 |
| 量化仓库污染张量名兜底、参数量算错 | 🟠 | 解析器内置 GPTQ/AWQ/FP8/bnb 识别 |
| llm-viz 无 LICENSE | 🟡 | clean-room 纪律成文:不读其源码、零文案复制、显式 attribution |
| hfviewer(商业)若加 3D | 🟡 | 押注叙事讲解质量与开源社区;快速出 MVP 占位 |
| transformers.js 拿不到中间激活 | 已裁决 | 通用真实推理砍掉,V2 策划模型手写 forward |
| 超大 index.json(8.9MB)/ 数百 Range 请求 | 🟡 | Worker 线程解析;代理聚合;已知家族不读分片头 |

---

## 八、下一步(建议本周)

1. 建 repo(`llm-arch-viz`),Vite+TS+R3F 脚手架,清空的这个本地文件夹就是项目根。
2. 先写 `hf-arch-parser` 的 llama + qwen3_moe 两个适配器 + IR schema(纯逻辑,可单测,不碰 3D)。
3. 用本次已算好的 Qwen3-MoE 数据做第一个渲染快照(即 M0 的验收标准:输入 repo id,3 秒内出 3D 图,悬停显示 `q_proj [2048×4096]`)。

**参考来源**:[llm-viz](https://github.com/bbycroft/llm-viz) · [hfviewer + HF 官博](https://huggingface.co/blog/embedl/how-to-visualize-any-hugging-face-model) · [safetensors 远程头解析文档](https://huggingface.co/docs/safetensors/main/en/metadata_parsing) · [@huggingface/hub(浏览器可用)](https://huggingface.co/docs/huggingface.js/hub/README) · [cas-bridge CORS issue](https://github.com/huggingface/datasets/issues/7931) · [HF 限流文档](https://huggingface.co/docs/hub/rate-limits) · [transformers v5(400+ 架构)](https://huggingface.co/blog/transformers-v5) · [Netron](https://github.com/lutzroeder/netron) · [Model Explorer](https://github.com/google-ai-edge/model-explorer) · [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
