# 🧊 LLM Arch Viz — 任意开源 LLM 架构 3D 可视化

输入 HuggingFace repo id(或粘贴 config.json),在浏览器里生成交互式 3D 架构可视化:
GQA / MoE / MLA 结构、逐张量 shape 与参数量、折叠/展开、数据流动画、多格式导出。
灵感来自 [bbycroft.net/llm](https://bbycroft.net/llm)(完全独立实现,零代码复用)。

**零依赖、零构建**:纯原生 ES Modules + 手写 WebGL2,没有 npm、没有打包步骤——
整个文件夹本身就是可部署产物。

## 功能

- **连接 HuggingFace**:输入 repo id → 自动拉取 config.json(gated 模型可填 HF token,仅存本机)
- **适配器注册表**:llama 系(mistral/qwen2/qwen3/gemma/phi…)、qwen2/3-MoE、Mixtral、DeepSeek-V2/V3(MLA)、GPT-2;未知 model_type 走通用兜底解析(UI 明示降级)
- **重复结构去重**:N 个相同层折叠为 "×N" 聚合体,点击懒展开;MoE 专家默认聚合、点击展开专家网格;单 InstancedDraw 渲染(全场景 1 个 draw call)
- **能力分级(T0–T3)**:按模型规模自动分级——教学级全功能动画;15B–100B 仅单层聚焦动画;100B+ 完全静态(可在 `src/parser/ir.js` 调阈值)
- **参数预算**:总参/激活逐块统计(与官方数字对齐:Qwen3-30B-A3B → 30.53B/3.35B,DeepSeek-V3 → 671B/37.5B)
- **2D 视图**:hfviewer 式分层图,移动端默认,矢量可导出
- **导出**:PNG 截图(2×)/ 海报 PNG(含参数卡与来源水印)/ SVG / GLB 3D 模型(可进 Blender/PPT)/ `.llmarch` 工程文件
- **非开源模型通道**:「粘贴配置」手动编辑(带模板)、拖入本地 config.json / `.llmarch`——全部本地解析,零上传
- **`.llmarch` v1**:HF config 的超集(config 原样内嵌 + 来源声明 + 视角/展开状态),裸 config.json 也能直接导入

## 本地运行

任何静态文件服务器都行(ES Modules 不支持 file:// 直开):

```bash
cd llm-arch-viz
python3 -m http.server 8000        # 或:npx serve .
# 打开 http://localhost:8000
```

macOS 用户可直接双击 `serve.command`。

## 部署(整个文件夹就是产物)

- **Cloudflare Pages**:创建项目 → 直接上传本文件夹(或连 git 仓库,构建命令留空,输出目录 `/`)
- **GitHub Pages**:推到仓库 → Settings → Pages → Deploy from branch(根目录)
- **Vercel / Netlify**:导入,Framework 选 Other,无构建命令,输出目录 `.`
- **Docker/自托管**:`docker run -v $(pwd):/usr/share/nginx/html:ro -p 8080:80 nginx:alpine`

支持分享链接:`https://你的域名/?repo=Qwen/Qwen3-30B-A3B` 或 `?sample=deepseek-v3`。

> 注:HF 直连依赖 huggingface.co 的 CORS(config.json 为普通 git 文件,浏览器可直取)。
> 若未来受限,可部署一个转发 Range/CORS 的 Cloudflare Worker,把 `src/hf.js` 的
> `proxyBase` 指过去(接口已预留)。

## 测试

```bash
node tests/parser.test.mjs      # 6 个架构家族的参数量对齐官方数字(26 项断言)
node tests/layout.test.mjs      # 布局 SoA / .llmarch round-trip / SVG(47 项)
python3 tests/browser_test.py   # 需要 playwright+chromium:端到端冒烟(渲染/切换/导出下载)
python3 tests/browser_experts_test.py  # 专家网格 + DeepSeek 全展开压力测试(16k 实例)
```

## 目录

```
index.html            应用外壳(部署入口)
styles.css
src/main.js           UI 状态机 / 交互 / 导入导出
src/parser/adapters.js  config → 结构描述(适配器注册表)
src/parser/ir.js        ArchGraph IR + 参数量 + T0-T3 分级
src/layout.js           IR + 折叠状态 → 实例 SoA(原型×计数,懒展开)
src/gl/renderer.js      手写 WebGL2 实例化渲染器(单 draw call,按需渲染)
src/gl/camera.js        轨道相机   src/gl/pick.js  AABB 拾取   src/gl/mat4.js  数学
src/viz2d.js            2D SVG 渲染器
src/llmarch.js          .llmarch v1 读写   src/hf.js  HF Hub 连接
src/export.js           PNG/海报/GLB 导出   src/samples.js  内置示例
tests/                  Node 单测 + Playwright 冒烟
```

## 许可

MIT(见 LICENSE)。视觉风格灵感来自 bbycroft.net/llm,代码为独立实现。
