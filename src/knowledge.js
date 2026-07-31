// Knowledge base (FR-E1): per-component math + intuition + references.
// Keys are attached to IR segments/tensors via knowledgeKeyFor() in ir.js.
// Formula ASTs are rendered by formula.js. Content written in Chinese (en reserved).

export const KNOWLEDGE = {
  embedding: {
    title: 'Token Embedding 词元嵌入',
    formula: { r: [{ b: 'h' }, { sub: ['', '0'] }, ' = ', { b: 'W' }, { sub: [{ text: '' }, 'emb'] }, '[', { v: 'id' }, ']'] },
    intuition: '把离散的 token id 变成连续向量:查表取出第 id 行。这是模型「理解」文字的第一步——每个词从此有了可以做数学运算的坐标。',
    details: '嵌入矩阵形状 [vocab, hidden],每行是一个 token 的向量表示。训练中相似语义的 token 会被推向相近的方向。它常是模型里最大的单一矩阵之一,和 LM Head 可以共享权重(tied)。',
    refs: [{ t: 'Attention Is All You Need (2017)', u: 'https://arxiv.org/abs/1706.03762' }],
  },
  pos_embedding_learned: {
    title: 'Learned Positional Embedding 可学习位置嵌入',
    formula: { r: [{ b: 'h' }, { sub: ['', '0'] }, ' = ', { b: 'W' }, { sub: ['', 'te'] }, '[', { v: 'id' }, '] + ', { b: 'W' }, { sub: ['', 'pe'] }, '[', { v: 't' }, ']'] },
    intuition: '注意力本身不知道词的先后顺序,GPT-2 直接为每个位置学一个向量加进去,让模型分得清「狗咬人」和「人咬狗」。',
    details: '形状 [n_positions, hidden],上限即最大上下文长度。现代模型多改用 RoPE 等相对位置方案以支持更长上下文外推。',
    refs: [{ t: 'GPT-2 (2019)', u: 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf' }],
  },
  rope: {
    title: 'RoPE 旋转位置编码',
    formula: { r: [{ func: ['RoPE', { r: [{ b: 'q' }, ', ', { v: 't' }] }] }, ' = ', { b: 'R' }, { sub: ['', { r: ['Θ,', { v: 't' }] }] }, { b: 'q' }] },
    intuition: '不再「加」位置向量,而是按位置把 Q/K 向量成对维度旋转一个角度——两个词的注意力得分只取决于它们的相对距离,天然适合外推到更长上下文。',
    details: '每对维度以不同频率旋转,频率由 θ(rope_theta)决定:θ 越大,低频分量越多,可编码的距离越远。把 θ 从 1e4 调到 1e7 是长上下文模型的常见配方(配合 NTK/YaRN 等缩放)。',
    refs: [{ t: 'RoFormer (2021)', u: 'https://arxiv.org/abs/2104.09864' }],
  },
  rmsnorm: {
    title: 'RMSNorm 均方根归一化',
    formula: { r: [{ v: 'y' }, ' = ', { frac: [{ v: 'x' }, { sqrt: { r: [{ func: ['E', { sup: [{ v: 'x' }, '2'] }] }, ' + ε'] } }] }, ' · ', { v: 'γ' }] },
    intuition: '把每个 token 的向量长度拉回统一尺度,防止数值在几十层里越滚越大。比 LayerNorm 少了减均值和 β,更省也几乎同效——所以 Llama 之后成了默认。',
    details: '逐 token 独立进行:除以向量的均方根,再乘可学习的逐维缩放 γ。ε(rms_norm_eps)防止除零。放在每个子层之前(Pre-Norm)是现代 decoder 的标准位置,训练更稳定。',
    refs: [{ t: 'RMSNorm (2019)', u: 'https://arxiv.org/abs/1910.07467' }],
  },
  layernorm: {
    title: 'LayerNorm 层归一化',
    formula: { r: [{ v: 'y' }, ' = ', { frac: [{ r: [{ v: 'x' }, ' − ', { func: ['E', { v: 'x' }] }] }, { sqrt: { r: [{ func: ['Var', { v: 'x' }] }, ' + ε'] } }] }, ' · ', { v: 'γ' }, ' + ', { v: 'β' }] },
    intuition: '对每个 token 的向量做「减均值、除标准差」,再用可学习的 γ/β 拉回合适的分布——bbycroft 演示里那两根细条就是 γ 和 β。',
    details: 'GPT-2 时代的标准归一化。与 RMSNorm 相比多了中心化(−E[x])和偏置 β。ε 防止除零(典型 1e-5)。',
    refs: [{ t: 'Layer Normalization (2016)', u: 'https://arxiv.org/abs/1607.06450' }],
  },
  residual: {
    title: 'Residual Connection 残差连接',
    formula: { r: [{ b: 'h' }, { sub: ['', { r: [{ v: 'l' }, '+1'] }] }, ' = ', { b: 'h' }, { sub: ['', { v: 'l' }] }, ' + ', { func: ['F', { r: [{ func: ['Norm', { b: 'h' }] }] }] }] },
    intuition: '每个子层只是在主干上「加一点修正」,信息与梯度沿这条高速公路直通首尾——这是能把网络堆到几十层的关键。可视化中贯穿上下的中轴就是残差流。',
    details: '残差流(residual stream)是理解 Transformer 的核心心智模型:attention 和 MLP 都从流中读取、处理、再写回。Pre-Norm 结构(先归一化再进子层)让深层训练稳定。',
    refs: [{ t: 'ResNet (2015)', u: 'https://arxiv.org/abs/1512.03385' }],
  },
  attention_core: {
    title: 'Scaled Dot-Product Attention 缩放点积注意力',
    formula: { r: [{ func: ['softmax', { frac: [{ r: [{ b: 'Q' }, { b: 'K' }, { sup: ['', { text: 'T' }] }] }, { sqrt: { v: 'd' } }] }] }, { b: 'V' }] },
    intuition: '每个 token 用自己的 Q 去和之前所有 token 的 K 比相似度,按得分加权取回它们的 V——这是模型「回头看上文」的机制,也是 Transformer 的心脏。',
    details: '除以 √d 防止内积随维度变大导致 softmax 饱和;因果掩码 M 把未来位置设为 −∞,保证只能看过去。每个 head 在 head_dim 维的子空间里独立做一遍,不同 head 学会关注不同关系(语法、指代、位置…)。',
    refs: [{ t: 'Attention Is All You Need (2017)', u: 'https://arxiv.org/abs/1706.03762' }],
  },
  causal_mask: {
    title: 'Causal Mask 因果掩码',
    formula: { r: [{ v: 'M' }, { sub: ['', { r: [{ v: 'i' }, { v: 'j' }] }] }, ' = 0 if ', { v: 'j' }, ' ≤ ', { v: 'i' }, ',  −∞ otherwise'] },
    intuition: '训练时让第 i 个词只能看见前 i 个词,否则「预测下一个词」就成了抄答案。',
    details: '实现上是在 softmax 之前把上三角位置加 −∞。推理时配合 KV cache,新 token 天然只看得到历史。',
    refs: [{ t: 'GPT (2018)', u: 'https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf' }],
  },
  gqa: {
    title: 'GQA 分组查询注意力',
    formula: { r: [{ v: 'g' }, ' = ', { frac: [{ text: 'Q heads' }, { text: 'KV heads' }] }, { text: '  个 Q 头共享一组 K/V' }] },
    intuition: '32 个 Q 头没必要各配一份 K/V——让每 8 个 Q 头共用一组,KV cache 直接缩小 8 倍,推理显存与带宽大幅下降,质量几乎无损。',
    details: '介于 MHA(每头独享)与 MQA(全部共享一组)之间。KV cache 大小 ∝ kv_heads × head_dim × 层数 × 上下文长度,GQA 是长上下文推理的关键优化。可视化里 K/V 投影比 Q 窄,就是这个比例。',
    refs: [{ t: 'GQA (2023)', u: 'https://arxiv.org/abs/2305.13245' }],
  },
  mqa: {
    title: 'MQA 多查询注意力',
    formula: { r: [{ text: 'KV heads' }, ' = 1'] },
    intuition: '所有 Q 头共享唯一一组 K/V,KV cache 最小化——极致省显存,代价是质量略降。',
    details: 'GQA 的极端情形。',
    refs: [{ t: 'MQA (2019)', u: 'https://arxiv.org/abs/1911.02150' }],
  },
  mla: {
    title: 'MLA 多头潜在注意力',
    formula: { r: [{ b: 'c' }, { sub: ['', { text: 'KV' }] }, ' = ', { b: 'W' }, { sub: ['', { text: 'DKV' }] }, { b: 'h' }, ',   ', { b: 'k' }, ',', { b: 'v' }, ' = ', { b: 'W' }, { sub: ['', { text: 'UK,UV' }] }, { b: 'c' }, { sub: ['', { text: 'KV' }] }] },
    intuition: 'DeepSeek 的招:不缓存每个头的 K/V,只缓存一个低秩压缩向量 c(kv_lora_rank 维),用时再解压——KV cache 比 GQA 还小一个量级,同时保留全部 head 的表达力。',
    details: 'K/V 经低秩瓶颈(kv_a_proj → kv_b_proj)重建;Q 也可走低秩(q_lora_rank)。RoPE 部分单独走一条小通道(qk_rope_head_dim)以兼容旋转编码。可视化中 kv_a/kv_b 两块窄高矩阵就是这个瓶颈。',
    refs: [{ t: 'DeepSeek-V2 (2024)', u: 'https://arxiv.org/abs/2405.04434' }],
  },
  qk_norm: {
    title: 'QK-Norm(逐头归一化)',
    formula: { r: [{ b: 'q' }, ' ← ', { func: ['RMSNorm', { b: 'q' }] }, ',  ', { b: 'k' }, ' ← ', { func: ['RMSNorm', { b: 'k' }] }] },
    intuition: '在算注意力得分前把每个 head 的 q、k 各自归一化,防止个别 head 的数值爆炸导致注意力「烧穿」——Qwen3、Gemma 等新模型的稳定性配方。',
    details: '对每个 head 的 head_dim 维向量做 RMSNorm,参数只有 head_dim 个,成本可忽略,却显著改善大模型训练稳定性。',
    refs: [{ t: 'Scaling ViT (2023, QK-Norm)', u: 'https://arxiv.org/abs/2302.05442' }],
  },
  swiglu: {
    title: 'SwiGLU 门控前馈网络',
    formula: { r: [{ v: 'y' }, ' = ', { b: 'W' }, { sub: ['', { text: 'down' }] }, '[', { func: ['SiLU', { r: [{ b: 'W' }, { sub: ['', { text: 'gate' }] }, { v: 'x' }] }] }, ' ⊙ ', { b: 'W' }, { sub: ['', { text: 'up' }] }, { v: 'x' }, ']'] },
    intuition: 'MLP 是模型存「知识」的地方。SwiGLU 用一条门控通路(gate)逐维决定另一条通路(up)的信息放行多少,比朴素 ReLU MLP 更高效——同算力下效果更好。',
    details: '三个矩阵:gate/up 把 hidden 升到 intermediate 维,逐元素相乘后 down 投回。注意力负责「搬运信息」,MLP 负责「加工与记忆」——可解释性研究发现事实性知识多存于 MLP 权重中。',
    refs: [{ t: 'GLU Variants (2020)', u: 'https://arxiv.org/abs/2002.05202' }],
  },
  gelu_mlp: {
    title: 'MLP(GELU 前馈网络)',
    formula: { r: [{ v: 'y' }, ' = ', { b: 'W' }, { sub: ['', '2'] }, ' · ', { func: ['GELU', { r: [{ b: 'W' }, { sub: ['', '1'] }, { v: 'x' }] }] }] },
    intuition: '把向量升到 4 倍宽、过非线性、再压回来——经典两层前馈,GPT-2 的知识仓库。',
    details: 'GELU 是平滑版 ReLU。现代模型多已换成 SwiGLU。',
    refs: [{ t: 'GELU (2016)', u: 'https://arxiv.org/abs/1606.08415' }],
  },
  moe_router: {
    title: 'MoE Router 专家路由',
    formula: { r: [{ v: 'w' }, ' = ', { func: ['softmax', { r: [{ func: ['topk', { r: [{ b: 'W' }, { sub: ['', { v: 'g' }] }, { v: 'x' }] }] }] }] }] },
    intuition: '一个小线性层给每个 token 打分,只挑得分最高的 k 个专家干活——总参数可以巨大(知识容量),每个 token 只付 k/N 的计算(推理便宜)。这就是「30B 总参、3B 激活」的秘密。',
    details: '得分 softmax 后取 top-k;norm_topk_prob=true 时把选中的 k 个权重再归一化。训练需负载均衡辅助损失(router_aux_loss_coef)防止「明星专家」垄断。路由是逐 token 逐层独立的——同一句话的不同词会走不同专家。',
    refs: [{ t: 'Switch Transformer (2021)', u: 'https://arxiv.org/abs/2101.03961' }, { t: 'Mixtral (2024)', u: 'https://arxiv.org/abs/2401.04088' }],
  },
  moe_expert: {
    title: 'Expert 专家(小型 SwiGLU)',
    formula: { r: [{ v: 'y' }, ' = ', { sum: { below: { r: [{ v: 'i' }, '∈topk'] }, body: { r: [{ v: 'w' }, { sub: ['', { v: 'i' }] }, ' · ', { func: ['E', { sub: ['', { v: 'i' }] }] }, '(', { v: 'x' }, ')'] } } }] },
    intuition: '每个专家就是一个窄版 SwiGLU MLP(moe_intermediate_size 通常远小于 dense 的 intermediate)。被选中的 k 个专家的输出按路由权重加权求和。',
    details: '专家间不共享参数,训练中会自发分化(有的偏代码、有的偏多语言)。可视化的专家网格里,每个格子都是 gate/up/down 三个矩阵。',
    refs: [{ t: 'Mixtral (2024)', u: 'https://arxiv.org/abs/2401.04088' }],
  },
  shared_expert: {
    title: 'Shared Expert 共享专家',
    formula: { r: [{ v: 'y' }, ' = ', { v: 'y' }, { sub: ['', { text: 'routed' }] }, ' + ', { func: ['E', { sub: ['', { text: 'shared' }] }] }, '(', { v: 'x' }, ')'] },
    intuition: '所有 token 都必经的「通识专家」,负责公共知识;路由专家只管专门知识——DeepSeek/Qwen2-MoE 用它提升稳定性与参数利用率。',
    details: '恒定激活,不参与路由竞争,因此也不吃负载均衡损失。',
    refs: [{ t: 'DeepSeekMoE (2024)', u: 'https://arxiv.org/abs/2401.06066' }],
  },
  lm_head: {
    title: 'LM Head 输出投影',
    formula: { r: [{ b: 'logits' }, ' = ', { b: 'W' }, { sub: ['', { text: 'head' }] }, { b: 'h' }, { sub: ['', { v: 'L' }] }] },
    intuition: '把最后一层的隐向量投回词表空间:每个 token 得到 vocab 个分数,谁高谁就更可能是下一个词。',
    details: '形状 [vocab, hidden]。tie_word_embeddings=true 时与嵌入矩阵共享权重(转置使用),省一份大矩阵。',
    refs: [{ t: 'Weight Tying (2016)', u: 'https://arxiv.org/abs/1608.05859' }],
  },
  softmax_logits: {
    title: 'Softmax 采样分布',
    formula: { r: [{ v: 'p' }, { sub: ['', { v: 'i' }] }, ' = ', { frac: [{ sup: [{ v: 'e' }, { r: [{ v: 'z' }, { sub: ['', { v: 'i' }] }, '/', { v: 'T' }] }] }, { sum: { below: { v: 'j' }, body: { sup: [{ v: 'e' }, { r: [{ v: 'z' }, { sub: ['', { v: 'j' }] }, '/', { v: 'T' }] }] } } }] }] },
    intuition: 'logits 过 softmax 变成概率分布;温度 T 控制「保守还是狂野」:T→0 永远选最大,T 大则更随机。',
    details: '实际采样还配合 top-p / top-k 截断。这一步之后模型输出一个 token,拼回输入,循环往复——自回归生成。',
    refs: [{ t: 'The Curious Case of Neural Text Degeneration (2019)', u: 'https://arxiv.org/abs/1904.09751' }],
  },
  io_input: {
    title: 'Input token ids',
    formula: { r: [{ v: 'x' }, ' ∈ {0…', { v: 'V' }, '−1}', { sup: ['', { v: 'T' }] }] },
    intuition: '一段文字先被 tokenizer 切成整数序列——模型看到的从来不是文字,而是这些 id。',
    details: '词表大小 V 由 tokenizer 决定;常见 BPE/SentencePiece。',
    refs: [{ t: 'BPE (2015)', u: 'https://arxiv.org/abs/1508.07909' }],
  },
  io_logits: {
    title: 'Logits 输出',
    formula: { r: [{ b: 'z' }, ' ∈ ℝ', { sup: ['', { r: [{ v: 'T' }, '×', { v: 'V' }] }] }] },
    intuition: '每个位置对「下一个 token 是谁」的原始打分,尚未归一化。',
    details: '推理时通常只取最后一个位置的 logits 做采样。',
    refs: [],
  },
};

export function getKnowledge(key) {
  return KNOWLEDGE[key] || null;
}
