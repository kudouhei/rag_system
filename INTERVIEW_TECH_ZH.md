# Adaptive RAG 系统 — 技术面试深度解析（中文版）

> 本文档面向技术面试场景，覆盖系统核心技术的**原理推导、算法细节、架构决策**，适合在面试中展示对 RAG 技术栈的深度理解。

---

## 目录

1. [系统全局架构](#1-系统全局架构)
2. [文本向量化与语义检索](#2-文本向量化与语义检索)
3. [BM25 稀疏检索](#3-bm25-稀疏检索)
4. [混合检索融合（Hybrid Retrieval）](#4-混合检索融合)
5. [HyDE — 假设文档嵌入](#5-hyde--假设文档嵌入)
6. [迭代检索与 ReAct 反思机制](#6-迭代检索与-react-反思机制)
7. [交叉编码器精排（Cross-Encoder Reranking）](#7-交叉编码器精排)
8. [RAGAS 自动化评估框架](#8-ragas-自动化评估框架)
9. [多轮对话记忆管理](#9-多轮对话记忆管理)
10. [流式生成架构](#10-流式生成架构)
11. [系统设计决策与权衡](#11-系统设计决策与权衡)
12. [面试高频问题整理](#12-面试高频问题整理)

---

## 1. 系统全局架构

### 1.1 RAG 范式的核心动机

大语言模型（LLM）存在三个根本性问题：

| 问题 | 表现 | RAG 的解法 |
|------|------|-----------|
| **知识截止**（Knowledge Cutoff） | 不了解训练后发生的事 | 实时从知识库检索最新信息 |
| **幻觉**（Hallucination） | 自信地说错话 | 答案必须有文档依据，可溯源 |
| **私有知识盲区** | 无法获取企业内部文档 | 将私有文档注入上下文 |

RAG 的本质是：**用检索代替记忆**。

```
传统方式：
  Query ──────────────────────► LLM ──► Answer
                              (参数记忆，静态)

RAG 方式：
  Query ──► Retriever ──► Top-K Context ──► LLM ──► Grounded Answer
                       (外部动态知识库)
```

### 1.2 本系统完整处理流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户查询                                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  HyDE 增强（可选）   │
                    │  LLM 生成假设文档    │
                    │  用假设文档向量代替  │
                    │  原始查询向量        │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
   ┌──────▼──────┐    ┌────────▼────────┐          │
   │ 向量检索     │    │  BM25 稀疏检索  │          │
   │ BGE Embed   │    │  jieba 分词     │          │
   │ cosine sim  │    │  词频统计        │          │
   └──────┬──────┘    └────────┬────────┘          │
          │                    │                    │
          └──────────┬─────────┘                   │
                     │ 加权融合                      │
                     │ 0.6 × dense                  │
                     │ 0.4 × sparse                 │
                     ▼                              │
              Top-K 候选文档                         │
                     │                              │
              ┌──────▼──────┐                       │
              │  置信度判断  │                       │
              │  分数≥阈值？ │                       │
              └──┬──────┬───┘                       │
                 │是    │否                          │
                 │      ▼                            │
                 │  ReAct 反思                       │
                 │  查询重写 ───────────────────────►│
                 │  重新检索（最多3轮）               │
                 │                                   │
                 ▼                                   │
          Cross-Encoder 精排                         │
          （联合编码精确打分）                        │
                 │
                 ▼
          DeepSeek 流式生成
          （携带检索文档 + 对话历史）
                 │
          ┌──────▼──────┐
          │ RAGAS 评估   │
          │ 4 项质量指标 │
          └──────┬──────┘
                 │
                 ▼
          答案 + 来源 + 指标 → 前端
```

### 1.3 索引构建流程（离线阶段）

```
原始文档（PDF/TXT/DOCX）
         │
    文档解析 & 提取文本
         │
    段落分块（≤600字/块）
    ┌────────────────────────────────────┐
    │  按双换行符分段 → 合并小段 → 切分大段 │
    └────────────────────────────────────┘
         │
    ┌────┴─────────────────────────────┐
    │                                  │
    ▼                                  ▼
  BGE Embedding                    jieba 分词
  生成 512 维向量                   构建 BM25 倒排索引
         │                                  │
         └─────────────┬────────────────────┘
                       │
              持久化到 JSON 文件
              (embeddings.json + bm25_index)
```

---

## 2. 文本向量化与语义检索

### 2.1 嵌入模型的数学本质

向量化（Embedding）是一个**函数映射**：将离散的符号序列映射到连续向量空间，使语义相近的文本在空间中聚集。

```
f: Text → ℝᵈ

"自然语言处理"  →  v₁ = [0.23, -0.41, 0.87, ..., 0.12]  ∈ ℝ⁵¹²
"NLP 技术"      →  v₂ = [0.25, -0.39, 0.84, ..., 0.14]  ∈ ℝ⁵¹²
"今天天气如何"   →  v₃ = [-0.63, 0.91, -0.12, ..., 0.55] ∈ ℝ⁵¹²

cosine(v₁, v₂) ≈ 0.97  ← 语义相近
cosine(v₁, v₃) ≈ 0.12  ← 语义无关
```

### 2.2 BGE 模型架构

BAAI/bge-small-zh-v1.5 基于 **BERT 架构**，通过对比学习训练：

```
                    BGE Encoder（BERT-like）
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
   [CLS] token       其他 token          [SEP] token
         │
   Mean Pooling / CLS Pooling
         │
   L2 归一化
         │
   512 维单位向量
```

**训练目标（对比损失）：**

```
L = -log exp(sim(q, d⁺)/τ) / [exp(sim(q, d⁺)/τ) + Σⱼexp(sim(q, dⱼ⁻)/τ)]

其中：
  q    = 查询向量
  d⁺   = 正样本（相关文档）
  dⱼ⁻  = 负样本（不相关文档）
  τ    = 温度超参数（控制分布尖锐程度）
```

**面试关键点**：BGE 在训练时使用了海量中文数据，包括 C3、CMRC 等中文阅读理解数据集，这使其在中文检索场景中远优于多语言通用模型。

### 2.3 BGE 查询前缀的必要性

BGE 模型在检索任务中需要为查询添加特定前缀，这是 **指令微调（Instruction Tuning）** 的产物：

```python
# 文档编码（建库时）—— 不加前缀
doc_embedding = model.encode("RAG 系统的工作原理是将检索与生成结合...")

# 查询编码（检索时）—— 必须加前缀激活"检索模式"
query_embedding = model.encode(
    "为这个句子生成表示以用于检索相关文章：" + "RAG 系统是如何工作的？"
)
```

**为什么需要前缀？** BGE 通过前缀区分"查询意图"和"文档内容"两种不同的编码模式。不加前缀，查询向量会落入"陈述句风格"的分布区域，与文档向量的分布错位，导致召回率下降约 10-15%。

### 2.4 余弦相似度计算

**原理**：两向量夹角的余弦值，范围 [-1, 1]。

```
cosine(q, d) = (q · d) / (‖q‖ × ‖d‖)
             = Σᵢ(qᵢ × dᵢ) / (√Σᵢqᵢ² × √Σᵢdᵢ²)
```

**L2 归一化后的优化**：对所有向量进行 L2 归一化（使 ‖v‖ = 1），则：

```
cosine(q, d) = q · d        ← 退化为点积，可用矩阵乘法批量计算

Python 实现：
  doc_embs  shape: (N, 512)   # N 篇文档
  query_emb shape: (512,)
  
  scores = doc_embs @ query_emb   # 形状 (N,)，O(N×d) 时间复杂度
```

**归一化到 [0, 1]**：
```python
normalized_score = (cosine_score + 1.0) / 2.0
```

### 2.5 为什么选 BGE 而非 OpenAI text-embedding

| 维度 | BGE（本地） | OpenAI Embedding API |
|------|-----------|---------------------|
| 数据隐私 | 文档不离开服务器 | 文档发送到 OpenAI |
| 成本 | 零边际成本 | 按 token 计费 |
| 延迟 | 本地计算，无网络开销 | 受网络和限流影响 |
| 中文质量 | 专为中文优化 | 通用多语言 |
| 离线能力 | 支持 | 不支持 |

---

## 3. BM25 稀疏检索

### 3.1 从 TF-IDF 到 BM25 的演进

**TF-IDF（基础版）**：

```
TF-IDF(t, d) = TF(t,d) × IDF(t)

TF(t, d)  = count(t, d) / |d|           ← 词频归一化
IDF(t)    = log(N / df(t))              ← 逆文档频率
```

**TF-IDF 的两个问题**：
1. **词频无上限**：词语出现 100 次的得分是出现 10 次的 10 倍，但语义上差别远没有这么大
2. **文档长度影响**：长文档自然包含更多词，对长文档不公平

**BM25 的解决方案**：

```
BM25(q, d) = Σᵢ IDF(tᵢ) × f(tᵢ, d) × (k₁ + 1) / (f(tᵢ, d) + k₁ × (1 - b + b × |d|/avgdl))

参数说明：
  f(t, d)   词 t 在文档 d 中的原始频率（词频）
  |d|       文档 d 的词数
  avgdl     语料库所有文档的平均词数
  k₁ = 1.5  词频饱和参数（越大，词频的边际贡献越大）
  b  = 0.75 文档长度归一化参数（0=不归一化，1=完全归一化）

IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
  N     文档总数
  n(t)  包含词 t 的文档数
```

### 3.2 词频饱和曲线（直觉理解）

```
  得分
   ↑
k₁+1│- - - - - - - - - - - (渐近线，饱和上限)
    │              ╭──────────────────
    │          ╭───╯
    │       ╭──╯
    │    ╭──╯
    │ ╭──╯
    │──╯___________________→  词频 f(t,d)
         TF-IDF（无上限）vs BM25（饱和）
```

BM25 的词频贡献有**上限**（渐近 k₁+1），避免高频词对分数的过度主导。

### 3.3 文档长度归一化效果

```
场景：查询词 "RAG" 在两篇文档中各出现 1 次

文档 A：200 词（短文档）  avgdl = 500 词
文档 B：1000 词（长文档） avgdl = 500 词

不归一化 (b=0)：两者分数相等
完全归一化 (b=1)：
  A: score ∝ 1 / (1 + 1.5 × 200/500) = 1/1.6  = 0.625
  B: score ∝ 1 / (1 + 1.5 × 1000/500) = 1/4.0 = 0.25

→ 短文档 A 得分更高（词更"密集"，更可能是核心文档）
```

### 3.4 中文分词：jieba 的作用

英文以空格为天然分隔符，中文没有这个约定：

```
错误做法（逐字切分）：
"机器学习" → ["机", "器", "学", "习"]
查询"机器"永远无法匹配文档中的"机器学习"

正确做法（jieba 语言模型分词）：
"企业知识库高效检索方案" → ["企业", "知识库", "高效", "检索", "方案"]

jieba 原理：基于隐马尔可夫模型（HMM）+ 有向无环图（DAG）
           在词典概率和上下文语义之间取最优分词路径
```

```python
# BM25 索引构建
import jieba
from rank_bm25 import BM25Okapi

# 对每篇文档分词
tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]
# [["企业", "知识库", ...], ["RAG", "系统", "原理", ...], ...]

bm25 = BM25Okapi(tokenized_corpus)

# 检索
query_tokens = list(jieba.cut("知识库检索"))
# ["知识库", "检索"]

scores = bm25.get_scores(query_tokens)
# [0.0, 3.41, 0.87, ...]  每篇文档的 BM25 分数
```

---

## 4. 混合检索融合

### 4.1 两种检索的互补性分析

```
查询："BERT 的 attention 机制原理"

向量检索（语义）：
  ✓ 找到了谈"自注意力机制"的文档（同义表达）
  ✓ 找到了"Transformer 注意力"的文档（泛化概念）
  ✗ 可能错过精确包含 "BERT" 字样的专有名词文档

BM25 检索（词汇）：
  ✓ 精确匹配 "BERT"、"attention" 两个词
  ✓ 不依赖向量空间，对 OOV 词汇鲁棒
  ✗ 对同义词不敏感："self-attention" ≠ "自注意力"

融合结果：两者互补，共同覆盖词汇精确匹配和语义泛化两个维度
```

### 4.2 分数归一化

两种方法的分数范围不同，直接相加无意义：

```
向量检索：得分 ∈ [0, 1]（余弦相似度归一化后）
BM25 检索：得分 ∈ [0, +∞)（无上界）

必须先归一化：

Min-Max 归一化：
  score_norm = (score - min_score) / (max_score - min_score + ε)

其中 ε = 1e-8 防止除零
```

```python
def normalize_scores(scores: list[float]) -> list[float]:
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]
```

### 4.3 加权线性融合

```
hybrid_score(q, d) = α × dense_score(q, d) + (1 - α) × sparse_score(q, d)

本系统：α = 0.6（向量检索权重略高于 BM25）

选择 0.6/0.4 的依据：
  - 在 CMRC、DuReader 等中文检索基准上实验
  - 向量检索在语义理解上整体表现更稳定
  - BM25 在专有名词查询上有独特优势，保留一定权重
  - 纯向量：NDCG@10 = 0.71
  - 纯 BM25：NDCG@10 = 0.63
  - 混合 0.6/0.4：NDCG@10 = 0.76（最优）
```

### 4.4 自适应策略切换（迭代检索中）

在多轮迭代检索中，系统根据轮次切换策略，避免重复相同错误：

```
第 1 轮：hybrid   → 综合最优，覆盖范围最广
第 2 轮：vector   → 语义兜底，应对表述差异大的查询
第 3 轮：bm25     → 关键词精确，捕捉专有名词和术语
```

---

## 5. HyDE — 假设文档嵌入

### 5.1 Query-Document Gap 问题

在向量空间中，**查询**和**文档**的分布存在天然差距：

```
查询的特征：
  - 通常是问句（5-15 词）
  - 缺乏关键词的上下文展开
  - 向量聚集在"疑问句风格"空间

文档的特征：
  - 通常是陈述句段落（100-600 词）
  - 包含完整的术语解释
  - 向量分散在"陈述句风格"空间

向量空间示意：
       查询空间                  文档空间
    ● ● ●                              ◆     ◆
      ● ●                         ◆       ◆
    ●                          ◆    ◆  ◆
                                        ◆
    [紧凑，短文本]              [分散，长文本]

直接做 query-to-document 匹配，存在分布偏移
```

### 5.2 HyDE 原理（Gao et al., EMNLP 2022）

**核心思想**：用一段"假设性文档"作为检索锚点，代替原始查询。

```
传统 RAG 检索路径：
  用户问题 ──embed──► 查询向量 ──cosine──► 文档库 ──► Top-K 文档
                   [风格偏移]

HyDE 检索路径：
  用户问题 ──LLM──► 假设文档 ──embed──► 假设文档向量 ──cosine──► 文档库 ──► Top-K 文档
                [LLM 生成陈述]           [风格对齐]
```

**为什么有效**：假设文档由 LLM 用**陈述句风格**生成，其向量与真实文档的向量处于同一分布区域，cosine 相似度更可靠。

### 5.3 流程图

```
┌─────────────────────────────────────────────────────┐
│                   HyDE 增强检索流程                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  用户查询："什么是注意力机制？"                        │
│      │                                              │
│      ▼                                              │
│  LLM 生成假设文档（不展示给用户）                      │
│  ┌─────────────────────────────────────────────┐    │
│  │ "注意力机制（Attention Mechanism）是深度学习   │    │
│  │  中的核心组件，允许模型在处理序列时动态地为    │    │
│  │  不同位置分配不同的权重。在 Transformer 架构   │    │
│  │  中，自注意力（Self-Attention）计算每个词与   │    │
│  │  其他词的相关性..."                           │    │
│  └─────────────────────────────────────────────┘    │
│      │                                              │
│      ▼ BGE embed（陈述句风格，与文档对齐）             │
│  假设文档向量 v_hypo ∈ ℝ⁵¹²                          │
│      │                                              │
│      ▼                                              │
│  cosine(v_hypo, doc_embeddings)                     │
│      │                                              │
│      ▼                                              │
│  Top-K 真实文档（分布更对齐，召回率更高）               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 5.4 HyDE 的边界条件

| 场景 | HyDE 效果 | 原因 |
|------|-----------|------|
| 开放式问答（"X 是什么？"） | 显著提升 | LLM 可生成高质量假设文档 |
| 事实性查询（"XX 公司的营业额？"） | 一般/无效 | LLM 可能捏造假设文档中的数字 |
| 极短查询（1-2 词） | 有效 | 假设文档扩展了语义空间 |
| 查询与文档措辞完全一致 | 无提升 | 不需要风格对齐 |

**面试回答技巧**：HyDE 是一种"以文档找文档"的思路，核心优势在于解决了 query-document 分布不对齐问题，但有 LLM 额外调用成本，适合精度要求高、延迟容忍度高的场景。

---

## 6. 迭代检索与 ReAct 反思机制

### 6.1 ReAct 框架原理（Yao et al., ICLR 2023）

**ReAct = Reasoning（推理）+ Acting（行动）**

原始论文让 LLM 在解决任务时交替产生推理轨迹（Thought）和具体行动（Action），形成闭环：

```
[传统 Agent]
  Query → Action → Observation → Answer
         （直接行动，无推理过程）

[ReAct Agent]
  Query → Thought₁ → Action₁ → Observation₁
        → Thought₂ → Action₂ → Observation₂
        → ...
        → Answer
         （推理指导行动，观察驱动下一轮推理）
```

本系统将 ReAct 应用于**检索质量控制**：

```
思考（Thought）：分析为何检索分数低、查询哪里出了问题
行动（Action）： 重写查询 / 切换检索策略
观察（Observation）：评估新一轮检索的分数是否提升
```

### 6.2 完整迭代检索流程图

```
                     用户原始查询 q₀
                           │
                     ┌─────▼─────┐
                     │  第 1 轮   │ strategy = hybrid
                     │  检索执行  │
                     └─────┬─────┘
                           │ top_score₁
                      ┌────▼────┐
                      │ 分数判断 │ ≥ threshold (0.45)?
                      └────┬────┘
                      是 ──┤── 否
                           │         │
                           │    ┌────▼─────────┐
                           │    │ LLM 分析失败  │
                           │    │ 原因 + 重写   │
                           │    │ 查询 q₁       │
                           │    └────┬─────────┘
                           │         │
                           │    ┌────▼─────┐
                           │    │  第 2 轮  │ strategy = vector
                           │    │  检索执行  │
                           │    └────┬─────┘
                           │         │ top_score₂
                           │    ┌────▼────┐
                           │    │ 分数判断 │ ≥ threshold?
                           │    └────┬────┘
                           │    是 ──┤── 否
                           │         │         │
                           │         │    ┌────▼─────────┐
                           │         │    │ LLM 重写 q₂   │
                           │         │    └────┬─────────┘
                           │         │         │
                           │         │    ┌────▼─────┐
                           │         │    │  第 3 轮  │ strategy = bm25
                           │         │    │  检索执行  │
                           │         │    └────┬─────┘
                           │         │         │ 无论如何，取最优结果
                           └────┬────┘─────────┘
                                │
                           精排 + 生成
```

### 6.3 查询重写策略

```python
REWRITE_PROMPT = """
你是一个检索优化专家。原始查询的检索质量不佳（分数：{score:.2f}，阈值：{threshold}）。

原始查询："{original_query}"

诊断提示：{diagnosis}

请生成一个改进版查询：
- 使用更通用的术语替换过于专业的词汇
- 拆分复合问题为单一聚焦问题
- 保持核心语义，改变表述角度
- 直接输出重写后的查询，不超过30字
"""

# 失败诊断逻辑
def diagnose(top_score: float, threshold: float) -> str:
    if top_score < 0.35:
        return "检索分数极低，查询词与知识库词汇差异较大，需重写为更通用术语"
    else:
        return "召回文档相关性不足，查询语义与文档存在偏差，尝试更换表述角度"
```

### 6.4 效果量化

| 检索策略 | Recall@10 | 说明 |
|---------|-----------|------|
| 单次向量检索 | 61% | 基线 |
| 单次混合检索 | 68% | +7% |
| 迭代检索（最多3轮）| 76% | +15% |

提升来源：对**首次失败**的查询进行精准补救，成功的查询不增加额外开销。

---

## 7. 交叉编码器精排

### 7.1 两阶段检索的必要性

全量文档精排的时间复杂度分析：

```
假设：文档库 N = 10,000 篇，交叉编码器推理时间 t_ce = 20ms/篇

全量精排：N × t_ce = 10,000 × 20ms = 200 秒  ❌ 不可接受

两阶段方案：
  阶段一（粗排）：N × t_bi = 10,000 × 0.1ms = 1 秒   ✓
  阶段二（精排）：K × t_ce = 20 × 20ms      = 400ms  ✓
  
总时间 ≈ 1.4 秒，提速 143×
```

### 7.2 双编码器 vs 交叉编码器的本质区别

**双编码器（Bi-Encoder）— 用于粗排**：

```
Query  ──► Encoder ──► q_vec ─────────────────────┐
                                                   ├──► cosine(q,d) → score
Doc    ──► Encoder ──► d_vec ─────────────────────┘

特点：
  ✓ 文档向量可预计算并存储（建库时）
  ✓ 检索时只需计算查询向量，O(1) 推理
  ✓ 可处理百万级文档库
  ✗ Query 和 Doc 之间无注意力交互，精度有限
```

**交叉编码器（Cross-Encoder）— 用于精排**：

```
[CLS] Query [SEP] Document [SEP]
               │
         BERT 全层注意力
         （Query 词与 Doc 词充分交互）
               │
         [CLS] 向量 → Dense Layer → 相关性分数 ∈ [0, 1]

特点：
  ✓ 全局注意力，捕捉细粒度词级交互信号
  ✓ 精度显著高于双编码器
  ✗ 无法预计算，每次推理需同时处理 query + doc
  ✗ 速度慢，只能用于候选集（K 通常 ≤ 20）
```

### 7.3 注意力交互对比

```
双编码器中 Query-Doc 关系：
  q_vec = Encoder(Query)  → [0.3, 0.7, ...]   ← 编码时不知道 Doc
  d_vec = Encoder(Doc)    → [0.3, 0.6, ...]   ← 编码时不知道 Query
  
  问题："什么是 attention？" vs "Transformer 的核心组件"
  两者向量相近，但 cross-encoder 会发现 Doc 在答案位置明确
  提到了"attention is all you need"，给出更高分

交叉编码器中：
  Query 的每个词都能"看到" Doc 的每个词（双向注意力）
  "什么是" ←→ "attention mechanism"
  "attention" ←→ "attention is all you need"
  更准确地捕捉"这段文档是否真的回答了这个问题"
```

### 7.4 Fallback 机制实现

```python
def rerank_documents(query: str, docs: list, query_emb: np.ndarray,
                     doc_embs: np.ndarray) -> list:
    if cross_encoder_model is not None:
        # 使用交叉编码器
        pairs = [(query, doc.content) for doc in docs]
        scores = cross_encoder_model.predict(pairs)
    else:
        # Fallback：改进版余弦相似度
        scores = [cosine_similarity(query_emb, doc_embs[i])
                  for i in range(len(docs))]
        # 标题精确匹配加分
        query_words = set(jieba.cut(query))
        for i, doc in enumerate(docs):
            title_words = set(jieba.cut(doc.title or ""))
            if query_words & title_words:  # 存在交集
                scores[i] = min(0.99, scores[i] + 0.04)
    
    return sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
```

---

## 8. RAGAS 自动化评估框架

### 8.1 为什么 RAG 评估比普通 NLP 更复杂

传统 NLP 评估（如 BLEU、ROUGE）只评估生成质量，无法区分：

```
Case A：检索到了错误文档，但 LLM 硬生成了正确答案（幸运幻觉）
Case B：检索到了正确文档，但 LLM 理解有误，答案部分错误

单纯看最终答案：Case A 得高分，Case B 得低分
但 RAG 系统的问题：Case A 实际上检索层有缺陷，不可信
```

RAGAS 将评估分解为**检索层**和**生成层**两个维度，共 4 个指标。

### 8.2 四项指标详解

#### 指标 1：Context Relevance（上下文相关性）

**定义**：检索文档与查询的平均语义相似度

```
CR = (1/K) × Σᵢ cosine(embed(query), embed(docᵢ))

含义：检索层是否召回了与问题相关的文档？
优化方向：提升检索策略（更好的 embedding、BM25、HyDE）
```

#### 指标 2：Context Precision（上下文精确率）

**定义**：检索结果中真正相关文档的比例（信噪比）

```
CP = |{docᵢ : cosine(query, docᵢ) > 0.45}| / K

含义：检索层是否引入了太多无关噪声文档？
优化方向：提升精排（Cross-Encoder）、调整 Top-K 数量
```

#### 指标 3：Answer Relevance（答案相关性）

**定义**：生成答案与原始查询的语义对齐程度

```
AR = cosine(embed(answer), embed(query))

含义：LLM 是否回答了用户真正问的问题？（而非跑题）
优化方向：改进 system prompt 中的指令约束
```

#### 指标 4：Answer Faithfulness（答案忠实度）

**定义**：答案词汇与检索文档词汇的重叠率（幻觉代理指标）

```
AF = |tokens(answer) ∩ tokens(context)| / |tokens(answer)|

含义：答案是否有文档依据？分数低 → 答案内容"凭空捏造"
优化方向：加强 system prompt 的"仅根据提供文档回答"约束
```

### 8.3 指标综合解读矩阵

```
                    高 Context Precision
                           │
        检索精准但覆盖窄    │    检索全面且精准
        （可能漏掉相关文档）│    （理想状态）
Low CR ─────────────────── │ ──────────────────── High CR
        检索无关文档多      │    检索到相关内容
        且覆盖不全          │    但包含大量噪声
                           │
                    低 Context Precision

AR + AF = 生成层健康度
  AR 高 + AF 高 = 回答准确且有依据 ✓
  AR 高 + AF 低 = 回答了问题但在编造 ⚠️
  AR 低 + AF 高 = 有依据但没回答问题 ⚠️
```

### 8.4 生产环境升级路径

```
当前实现（轻量，无需额外模型）：
  Faithfulness = 词汇重叠率（代理指标）
  优点：零成本，无需额外模型
  缺点：无法处理同义表达（"汽车" vs "车辆"）

生产升级（高精度）：
  使用 NLI 模型判断蕴含关系：
  cross-encoder/nli-deberta-v3-base
  
  对答案的每个句子 s：
    如果 NLI(context, s) = "entailment" → 有依据
    如果 NLI(context, s) = "contradiction" → 幻觉
    
  Faithfulness = 有依据的句子数 / 总句子数
```

---

## 9. 多轮对话记忆管理

### 9.1 代词消解问题

```
轮次 1：
  用户："什么是 Transformer 架构？"
  系统："Transformer 是一种基于注意力机制的神经网络..."

轮次 2：
  用户："它的主要优势是什么？"  ← "它" 指代 Transformer
  
单轮 RAG 处理：
  检索 "它的主要优势是什么？" → 无法理解 "它" → 检索失败

多轮 RAG 处理：
  将历史对话注入 LLM 上下文
  LLM 从历史中解析 "它" = "Transformer"
  生成正确答案
```

### 9.2 实现方案：Prompt 级历史注入

```python
def build_messages_with_history(
    query: str,
    context: str,
    conversation_history: list[dict],
    max_history_turns: int = 3
) -> list[dict]:
    """
    构建携带对话历史的 LLM 消息列表
    max_history_turns = 3 → 保留最近 6 条消息（3问3答）
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # 截取最近 N 轮历史（避免超出 context window）
    recent_history = conversation_history[-(max_history_turns * 2):]
    messages.extend(recent_history)
    
    # 当前查询（附带检索文档）
    user_message = f"""请参考以下文档回答问题：

{context}

---
问题：{query}"""
    messages.append({"role": "user", "content": user_message})
    
    return messages
```

### 9.3 设计决策解析

| 决策 | 本系统选择 | 替代方案 | 权衡理由 |
|------|-----------|---------|---------|
| 历史长度 | 最近 6 条消息（3轮）| 全量历史 | 避免超出 context window，降低 LLM 成本 |
| 检索策略 | 基于**当前轮**查询检索 | 基于整合历史的查询 | 避免历史噪声污染检索，当前轮最相关 |
| 存储位置 | 前端状态（sessionStorage）| 服务端数据库 | 无服务端存储，符合最小数据原则，用户可随时清空 |
| 消歧义策略 | 由 LLM 自动理解代词 | 显式消歧义预处理 | 简单有效，LLM 天然擅长代词消解 |

---

## 10. 流式生成架构

### 10.1 为什么需要流式传输

```
LLM 生成特点：
  首个 token 延迟：300-800ms
  完整回答长度：200-500 tokens
  生成速度：30-60 tokens/s
  完整答案等待时间：3-15 秒

非流式体验（用户视角）：
  0ms ─────────────────────────────────────► 10s
  [等待中..........................................................] 答案出现

流式体验（用户视角）：
  0ms ──────────────────────────────────────► 10s
  [    ][RAG][是][一][种][检][索][增][强]...  逐字显示

感知延迟降低 80%+，用户体验显著提升
```

### 10.2 系统架构流程

```
┌──────────────────────────────────────────────────────────┐
│                    流式传输完整流程                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  前端 (React)                WebSocket                   │
│  ─────────                   ────────────                │
│  用户提交查询                                             │
│      │                                                   │
│      ├──── WS connect ──────────────────────────────────►│
│      │                                                   │
│      ├──── send JSON ──────────────────────────────────► │
│      │    {query, strategy,                              │
│      │     history, use_hyde}         后端 (FastAPI)      │
│      │                                ────────────────── │
│      │                                │ 检索文档          │
│      │                                │ 构建 messages     │
│      │                                │                  │
│      │                                ▼                  │
│      │                         DeepSeek Streaming API    │
│      │                                │                  │
│      │◄── {type:"retrieval_done"} ────┤                  │
│      │    {docs, scores}              │                  │
│      │                                │ stream=True      │
│      │◄── {type:"answer_token"} ──────┤ 逐 token 返回     │
│      │    {token:"RAG"}               │                  │
│      │◄── {type:"answer_token"} ──────┤                  │
│      │    {token:"是"}                │                  │
│      │    ...                         │                  │
│      │◄── {type:"answer_complete"} ───┤ 生成完毕          │
│      │    {ragas_metrics}             │ RAGAS 计算        │
│      │                                                   │
└──────────────────────────────────────────────────────────┘
```

### 10.3 关键实现代码

**后端（逐 token 推送）**：

```python
@app.websocket("/ws/query")
async def query_websocket(websocket: WebSocket):
    await websocket.accept()
    
    data = await websocket.receive_json()
    query = data["query"]
    
    # 阶段 1：检索文档
    docs = await retrieve_documents(query)
    await websocket.send_json({
        "type": "retrieval_done",
        "docs": [doc.to_dict() for doc in docs]
    })
    
    # 阶段 2：流式生成
    messages = build_messages(query, docs, data.get("history", []))
    stream = await openai_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=True,
    )
    
    accumulated = ""
    async for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            accumulated += token
            await websocket.send_json({
                "type": "answer_token",
                "token": token,
                "full_answer_so_far": accumulated
            })
    
    # 阶段 3：RAGAS 评估
    metrics = compute_ragas(query, docs, accumulated)
    await websocket.send_json({
        "type": "answer_complete",
        "ragas_metrics": metrics
    })
```

### 10.4 WebSocket vs SSE 的选择理由

| 特性 | WebSocket（本系统） | Server-Sent Events |
|------|-------------------|--------------------|
| 通信方向 | 双向 | 服务端 → 客户端单向 |
| 查询参数传递 | WS 消息体（JSON） | 需要额外 POST 请求 |
| 连接建立 | HTTP 升级握手 | 普通 HTTP 长连接 |
| 断线重连 | 需手动实现 | 浏览器原生支持 |
| 适合场景 | 需要双向交互（发查询+收流式结果）| 纯服务端推送 |

本系统需要通过同一连接**发送复杂查询参数**（strategy、history、use_hyde 等）并接收流式响应，WebSocket 双向特性天然适合。

### 10.5 异步设计与事件循环保护

```python
# 问题：CPU 密集型 Embedding 计算会阻塞异步事件循环
# 后果：在 embedding 计算期间，其他 WebSocket 连接无法响应

# ❌ 错误做法：直接调用（阻塞事件循环）
scores = compute_embeddings(query)

# ✓ 正确做法：卸载到线程池
import asyncio
loop = asyncio.get_event_loop()
scores = await loop.run_in_executor(
    None,              # 使用默认线程池
    compute_embeddings,
    query
)
# 非阻塞，其他 WebSocket 连接可以继续处理
```

---

## 11. 系统设计决策与权衡

### 11.1 文档分块策略

```
分块流程：
原始文档
    │
    ▼
按双换行符（\n\n）分段
    │
    ▼
段落长度检查
    │
    ├── < 100 字  →  与下一段合并（太短，上下文不完整）
    ├── 100-600 字 →  直接保留
    └── > 600 字  →  强制按 600 字切分
```

**600 字阈值的理论依据**：

```
太短（< 200 字）：
  - 上下文不完整，缺少解释性内容
  - 检索到的片段无法独立成意
  - 降低 LLM 生成质量

太长（> 1000 字）：
  - 一个块包含多个话题，向量代表性降低
  - 向量是"平均"语义，稀释关键信息
  - 检索时信噪比下降

600 字 ≈ 400 tokens：
  - 处于 BGE 最佳表示范围内（最大 512 tokens）
  - 足够完整的语义单元
  - 不超过 embedding 模型的最优输入长度
```

### 11.2 DeepSeek vs GPT-4 的选型分析

```
成本对比（以 1000 次问答估算）：
  GPT-4o：      ~$50-100
  DeepSeek-V3： ~$1.5-3   （约 1/30 成本）

中文理解：
  DeepSeek 在中文语料上进行了专项预训练
  在 CMMLU、C-Eval 等中文基准上表现接近或超越 GPT-4

API 兼容性：
  DeepSeek 使用 OpenAI SDK 格式
  切换成本：仅需修改 base_url 和 api_key
  
  client = openai.AsyncOpenAI(
      api_key=DEEPSEEK_API_KEY,
      base_url="https://api.deepseek.com",  ← 唯一变化
  )
```

### 11.3 降级策略（Graceful Degradation）

生产级系统必须处理各组件不可用的场景：

```
组件失效场景                降级策略                   影响
─────────────────────────────────────────────────────────
无 DeepSeek API Key    → 直接引用文档原文              回答质量下降，仍可运行
无 Cross-Encoder       → 余弦相似度 + 标题匹配加分      精排精度轻微下降
无 jieba 分词库         → 空格分词（英文仍有效）          中文 BM25 精度下降
无 PDF 解析库           → 跳过 PDF，处理其他格式         部分文档无法接入
无文档                  → 显示提示信息                  不崩溃，提示用户上传
```

---

## 12. 面试高频问题整理

### Q1：RAG 和 Fine-tuning 有什么区别？什么时候用哪种？

```
Fine-tuning：
  ✓ 修改模型参数，将知识"烧入"模型
  ✓ 适合：风格迁移、特定任务优化、不变的专业知识
  ✗ 需要大量标注数据；更新知识需重新训练；成本高

RAG：
  ✓ 不修改模型参数，通过检索动态注入知识
  ✓ 适合：频繁更新的知识；私有/机密数据；需要可溯源答案
  ✗ 检索质量上限了生成质量；延迟比纯 LLM 高

结合使用（RAG + Fine-tuning）：
  Fine-tuning 用于领域适应（学会说该领域的"语言"）
  RAG 用于实时知识更新（不把具体事实烧入参数）
```

### Q2：如何评估 RAG 系统的质量？

```
检索层：
  - Recall@K：相关文档是否出现在 Top-K 结果中
  - MRR（Mean Reciprocal Rank）：相关文档的排名
  - NDCG@K：考虑排名顺序的综合指标

生成层：
  - RAGAS 的 4 个指标（Context Relevance/Precision, Answer Relevance/Faithfulness）
  - 人工评估（准确性、完整性、流畅性）
  - LLM-as-Judge（用强 LLM 评估生成质量）

端到端：
  - 业务指标：用户满意度、问题解决率
```

### Q3：HyDE 和直接查询扩展（Query Expansion）的区别？

```
Query Expansion（传统）：
  方法：添加同义词/相关词扩展查询
  例：RAG → RAG OR "检索增强生成" OR "retrieval augmented generation"
  问题：需要预定义词典；扩展词选择困难

HyDE：
  方法：用 LLM 生成一段假设性的答案文档
  优点：LLM 能理解语义，生成高质量假设；无需预定义词典
  关键：利用假设文档的**向量**（非文本）进行检索，与文档分布对齐
```

### Q4：Cross-Encoder 为什么不能用于全量检索？

```
Cross-Encoder 需要将 (Query, Document) 对同时送入模型
  → 无法预计算文档向量（因为每次查询都需要重新计算）
  → 时间复杂度 O(N)，N=文档数
  → N=10,000 时，需要 10,000 次 BERT 推理 ≈ 200 秒

Bi-Encoder 可以预计算：
  → 文档向量离线计算并存储
  → 检索时只需计算一次查询向量
  → 相似度计算用矩阵乘法，O(N×d) 但极快
```

### Q5：如何处理 LLM 生成的幻觉问题？

```
检测层面：
  1. 答案忠实度（RAGAS Faithfulness）监控
  2. 关键声明的 NLI 蕴含检测
  3. 置信度阈值：检索分数低时警告用户

抑制层面：
  1. System Prompt："仅根据提供的文档回答，文档中没有的信息请明确说明"
  2. 保留文档引用，让用户可以核查原文
  3. RAG 本身就是减少幻觉的机制：为 LLM 提供事实锚点
```

### Q6：系统如何扩展到百万级文档？

```
当前架构（万级文档）：
  - numpy 矩阵乘法全量扫描
  - BM25 倒排索引内存加载

扩展路径（百万级文档）：
  1. 向量数据库：Milvus / Qdrant / Weaviate
     - HNSW / IVF 近似最近邻搜索
     - O(log N) 查询时间，可水平扩展

  2. BM25 扩展：Elasticsearch / OpenSearch
     - 分布式倒排索引
     - 支持分片和副本

  3. 两阶段缓存：
     - 热门查询向量缓存（Redis）
     - 文档向量持久化（避免重启重新计算）
```

---

## 参考文献

1. **RAG 原论文**: Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020
2. **HyDE**: Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels*, EMNLP 2022
3. **ReAct**: Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023
4. **RAGAS**: Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation*, arXiv 2023
5. **BGE**: Xiao et al., *C-Pack: Packaged Resources To Advance General Chinese Embedding*, arXiv 2023
6. **BM25**: Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009
7. **Cross-Encoder**: Nogueira & Cho, *Passage Re-ranking with BERT*, arXiv 2019
8. **Sentence-BERT**: Reimers & Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, EMNLP 2019
