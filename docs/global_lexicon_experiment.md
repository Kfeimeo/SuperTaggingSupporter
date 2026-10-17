# 全局 CCG 词典：构建与跨语料覆盖实验

流程：**源语料 → 上下文超标注 → 固定全局词典 → 目标语料查词典 → Eisner 正规形 CKY**。
目标语料测评期间不加载超标注器、不更新词典，未登录词采用严格失败策略。

## 1. 每个词的范畴集合和概率怎样汇总

设词形 `w` 在源语料中出现 `N_w` 次，每次超标注输出完整分布 `p_i(c)`。
保存三个独立的统计量：

```text
count[w]          = N_w
soft_counts[w,c]  = Σ_i p_i(c)
support[w,c]      = Σ_i 1[c ∈ 第 i 次出现的候选集合]
```

完整的全局分布取算术平均：

```text
q(c|w) = soft_counts[w,c] / count[w]
```

这是把源语料中该词的上下文按出现频率混合后的模型分布，也是用软标签拟合一个无上下文分类分布的交叉熵最优解。
它反映模型在源语料上的预测，不是经过人工验证的真实范畴概率，也不保证适应目标领域的用法频率。

例如 `fly` 两次出现时输出 `(N:0.95, VP:0.05)` 和 `(N:0.15, VP:0.85)`，
全局平均为 `(N:0.55, VP:0.45)`。这里 `VP` 仅是示例简写，实际保存完整 CCG 范畴字符串。

不要直接把不同出现位置的概率相乘：它们可能对应不同用法，并非对同一个隐含范畴的重复观测。
取最大值可以表示“是否曾经支持某个范畴”的分数，但会被一次高置信误标影响，归一化后也不表示用法频率。
硬计数 `argmax` 的频率适合作为基线，会丢失每次预测的不确定性。

范畴集合另行定义为：

```text
L(w) = 各次出现候选集合的并集
     = { c : support[w,c] >= min_support }
```

每次出现先按概率从高到低取候选，达到累计概率 `beta` 或数量 `k` 时停止；
与现有项目中的定义一致。`--k 1` 就是每次只收集 argmax 的集合。
不要把完整 softmax 中一切大于零的范畴都收入集合，否则微小尾部概率会使集合失去区分度。

**先用完整概率做 soft_counts，再独立记录候选支持次数**。不要先截断并归一化每次出现的概率再平均，
否则阈值不同会改变估计含义。当前代码已经把这两步分开。

测评时只在 `L(w)` 中分配概率，采用 `q(c|w)` 在集合内归一化后的 `log` 值。
这个归一化只给同一个位置的所有范畴加同一常数，不改变该句最佳推导的排序。
`--weighting uniform` 可让集合中每个范畴等概率，用于集合覆盖实验。
在没有 beam 的情况下，两种权重使用相同集合，应得到相同的可解析性，最佳推导可以不同。

## 2. 在 TinyStories 上构建词典

在项目根目录运行，使用已配置好的 `ccg` 环境：

```powershell
conda activate ccg

# 项目自带的是 TinyStories 小样本，不是完整 TinyStories。
python -m stsupport build-lexicon `
  --corpus data/tinystories_sample.txt `
  --format txt `
  --lexicon output/tinystories.lexicon.json `
  --k 8 --beta 0.99
```

对自己的完整 TinyStories 原始文件，使用 `stories` 格式：

```powershell
python -m stsupport build-lexicon `
  --corpus "D:/corpora/TinyStories-train.txt" `
  --format stories `
  --lexicon output/tinystories_full.lexicon.json `
  --k 8 --beta 0.99
```

输出：

| 文件 | 内容 |
| --- | --- |
| `tinystories.lexicon.json` | 类别表、词频、完整 soft_counts、候选 support、源语料和模型设置 |
| `tinystories.lexicon.map.json` | 直接可用的 `word → [category, ...]`，对应 `map<word,set<category>>` |
| `tinystories.lexicon.summary.json` | 源句数、词数、词形数、范畴对数量、运行时间 |

Python 中可以得到完整全局概率表：

```python
from stsupport.lexicon import GlobalLexicon

lex = GlobalLexicon.load("output/tinystories.lexicon.json")
word_to_categories = lex.category_map()
q = lex.distribution("dog")  # 与 lex.categories 对齐的完整平均概率；未登录词返回 None
if q is not None:
    table = dict(zip(lex.categories, q.tolist()))
```

默认区分大小写。构建时加 `--lowercase` 可合并大小写变体，目标测评自动沿用，
但这也可能混合专名和普通词。默认不做词形还原，`run`、`runs`、`ran` 是三个词典键。

源语料按批读取并超标注，不必一次缓存全部句子；统计空间仍为 `O(词形数 × 范畴数)`，
因为保留完整软计数向量。1341 类、10 万词形的 float64 软计数本身约占 1.07 GB，
JSON 文件和加载时的临时对象还会增加开销。可先加 `--limit 10000` 做规模试验。

## 3. 用自定义语料测评

支持 UTF-8 文件：

| `--format` | 输入格式 |
| --- | --- |
| `txt` | 每行一个英文句子，使用项目现有的 PTB 分词流程 |
| `stories` | TinyStories 原始文本，按 `<\|endoftext\|>` 分隔故事，再分句 |
| `jsonl` | 每行一个对象，含 `tokens` 数组或 `text` 字段 |
| `auto` | `.jsonl` 按 JSONL 读取，其余依据内容识别；正式实验建议显式指定 |

文本例子 `D:/corpora/my_test.txt`：

```text
The dog sleeps.
Lily found a ball.
```

运行：

```powershell
python -m stsupport eval-lexicon `
  --lexicon output/tinystories.lexicon.json `
  --corpus "D:/corpora/my_test.txt" `
  --format txt `
  --set output.name=my_test
```

对于已分词的目标语料，建议使用 JSONL，避免重新分词造成差异：

```jsonl
{"tokens": ["The", "dog", "sleeps", "."]}
{"tokens": ["Lily", "found", "a", "ball", "."]}
```

```powershell
python -m stsupport eval-lexicon `
  --lexicon output/tinystories.lexicon.json `
  --corpus "D:/corpora/my_test.jsonl" `
  --format jsonl `
  --set output.name=my_jsonl_test
```

`tokens` 应与源语料使用相同的 PTB 约定，例如缩写拆分、`-LRB-`/`-RRB-`。
现有加载器默认去掉引号，源和目标的 `corpus.drop_quotes` 必须相同。
默认过滤少于 2 个 token 的句子；如需保留单词句，构建和测评均加 `--set corpus.min_tokens=1`。
当前模型和范畴体系面向英语；输入其他语言文本不能直接作为同等条件的语言覆盖测评。

测评阶段没有神经模型依赖，也不使用 `search.k`、`search.beta` 或渐进加宽策略，
而是一次提交该词典允许的全部范畴。
`--min-support 2` 可以要求一个词—范畴对至少在源语料的两次出现中入选；默认 1，保留并集。
调高该值可能滤掉真实低频用法，甚至让一个已知词没有任何可用范畴，结果会单列 `empty_support`。
修改构建阶段的 k/β 要重新构建；修改测评阶段的支持次数阈值无需重新超标注。

输出到 `output/my_test.*`：

| 文件 | 内容 |
| --- | --- |
| `.summary.json` | 全局覆盖率、解析率、失败原因、候选数量和完整实验设置 |
| `.jsonl` | 每句 token、未登录词位置、候选数、状态、最终范畴及推导 |
| `.auto` | CCGbank 格式推导；失败行 `NUMPARSE=0` |
| `.oov.json` | 目标语料未登录词及次数，按频率降序 |
| `.config.yaml` | 测评配置 |

不要把输出前缀设为输入语料的同名路径；命令会拒绝覆盖输入文件。

候选规模很大时，可使用独立引擎多进程模式，保持相同词典、语法和概率权重：

```bash
python -u -m stsupport eval-lexicon --lexicon output/tinystories_full.lexicon.sqlite --corpus /mnt/datasets/my_test.jsonl --format jsonl --parser-backend processes --processes 4 --max-in-flight 8 --set output.name=my_test_processes
```

每个进程复用自己的解析缓存，主进程查询词典、动态派发和汇总；不需要重建词典。
此时 `search.n_threads` 不参与调度，`--batch-size` 仅控制主进程每轮准备句数，无解析批次屏障。
详见 [独立进程解析与云主机参数](data_download_and_cloud.md#独立进程解析动态派发与主进程汇总)。

## 4. 怎样解释测评指标

| 指标 | 分子 / 分母 |
| --- | --- |
| `oov_token_rate` | 未登录 token 数 / 全部目标 token 数 |
| `word_type_coverage` | 源词典中出现过的目标词形数 / 全部目标词形数 |
| `token_candidate_coverage` | 至少有一个可用范畴的 token 数 / 全部 token 数 |
| `sentence_candidate_coverage` | 每个 token 都有候选的句数 / 全部目标句数 |
| `parse_rate_all` | 解析成功句数 / 全部目标句数（超长跳过也计入分母） |
| `parse_rate_eligible` | 解析成功句数 / 未因句长跳过的句数 |
| `parse_rate_covered_eligible` | 解析成功句数 / 全词有候选且未超长的句数 |
| `avg_candidates_per_available_token` | 候选总数 / 有候选的 token 数 |

空分母返回 JSON `null`。词汇覆盖统计包含超长句，解析条件覆盖率则排除超长句。
失败的 `reason` 为 `oov`、`empty_support`、`no_derivation`，跳过的为 `max_tokens`；
即使句长超限，记录里仍保留 OOV 和候选信息。

**词汇覆盖不是正确范畴覆盖，可解析也不是解析正确。**
一个词在词典中存在并不代表该次出现所需的范畴已收录；一个错误范畴序列也可能推出 S。
有 gold CCG 范畴时，应另外计算 gold token 范畴是否在 `L(w)` 中、整句 gold 范畴是否全覆盖，
以及依存/结构准确率。没有 gold 时，可用目标语料超标注结果作独立的诊断参考，
但不能称为 gold，也不能用它补充测评词典。此诊断不包含在当前命令中。

默认根范畴为 `S`，不会接受以 `S[b]\\NP` 为根的祈使句，默认超过 80 token 跳过。
这些设置沿用现有项目，测评摘要会保存。可通过配置文件修改 `grammar.root_categories`，
并用 `--set search.max_tokens=120` 调整句长上限。各组实验必须采用相同规则和根范畴。
默认 `grammar.cell_beam=0`，在当前实现的语法与正规形约束下做精确图表搜索；
设置非零 beam 后摘要 `exact_chart=false`，失败可能来自剪枝。
启用 `grammar.use_seen_rules=true` 时，需要明确指定本地 `grammar.seen_rules_path`，
测评不会为了查找规则表而加载或下载模型。

Eisner 正规形约束用于减少推导歧义；覆盖结论仍以项目实际支持的规则、组合阶数和根范畴为条件。
参考：[Eisner 1996](https://aclanthology.org/P96-1011/)。

## 5. 建议的对照实验

1. 固定目标语料和语法，分别构建 `k=1`、`k=8, beta=0.99`、更宽候选的词典。
2. 固定词典，比较 `min_support=1` 和 `2`，观察噪声、候选规模、覆盖率的变化。
3. 对不同源语料规模构建词典，绘制词形/OOV/解析覆盖的增长曲线。
4. 单独报告同领域保留集和跨领域语料，不用词典构建语料上的解析率替代跨领域结果。

TinyStories 的训练与同领域保留集应按**完整故事**划分，再分句；相同或重复故事不要跨集合。
不要用最终目标测试集调 k/β、支持阈值或语法。当前仓库只附带一个样本文件，
由它构建的词典只能代表该样本规模，跨语料结果需在你选定目标语料后运行上述测评命令。

## 6. 本次流程验证

已使用仓库样本构建 `output/tinystories.lexicon.json`：2,226 句、27,797 个 token、
2,080 个词形、9,940 个词—范畴对，构建参数为 `k=8, beta=0.99`。
源样本前 30 句的词典解析自检为 30/30 成功，平均每个 token 11.71 个候选，
默认语法、无 beam、4 个解析线程，解析与统计用时约 20.1 秒。
记录在 `output/lexicon_source_smoke.summary.json`。
这是源语料上的流程自检，不能作为保留集或跨语料覆盖率结论。

随后已完成 Brown news 全子集跨语料测评及上下文超标注参考组，
结果与复现命令见 [Brown news 实验报告](brown_news_experiment_2026-10-15.md)。
