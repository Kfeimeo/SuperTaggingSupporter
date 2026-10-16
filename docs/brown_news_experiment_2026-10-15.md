# TinyStories 全局词典 → Brown news 跨语料覆盖实验

实验日期：2026-10-15。

## 结论

固定 TinyStories **小样本**词典在 Brown news 上的 token 候选覆盖率为 **60.48%**，
整句候选覆盖率为 **1.13%**，整体解析率为 **0.85%**。
在所有词都有候选的 52 句中，39 句成功，条件解析率为 **75.00%**。
当前主要限制是未登录词，不能据此推断完整 TinyStories 词典的效果，也不能推断全局词典方案本身不可行。

## 为什么选这个语料

[Brown Corpus](https://www.nltk.org/book/ch02) 有明确的 `news` 类别和现成句子边界。
本次选择完整 news 子集，让儿童故事词典迁移到新闻文本，直接测量跨领域词汇与组合覆盖。
它没有在本实验中使用的 gold CCG 推导，因此这里报告可解析率，不报告结构正确率。

## 固定设置与数据来源

- 源词典：仓库 TinyStories 样本，2,226 句、27,797 token、
  2,080 个词形、9,940 个词—范畴对；不是完整 TinyStories。
- 源超标注器：`ryosuke-yamaki/hol-ccg-roberta-large`，C&C grammar mask 开启。
- 构建集合：各次出现 `k=8, beta=0.99` 候选的并集；`min_support=1`。
  分数来自完整分布的算术平均，再在集合内归一化。
- 目标：Brown 全部 44 个 news 文档，原始 4,623 句。
  沿用原始句界，反分词后用项目 PTB 分词，去引号、保留大小写；
  过滤 35 个少于 2 token 的句子，最终 4,588 句、100,319 token。
  原始重复标点保留，不因失败而清洗或排除句子。
- 按预处理后的完整 token 序列检查，目标句子中有 0 句与源样本完全相同；
  这项检查不代表能够排除神经模型预训练数据中的潜在重叠。
- 语法：项目默认 Eisner 正规形 CKY，根范畴 `S`（匹配其特征变体），组合阶数上限 2，
  无 beam，4 个解析线程，超过 80 token 跳过。两组语法一致。
- 词典组：使用所有词典候选，遇到 OOV 严格失败，不查询超标注器，不向词典加入目标语料信息。
- 参考组：在同一目标语料上使用上下文超标注器，先检查 1-best，失败后用 `k=8, beta=0.99` lattice。
  这是独立的解析参考组，不是 gold；与词典组的候选集合大小也不同。

官方归档：[NLTK brown.zip](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip)。
[NLTK 许可元数据](https://github.com/nltk/nltk_data/blob/gh-pages/packages/corpora/brown.xml) 标注为非商业用途。
下载文件经过 SHA-256 校验；文档 ID、导出设置和数据校验值保存在 `data/brown_news.manifest.json`。

## 测评结果

| 指标 | 固定全局词典 |
| --- | ---: |
| token 候选覆盖率 | 60.48% |
| 目标词形覆盖率 | 9.98% |
| 未登录 token | 39,643 / 100,319（39.52%） |
| 未登录词形 | 12,507 / 13,893 |
| 整句所有词有候选 | 52 / 4,588（1.13%） |
| OOV 导致失败 | 4,532 句 |
| 所有词有候选但无 S 推导 | 13 句 |
| 超过 80 token 跳过 | 4 句 |
| 每个有候选 token 的平均候选数 | 14.40 |

| 相同目标句集 | 固定全局词典 | 上下文超标注参考组 |
| --- | ---: | ---: |
| 解析成功句数 | 39 | 4,263 |
| 全部 4,588 句上的解析率 | 0.85% | 92.92% |
| 排除超长句后的解析率 | 0.85% | 93.00% |
| 在词典全词覆盖的 52 句上解析成功 | 39 / 52 | 45 / 52 |

词典测评核心运行时间约 6.44 秒，参考组超标注 54.46 秒、
解析 89.95 秒；这些计时不含模型/词典加载，且参考组处理了更多可进入解析的句子，
不适合直接解释为两种方法的速度比。

75.00% 的条件解析率仅代表这个被词汇覆盖条件筛选出的短句集合，不能外推至全部新闻句子。
解析成功只表示当前语法下存在合法推导，不能保证词义、范畴或依存结构正确。

## 失败诊断

高频未登录项：

| 词形 | 次数 |
| --- | ---: |
| `--` | 300 |
| `Mrs.` | 253 |
| `$` | 217 |
| `Mr.` | 170 |
| `-LRB-` | 168 |
| `-RRB-` | 164 |
| `President` | 100 |
| `state` | 97 |
| `New` | 93 |
| `week` | 91 |
| `Kennedy` | 77 |
| `House` | 73 |

这些缺口包含新闻用语、专名、称谓、数字形式和源样本未出现过的标点。
39,643 个 OOV token 中，只有 1,421 个在源词典中存在大小写变体；
即使允许合并大小写，仍有 38,222 个 token 没有对应词形。
这只是词汇诊断，没有在本次测评中合并大小写或更改词典。

13 个全词覆盖失败记录保存在 `output/brown_news_diagnostics.json`，包含原始文档句子 ID、
完整跨度的候选根标签和参考组状态。其中包括标题/片段、疑问省略，以及词形已知但候选用法不足等可能原因。
没有 gold 分析，不能把每个失败直接归因于某一种语言现象。

## 复现

在项目根目录、`ccg` 环境运行：

```powershell
python scripts/prepare_brown_news.py

# 如需从头重建源词典：
python -m stsupport build-lexicon --corpus data/tinystories_sample.txt --format txt --lexicon output/tinystories.lexicon.json --k 8 --beta 0.99

python -m stsupport eval-lexicon --lexicon output/tinystories.lexicon.json --corpus data/brown_news.jsonl --format jsonl --set output.name=brown_news_global_lexicon --set search.n_threads=4 --batch-size 32

python -m stsupport run --corpus data/brown_news.jsonl --k 8 --beta 0.99 --set corpus.format=jsonl --set search.n_threads=4 --set output.name=brown_news_contextual_reference

python scripts/report_brown_news_experiment.py
```

语料导出脚本只在本地归档不存在时下载；重新运行会校验归档哈希。
神经推理受环境与数值精度影响，重建词典或参考组时结果可能有少量变化。

## 下一步如何验证词典规模效应

优先增加源 TinyStories 的故事数量，并固定这份 Brown news 目标集及语法设置，
观察 OOV 率、整句候选覆盖率和解析率随源语料规模的变化。
这能区分“源样本太小”和“源领域缺少目标用法”。
在开发集上选择参数后再用独立测试集做最终评估；本次 Brown news 可作为探索性基准，
不应既用来调参，又将调参后的同一结果作为独立泛化证据。

自定义语料的 TXT/JSONL 格式与命令见 [使用指南](global_lexicon_experiment.md)。
