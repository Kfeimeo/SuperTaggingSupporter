"""Generate the full-source comparison only after its lexicon is complete."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    out = root / "output"
    def load(name, optional=False):
        path = out / name
        if optional and not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    source = load("tinystories_full.lexicon.summary.json")
    if source.get("complete") is not True:
        raise ValueError("full-source lexicon is incomplete; do not report partial results as full results")
    small = load("brown_news_global_lexicon.summary.json", optional=True)
    full = load("brown_news_full_global_lexicon.summary.json")
    contextual = load("brown_news_contextual_reference.summary.json", optional=True)
    if small is not None:
        assert small["sentences"] == full["sentences"] and small["grammar"] == full["grammar"]
    if contextual is not None:
        assert contextual["sentences"] == full["sentences"]
    target = Path(full["target_corpus"]["path"])
    if not target.is_absolute():
        target = root / target
    manifest = json.loads(target.with_suffix(".manifest.json").read_text())
    assert hashlib.sha256(target.read_bytes()).hexdigest() == manifest["output_sha256"]
    p = lambda value: f"{value * 100:.2f}%" if value is not None else "无定义"
    def sample(key, kind="percent"):
        if small is None:
            return "未运行"
        if kind == "count":
            return f"{small[key]:,}"
        if kind == "float":
            return f"{small[key]:.2f}"
        return p(small[key])

    if contextual is not None:
        reference_n = contextual["onebest"] + contextual["recovered"]
        reference_text = (f"独立的上下文超标注参考组在相同目标语料上解析成功 {reference_n:,} 句，"
                          f"整体解析率 {p(contextual['parse_rate'])}。参考组不是 gold，且使用上下文候选，"
                          "其候选集合与全局词典组不同。")
    else:
        reference_text = "此主机尚未运行上下文超标注参考组。未提供的基线列标为“未运行”，不会冒充实测结果。"
    text = f"""# 完整 TinyStories 训练集 → Brown news

## 数据与设置

源数据为原始 TinyStories 的完整 `train`，共 {source['stories']:,} 篇故事、
{source['sentences']:,} 个有效句子、{source['tokens']:,} 个 token，
{source['word_types']:,} 个词形、{source['category_pairs']:,} 个词—范畴对。
读取到文件末尾，未设置故事数或句数上限。TinyStories validation 和 V2 不混入。
源数据 SHA-256：`{source['metadata']['source']['sha256']}`。

与小样本实验相同：`k=8, beta=0.99` 候选并集、完整概率分布的算术平均、
`min_support=1`、保留大小写、严格 OOV；目标固定为 Brown news 的 {full['sentences']:,} 句，
语法不变、无 beam、句长上限 80、4 个解析线程。
FP32 神经推理使用向量化词级池化；小批验证与原实现最大概率差值记录在 benchmark 文件中。
超过模型容量的个别源句会拆成子上下文，保留全部 token，
累计拆分操作为 {source.get('long_sentence_chunk_operations', 0):,} 次。

## 结果

| 指标 | TinyStories 2,226 句样本 | TinyStories 完整 train |
| --- | ---: | ---: |
| 目标 token 候选覆盖率 | {sample('token_candidate_coverage')} | {p(full['token_candidate_coverage'])} |
| 目标词形覆盖率 | {sample('word_type_coverage')} | {p(full['word_type_coverage'])} |
| 所有词有候选的句数 | {sample('covered_sentences', 'count')} | {full['covered_sentences']:,} |
| 整句候选覆盖率 | {sample('sentence_candidate_coverage')} | {p(full['sentence_candidate_coverage'])} |
| 解析成功句数 | {sample('parsed', 'count')} | {full['parsed']:,} |
| 整体解析率 | {sample('parse_rate_all')} | {p(full['parse_rate_all'])} |
| 已覆盖且未超长句中的解析率 | {sample('parse_rate_covered_eligible')} | {p(full['parse_rate_covered_eligible'])} |
| 有候选 token 的平均候选数 | {sample('avg_candidates_per_available_token', 'float')} | {full['avg_candidates_per_available_token']:.2f} |

{reference_text}

全量词典构建累计耗时约 {source['elapsed_seconds'] / 3600:.2f} 小时（不含初始化及最终导出），
词典测评耗时约 {full['seconds']:.2f} 秒（不含词典加载）。
可解析只代表存在合法推导，不等于结构正确率。这个 Brown news 已用于探索性对比，
后续若依据结果调参，最终泛化评估应另选独立测试集。

## 文件与复现

- `output/tinystories_full.lexicon.sqlite`：完整软计数、词频、支持次数和事务检查点。
- `output/tinystories_full.lexicon.map.json`：纯词典范畴集合。
- `output/tinystories_full.lexicon.summary.json`：完整源语料统计。
- `output/brown_news_full_global_lexicon.summary.json`：全量词典测评指标。
- `output/brown_news_full_global_lexicon.jsonl`、`.auto`、`.oov.json`：逐句结果、推导、未登录词。

```powershell
conda activate ccg
python -u scripts/run_full_tinystories_experiment.py
```

同一命令支持中断恢复，已提交的故事不会重复计数；同一数据库拒绝不同配置。
单独重跑测评：

```powershell
python -m stsupport eval-lexicon --lexicon output/tinystories_full.lexicon.sqlite --corpus data/brown_news.jsonl --format jsonl --set output.name=brown_news_full_global_lexicon --set search.n_threads=4 --batch-size 4
python scripts/report_full_tinystories_experiment.py
```

模型、数据来源及基础实验说明见 [小样本报告](brown_news_experiment_2026-10-15.md)。
"""
    report = root / "docs/tinystories_full_experiment.md"
    report.write_text(text, encoding="utf-8")
    print(json.dumps({"report": str(report), "source_stories": source["stories"],
                      "parse_rate": full["parse_rate_all"], "token_coverage": full["token_candidate_coverage"]}, indent=2))


if __name__ == "__main__":
    main()
