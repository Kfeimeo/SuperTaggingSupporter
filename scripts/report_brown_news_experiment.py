"""Validate experiment records and generate a reproducible Chinese report."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from dataclasses import asdict

from stsupport.config import load_config
from stsupport.corpus import load_corpus
from stsupport.lexicon import GlobalLexicon


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value):
    return f"{value * 100:.2f}%" if value is not None else "无定义"


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "output"
    lex_path = output / "tinystories.lexicon.json"
    lex = GlobalLexicon.load(lex_path)
    source_summary = json.loads((output / "tinystories.lexicon.summary.json").read_text())
    manifest = json.loads((root / "data/brown_news.manifest.json").read_text())
    summary = json.loads((output / "brown_news_global_lexicon.summary.json").read_text())
    ref_summary = json.loads((output / "brown_news_contextual_reference.summary.json").read_text())
    target = read_jsonl(root / "data/brown_news.jsonl")
    records = read_jsonl(output / "brown_news_global_lexicon.jsonl")
    reference = read_jsonl(output / "brown_news_contextual_reference.jsonl")
    ref_config = load_config(output / "brown_news_contextual_reference.config.yaml")
    assert asdict(ref_config.grammar) == summary["grammar"]
    assert ref_config.search.k == 8 and ref_config.search.beta == 0.99
    assert len(target) == len(records) == len(reference) == summary["sentences"] == manifest["sentences"]
    for original, record, ref in zip(target, records, reference):
        assert original["tokens"] == record["tokens"] == ref["tokens"]
    assert hashlib.sha256((root / "data/brown_news.jsonl").read_bytes()).hexdigest() == manifest["output_sha256"]
    assert sum(len(r["tokens"]) for r in records) == summary["tokens"]
    assert Counter(r["status"] for r in records) == {"parsed": summary["parsed"], "fail": summary["failed"], "skipped": summary["skipped"]}
    for ref, record in zip(reference, records):
        assert (ref["status"] == "skipped") == (record["status"] == "skipped")
    success = {"onebest", "recovered"}
    ref_parsed = sum(r["status"] in success for r in reference)
    covered_ids = [i for i, r in enumerate(records) if not r["oov_positions"] and not r["filtered_positions"] and r["status"] != "skipped"]
    ref_covered_parsed = sum(reference[i]["status"] in success for i in covered_ids)
    oov = Counter(w for r in records for w in r["tokens"] if lex.key(w) not in lex.counts)
    assert sum(oov.values()) == summary["oov_tokens"]
    lower_words = {w.lower() for w in lex.counts}
    source_sentences = {tuple(s) for s in load_corpus(root / "data/tinystories_sample.txt", fmt="txt")}
    overlap = [r["id"] for r in target if tuple(r["tokens"]) in source_sentences]
    case_only = sum(n for w, n in oov.items() if w.lower() in lower_words)
    no_derivation = [{"id": original["id"], "tokens": r["tokens"], "root_cell_cats": r["root_cell_cats"],
                      "contextual_reference_status": ref["status"]}
                     for original, r, ref in zip(target, records, reference) if r["reason"] == "no_derivation"]
    diagnostics = {
        "lexicon_sha256": hashlib.sha256(lex_path.read_bytes()).hexdigest(),
        "target_sha256": manifest["output_sha256"],
        "case_only_oov_tokens": case_only, "oov_tokens_still_absent_after_lowercase": sum(oov.values()) - case_only,
        "covered_eligible_sentences": len(covered_ids), "contextual_parsed_on_covered_eligible": ref_covered_parsed,
        "dictionary_parsed_contextual_failed": sum(r["status"] == "parsed" and ref["status"] not in success for r, ref in zip(records, reference)),
        "no_derivation_records": no_derivation,
        "target_corpus_used_to_update_dictionary": False,
        "exact_token_sequence_overlap_with_source": overlap,
    }
    (output / "brown_news_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    covered_n = len(covered_ids)
    text = f"""# TinyStories 全局词典 → Brown news 跨语料覆盖实验

实验日期：2026-10-15。

## 结论

固定 TinyStories **小样本**词典在 Brown news 上的 token 候选覆盖率为 **{pct(summary['token_candidate_coverage'])}**，
整句候选覆盖率为 **{pct(summary['sentence_candidate_coverage'])}**，整体解析率为 **{pct(summary['parse_rate_all'])}**。
在所有词都有候选的 {covered_n} 句中，{summary['parsed']} 句成功，条件解析率为 **{pct(summary['parse_rate_covered_eligible'])}**。
当前主要限制是未登录词，不能据此推断完整 TinyStories 词典的效果，也不能推断全局词典方案本身不可行。

## 为什么选这个语料

[Brown Corpus](https://www.nltk.org/book/ch02) 有明确的 `news` 类别和现成句子边界。
本次选择完整 news 子集，让儿童故事词典迁移到新闻文本，直接测量跨领域词汇与组合覆盖。
它没有在本实验中使用的 gold CCG 推导，因此这里报告可解析率，不报告结构正确率。

## 固定设置与数据来源

- 源词典：仓库 TinyStories 样本，{source_summary['sentences']:,} 句、{source_summary['tokens']:,} token、
  {source_summary['word_types']:,} 个词形、{source_summary['category_pairs']:,} 个词—范畴对；不是完整 TinyStories。
- 源超标注器：`ryosuke-yamaki/hol-ccg-roberta-large`，C&C grammar mask 开启。
- 构建集合：各次出现 `k=8, beta=0.99` 候选的并集；`min_support=1`。
  分数来自完整分布的算术平均，再在集合内归一化。
- 目标：Brown 全部 {manifest['document_count']} 个 news 文档，原始 {manifest['raw_sentences']:,} 句。
  沿用原始句界，反分词后用项目 PTB 分词，去引号、保留大小写；
  过滤 {manifest['dropped_under_two_tokens']} 个少于 2 token 的句子，最终 {summary['sentences']:,} 句、{summary['tokens']:,} token。
  原始重复标点保留，不因失败而清洗或排除句子。
- 按预处理后的完整 token 序列检查，目标句子中有 {len(overlap)} 句与源样本完全相同；
  这项检查不代表能够排除神经模型预训练数据中的潜在重叠。
- 语法：项目默认 Eisner 正规形 CKY，根范畴 `S`（匹配其特征变体），组合阶数上限 2，
  无 beam，4 个解析线程，超过 80 token 跳过。两组语法一致。
- 词典组：使用所有词典候选，遇到 OOV 严格失败，不查询超标注器，不向词典加入目标语料信息。
- 参考组：在同一目标语料上使用上下文超标注器，先检查 1-best，失败后用 `k=8, beta=0.99` lattice。
  这是独立的解析参考组，不是 gold；与词典组的候选集合大小也不同。

官方归档：[NLTK brown.zip]({manifest['archive_url']})。
[NLTK 许可元数据]({manifest['metadata_url']}) 标注为非商业用途。
下载文件经过 SHA-256 校验；文档 ID、导出设置和数据校验值保存在 `data/brown_news.manifest.json`。

## 测评结果

| 指标 | 固定全局词典 |
| --- | ---: |
| token 候选覆盖率 | {pct(summary['token_candidate_coverage'])} |
| 目标词形覆盖率 | {pct(summary['word_type_coverage'])} |
| 未登录 token | {summary['oov_tokens']:,} / {summary['tokens']:,}（{pct(summary['oov_token_rate'])}） |
| 未登录词形 | {summary['oov_types']:,} / {summary['word_types']:,} |
| 整句所有词有候选 | {covered_n} / {summary['sentences']:,}（{pct(summary['sentence_candidate_coverage'])}） |
| OOV 导致失败 | {summary['reasons']['oov']:,} 句 |
| 所有词有候选但无 S 推导 | {summary['reasons']['no_derivation']} 句 |
| 超过 80 token 跳过 | {summary['skipped']} 句 |
| 每个有候选 token 的平均候选数 | {summary['avg_candidates_per_available_token']:.2f} |

| 相同目标句集 | 固定全局词典 | 上下文超标注参考组 |
| --- | ---: | ---: |
| 解析成功句数 | {summary['parsed']} | {ref_parsed:,} |
| 全部 {summary['sentences']:,} 句上的解析率 | {pct(summary['parse_rate_all'])} | {pct(ref_parsed / len(records))} |
| 排除超长句后的解析率 | {pct(summary['parse_rate_eligible'])} | {pct(ref_parsed / summary['eligible_sentences'])} |
| 在词典全词覆盖的 {covered_n} 句上解析成功 | {summary['parsed']} / {covered_n} | {ref_covered_parsed} / {covered_n} |

词典测评核心运行时间约 {summary['seconds']:.2f} 秒，参考组超标注 {ref_summary['tagging_seconds']:.2f} 秒、
解析 {ref_summary['parsing_seconds']:.2f} 秒；这些计时不含模型/词典加载，且参考组处理了更多可进入解析的句子，
不适合直接解释为两种方法的速度比。

{pct(summary['parse_rate_covered_eligible'])} 的条件解析率仅代表这个被词汇覆盖条件筛选出的短句集合，不能外推至全部新闻句子。
解析成功只表示当前语法下存在合法推导，不能保证词义、范畴或依存结构正确。

## 失败诊断

高频未登录项：

| 词形 | 次数 |
| --- | ---: |
"""
    for word, count in oov.most_common(12):
        text += f"| `{word}` | {count} |\n"
    text += f"""
这些缺口包含新闻用语、专名、称谓、数字形式和源样本未出现过的标点。
{sum(oov.values()):,} 个 OOV token 中，只有 {case_only:,} 个在源词典中存在大小写变体；
即使允许合并大小写，仍有 {sum(oov.values()) - case_only:,} 个 token 没有对应词形。
这只是词汇诊断，没有在本次测评中合并大小写或更改词典。

{len(no_derivation)} 个全词覆盖失败记录保存在 `output/brown_news_diagnostics.json`，包含原始文档句子 ID、
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
"""
    report = root / "docs/brown_news_experiment_2026-10-15.md"
    report.write_text(text, encoding="utf-8")
    print(json.dumps({"report": str(report), "global_parsed": summary["parsed"], "contextual_parsed": ref_parsed,
                      "global_parse_rate": summary["parse_rate_all"], "contextual_parse_rate": ref_parsed / len(records),
                      "covered_sentences": covered_n, "contextual_parsed_on_covered": ref_covered_parsed,
                      "case_only_oov_tokens": case_only}, indent=2))


if __name__ == "__main__":
    main()
