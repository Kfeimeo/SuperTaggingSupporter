# TinyStories 全量实验的运行与恢复

在其他主机下载数据及指定云盘目录的完整步骤见 [数据下载与云主机运行](data_download_and_cloud.md)。

本次使用原始 TinyStories 的**完整训练集**，不设故事数或句数上限。
validation 保留，不混入 GPT-4 V2。官方文件固定到版本
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`，大小 1,924,281,556 字节，
SHA-256 为 `c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f`。

全量标注在本机 GPU 执行，使用与此前相同的模型和 FP32 精度。
为减少逐词 GPU 调用，提供可选的向量化子词池化；启动前的实际模型验证记录在
`output/tinystories_full.benchmark.json`。原来的默认超标注路径不变。

全量流程会依次：校验源文件、构建完整词典、导出集合、测评固定 Brown news、生成比较报告。
最终报告只在源词典完整处理到文件末尾后生成；运行中的统计不作为全量实验结论。

## 查看进度

```powershell
Get-Content output/tinystories_full.status.json
Get-Content output/tinystories_full_run.log -Tail 3
Get-Content output/tinystories_full_run.err.log -Tail 5
```

`stage` 表示 `building_lexicon`、`evaluating_brown_news`、`complete` 或 `failed` 等状态。
构建阶段包含已完成故事数、句数、token 数、源文件字节比例和估计剩余时间。
剩余时间来自当前平均吞吐量，会受到句长、GPU 温度和其他程序负载影响；不含后续解析耗时。

词典使用 `output/tinystories_full.lexicon.sqlite`，避免把全部 dense softmax 向量放进内存或巨大 JSON。
每 128 篇完整故事提交一次事务，同时保存累计统计和原始文件偏移。
如果中断，最多重做尚未提交的一组故事，已提交词频不会重复累计。
单个超过模型 subword 容量的源句会拆成子上下文，所有 token 都会计入，拆分操作单独统计。
其他未处理错误会停止并记录，不会静默丢弃源句。

## 恢复

先确认原进程已退出。使用同一个项目目录和 `ccg` 环境：

```powershell
python -u scripts/run_full_tinystories_experiment.py --batch-size 64 --stories-per-checkpoint 128
```

脚本自带互斥锁，拒绝两个实例同时修改同一词典。数据哈希、模型路径、候选阈值和
分句后端等设置必须与检查点相同；更换设置时应使用独立实验文件。
如果构建已完成而后续测评失败，完整源词典仍保留，可单独重跑：

```powershell
python -m stsupport eval-lexicon --lexicon output/tinystories_full.lexicon.sqlite --corpus data/brown_news.jsonl --format jsonl --set output.name=brown_news_full_global_lexicon --set search.n_threads=4 --batch-size 4
python scripts/report_full_tinystories_experiment.py
```

解析阶段沿用全部词典候选、严格 OOV、默认语法、无 beam、80 token 句长上限。
扩大源语料可能显著增加每个词的候选数，因此解析耗时会比小样本增加。

完成后的报告为 `docs/tinystories_full_experiment.md`，并保留小样本及上下文参考组以便对比。
