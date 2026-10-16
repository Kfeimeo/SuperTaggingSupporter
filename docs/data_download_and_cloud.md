# 数据下载与云主机运行

下载脚本不依赖本机用户名、Windows 盘符或 conda 环境名。
TinyStories 下载以及 Brown 的 `--download-only` 模式只使用 Python 标准库，
可先下载数据，再安装 GPU 实验环境。

## 1. 下载 TinyStories

进入项目根目录：

```bash
# 原始版本的完整训练集；默认保存至项目 data/raw。
python scripts/download_tinystories.py --split train

# 同时下载 train 和 validation，并指定云盘目录。
python scripts/download_tinystories.py --split all --output-dir /mnt/datasets/raw
```

| `--split` | 输出 | 大小 |
| --- | --- | ---: |
| `train`（默认） | `TinyStories-train.txt` | 1,924,281,556 字节 |
| `validation` | `TinyStories-valid.txt` | 19,447,282 字节 |
| `all` | 上述两个文件 | 约 1.94 GB |

数据固定为原始版本 `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`，
使用官方 SHA-256 校验，不混入 GPT-4 V2。
来源：[train](https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStories-train.txt)、
[validation](https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStories-valid.txt)。

每个文件旁会生成 `.manifest.json`，记录版本、下载地址、字节数和校验值。
validation 单独保存，当前词典实验只读取 train。

## 2. 续传、重试与离线校验

下载中断后，**重新运行相同命令**即可从 `.part` 文件继续：

```bash
python scripts/download_tinystories.py --split all --output-dir /mnt/datasets/raw --retries 8 --timeout 120
```

支持 HTTP Range 续传；服务端忽略 Range 时会重新下载该 partial，避免重复拼接。
正常文件名仅在字节数和 SHA-256 都通过后生成。
已经存在且校验通过的完整文件会复用，损坏文件会报错并保留；校验失败时先手动移走损坏文件再重新下载。
已经下载完整但尚未改名的 `.part` 文件可以直接校验并改名，无需请求网络。

只校验本地文件，不发起下载：

```bash
python scripts/download_tinystories.py --split all --output-dir /mnt/datasets/raw --verify-only
```

参数：`--retries` 为初次尝试后的重试次数，默认 5；`--timeout` 为单次网络操作超时秒数，默认 60；
`--progress-interval` 为进度输出间隔秒数，默认 10。
网络访问使用 Python 标准库的代理配置，可按云主机环境设置 `HTTPS_PROXY`。

## 3. 下载并准备 Brown news

先下载归档，这一步不需要 NLTK 或 GPU：

```bash
python scripts/prepare_brown_news.py --data-dir /mnt/datasets --download-only
```

安装项目及 NLTK 后，运行同一脚本导出测评语料：

```bash
python scripts/prepare_brown_news.py --data-dir /mnt/datasets
```

已有归档校验通过时不会重复下载。输出：

```text
/mnt/datasets/
  raw/
    TinyStories-train.txt
    TinyStories-train.manifest.json
    TinyStories-valid.txt             # 仅 --split all/validation 下载
    TinyStories-valid.manifest.json
    brown.zip
  brown_news.jsonl
  brown_news.manifest.json
```

Brown 也支持 `--retries`、`--timeout` 和 `--verify-only`；后者只离线校验归档，不重新导出。
导出保留原始句界，使用项目的 PTB 分词、去引号和长度过滤，输出统一为 UTF-8、LF 换行。
归档的固定 SHA-256 和预处理设置保存在 manifest 中。

## 4. 云主机启动全量实验

整个实验要求 Python 3.12+、可用的 CUDA PyTorch、兼容的 `holccg`、本项目及 C++ 构建工具链。
与本机一样，需要安装你使用的 hol-ccg 源码包；若两个仓库并排放置，可运行：

```bash
python -m pip install -e ../hol-ccg
python -m pip install -e .
```

hol-ccg 的 PyTorch/CUDA 依赖应与云主机驱动环境匹配，安装要求以对应仓库配置为准。
项目依赖已声明 NLTK 和 filelock。超标注模型由首次运行自动获取或复用 Hugging Face 缓存，
模型权重不包含在语料下载脚本中。

先测试 GPU 推理，再开始全量构建：

```bash
python scripts/run_full_tinystories_experiment.py --benchmark-only --data-dir /mnt/datasets
python -u scripts/run_full_tinystories_experiment.py --data-dir /mnt/datasets --batch-size 64 --stories-per-checkpoint 128
```

默认数据目录仍为项目的 `data/`，不传 `--data-dir` 即可沿用现有命令。
运行中断后重新执行相同命令，从 SQLite 检查点恢复。
检查点的源数据、模型路径和分句配置必须与创建时一致；云主机首次运行使用自己的输出目录。

Linux 通过 SSH 跑长任务时，也可将进程放到后台：

```bash
mkdir -p output
nohup python -u scripts/run_full_tinystories_experiment.py --data-dir /mnt/datasets --batch-size 64 > output/tinystories_full_cloud.log 2>&1 &
tail -f output/tinystories_full_cloud.log
```

进度保存在项目的 `output/tinystories_full.status.json`。
构建完成后自动测评该数据目录中的 Brown news，并生成 `docs/tinystories_full_experiment.md`。
新云主机没有历史小样本/上下文参考结果时，报告会将这些基线标为“未运行”，仍能正常生成全量结果。

如需补跑上下文参考组：

```bash
python -m stsupport run --corpus /mnt/datasets/brown_news.jsonl --k 8 --beta 0.99 --set corpus.format=jsonl --set search.n_threads=4 --set output.name=brown_news_contextual_reference
python scripts/report_full_tinystories_experiment.py
```

## 5. GPU batch size 与 CPU 解析线程参数

这两个阶段的批量分别配置，单位均为句子数：

| 命令 | 参数 | 作用与默认值 |
| --- | --- | --- |
| `run_full_tinystories_experiment.py` | `--batch-size N` | GPU 超标注每批句数，默认 64 |
| `run_full_tinystories_experiment.py` | `--pipeline-depth N` | GPU 阶段等待 CPU 汇总的批次数上限，默认 2 |
| `run_full_tinystories_experiment.py` | `--async-workers N` | GPU 阶段概率汇总的 CPU 后台线程上限，默认 1；另有 1 个固定预取线程 |
| `run_full_tinystories_experiment.py` | `--no-async` | 关闭 GPU 阶段的 CPU 预取和异步汇总，用于同步对照 |
| `run_full_tinystories_experiment.py` | `--parse-threads N` | 自动衔接的 CPU 解析线程数，默认 4；0 为硬件线程数 |
| `run_full_tinystories_experiment.py` | `--parse-batch-size N` | 自动衔接的 CPU 解析每批句数，默认 4 |
| `stsupport run` / `build-lexicon` | `--set tagger.batch_size=N` | GPU 超标注每批句数，默认 32 |
| `stsupport eval-lexicon` | `--set search.n_threads=N` | C++ 句子间并行解析线程数；不加载配置文件时默认 0，自动使用硬件线程数 |
| `stsupport eval-lexicon` | `--batch-size N` | 每批提交的目标句数，默认 16；限制该批可用并行度 |

配置文件或已有 `--set` 设置可以覆盖上述配置项的默认值。

### GPU：增大超标注 batch size

例如在 GPU 主机上每批处理 128 句：

```bash
python -u scripts/run_full_tinystories_experiment.py --data-dir /mnt/datasets --batch-size 128 --stories-per-checkpoint 128
```

显存充足时可以继续尝试 256。全量脚本遇到 CUDA 显存不足会拆小批重试，因此实际每批句数可能小于指定值。
增大批量不保证同比加速，效果受句长、padding 和 GPU 性能影响。
`--stories-per-checkpoint` 控制检查点提交间隔，不是 GPU batch size。

普通超标注参考组使用配置覆盖参数：

```bash
python -m stsupport run --corpus /mnt/datasets/brown_news.jsonl --k 8 --beta 0.99 --set corpus.format=jsonl --set tagger.batch_size=128 --set search.n_threads=16 --set output.name=brown_news_contextual_reference
```

### CPU：增加句子间并行解析

全量词典构建完成后，可以单独用 16 个线程、每批 64 句测评：

```bash
python -m stsupport eval-lexicon --lexicon output/tinystories_full.lexicon.sqlite --corpus /mnt/datasets/brown_news.jsonl --format jsonl --set search.n_threads=16 --batch-size 64 --set output.name=brown_news_full_global_lexicon
python scripts/report_full_tinystories_experiment.py
```

`search.n_threads=0` 会自动使用硬件线程数。每个线程独立解析一个句子，单句内部的 CKY 计算仍是单线程。
实际并行数不超过线程数和当批实际需要解析的句数；OOV 或句长超限的句子不会进入解析。
增加线程会同时增加图表内存占用。

全量流程可以用**同一条命令**配置 GPU 和 CPU 两个阶段，自动依次执行：

```bash
python -u scripts/run_full_tinystories_experiment.py --data-dir /mnt/datasets --batch-size 128 --parse-threads 16 --parse-batch-size 64 --stories-per-checkpoint 128
```

其中 GPU 每批 128 句；完成词典后自动切换至 CPU，使用 16 个线程、每批 64 句。
不传 CPU 参数时仍默认 4 线程、每批 4 句。无需另外启动 `eval-lexicon`；上面的独立命令用于单独重跑测评。
CPU 设置也会记录在测评摘要和最终报告中。

### GPU 阶段的 CPU/GPU 异步流水线

默认启用，无需另外启动进程。流水线按以下方式工作：

1. 一个 CPU 后台线程提前读取、分句、分词并排序下一组故事，最多预取一组。
2. 主线程独占 GPU 模型，推理当前批次。
3. `--async-workers` 个 CPU 汇总线程计算软计数和范畴支持次数，同时主线程继续下一批 GPU 推理。
   多线程时按词分配到独立队列，同一个词的出现顺序和累加顺序保持一致，各线程不共享可变统计数组。
4. 每组所有概率汇总完成后，主线程将统计与源文件偏移一起提交 SQLite。

```bash
python -u scripts/run_full_tinystories_experiment.py --data-dir /mnt/datasets --batch-size 128 --async-workers 4 --pipeline-depth 2 --parse-threads 16 --parse-batch-size 64
```

`--pipeline-depth 2` 表示最多有两批概率任务正在汇总或排队；加上当前 GPU 批次的结果，
内存中可能同时存在三批原始概率缓冲，以及汇总线程的行切片、归一化等临时数组。
队列满时会等待 CPU，避免缓存随语料增长。
增大这个值主要增加主机内存使用，不会改变 GPU batch size。
模型内部的子词输入准备、GPU 推理和结果回传仍由主线程执行；实际提速取决于 CPU/GPU 负载。
`--async-workers 4` 配置最多 4 个概率汇总线程，另有 1 个固定的预取线程；
默认 `--async-workers 1` 保留原来的单线程汇总行为。
`--parse-threads 16` 独立控制后续 CPU 解析的 16 个线程。
`--pipeline-depth` 按完整批次数限制在途任务，每批会分配给各汇总线程，不要求该值大于线程数。
更多线程会增加临时内存与调度开销，实际提速也受 Python GIL、词频分布和 NumPy 运算影响，不保证线性增长。
指定 `--no-async` 时不启动这些后台线程，`--async-workers` 不生效。

同步对照命令：

```bash
python -u scripts/run_full_tinystories_experiment.py --data-dir /mnt/datasets --batch-size 128 --no-async
```

异步与同步使用相同的故事分组，并保持每个词的累加顺序，可沿用同配置的既有检查点；
改变 `--async-workers` 不改变词典统计定义。
读取预取或概率汇总失败会停止任务，未完成的故事组不会提交；恢复时重做该组。
总进度百分比仍以已提交数据为准。主机上已经运行的进程需要更新代码后从检查点重新启动才会启用流水线。

## 6. 阶段标签与 tqdm 进度条

全量命令自动打印以下阶段，并在 GPU、CPU 阶段显示各自的 tqdm 进度条：

```text
[1/4 PREPARE] Verify source data and load GPU model
[2/4 GPU] Source processed: ...% .../1.92G ... [stories=..., sentences=..., batch=...]
[3/4 CPU] Parse Brown news using the fixed global lexicon
[CPU] Parse sentences: ...% .../4588 ... [parsed=..., failed=..., skipped=...]
[4/4 REPORT] Generate experiment report
[COMPLETE] All experiment stages finished
```

GPU 总进度按源文件已完成并提交检查点的字节量计，不代表故事数或句数百分比；
右侧还显示当前组的超标注句子进度。恢复运行时总进度从已保存偏移继续，ETA 按恢复后的处理速度估计。
CPU 总进度按目标语料的有效句数计，包括解析失败、OOV 和超长跳过的句子，按批次完成后更新。
CPU 开始前会先扫描目标语料以确定总句数，不会一次将整个语料载入内存。
单批解析尚未完成时，CPU 进度条会暂时停留在上一批的计数。

自动调用的 CPU 子进程输出会实时显示在终端，同时保存在
`output/brown_news_full_global_lexicon.log`。GPU 的 JSON 检查点日志和状态文件仍然保留。
使用 `nohup` 时，进度输出进入重定向日志，可用 `tail -f` 查看。

如只需要普通阶段日志，可关闭进度条：

```bash
python -u scripts/run_full_tinystories_experiment.py --data-dir /mnt/datasets --batch-size 128 --no-progress
python -m stsupport eval-lexicon --lexicon output/tinystories_full.lexicon.sqlite --corpus /mnt/datasets/brown_news.jsonl --set output.name=brown_news_full_global_lexicon --no-progress
```

## 验证范围

已测试续传、服务端忽略 Range、短响应重试、错误 Range、哈希失败、完整 partial 恢复、
离线校验、自定义目录，以及新主机缺少历史基线时的报告生成。
异步流水线已测试单组预取上限、CPU/GPU 调用端重叠、后台异常传播、
同步/异步软计数逐元素一致，以及失败组不会推进 SQLite 检查点。
本机使用 `python -S` 验证了下载/校验模式不依赖第三方包，并实际检查了已下载的官方训练文件。
云主机的 CUDA 与驱动环境需要在目标机器运行 benchmark 确认。
