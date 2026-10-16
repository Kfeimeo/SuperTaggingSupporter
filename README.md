# SuperTaggingSupporter

Validation search for CCG supertagger output: *given per-token category
distributions, find the most probable tag sequence that actually derives a
sentence (`S`).*

```
corpus ──► hol-ccg supertagger ──► P(category | token)        (Python / PyTorch, GPU)
                                        │
                                        ▼
                      1-best lattice (k = 1) ──► Viterbi CKY ──► S?  ──yes──► done ("onebest")
                                        │ no
                                        ▼
            top-k / cumulative-β lattice ──► Viterbi CKY (Eisner NF) ──► best S-rooted
                                                                         derivation ("recovered")
                                                                         or "fail"
```

The CKY is pure C++ (`cpp/`, exposed through pybind11 as `stsupport._ccgcky`);
everything model-related stays in Python (`src/stsupport/`).

## What the C++ engine does

* **Categories** — CCGbank strings (`(S[dcl]\NP)/NP`, `NP[nb]/N`, `N-->NP` chains,
  `S[dcl][conj]`) are parsed and interned. A bare `S` is the feature variable
  `S[X]` (so `S/S  S[dcl] → S[dcl]`, `(S\NP)/(S\NP)  (S[b]\NP)/NP → (S[b]\NP)/NP`);
  a bare feature on other atoms is a wildcard (`NP` matches `NP[expl]`); `[nb]`
  is stripped as in CCGbank derivations.
* **Binary rules** — application `>` `<`, composition `>B` `>B²` `<B`, crossed
  composition `<Bx` `<Bx²` (restricted to `S\NP`-shaped `Y`, as in C&C),
  conjunction `conj X → X[conj]` / `X X[conj] → X` (also `,`/`;` as conj), and
  punctuation absorption.
* **Unary rules** — the C&C type-changing table (`N → NP`, `S[dcl]\NP → NP\NP`, …)
  and CCGbank type-raising targets (`trNP`, `trAP`, `trPP`, `trVP_to`), up to
  `max_unary_depth` steps.
* **Eisner normal form** — `>Bⁿ` output is never the primary functor of `>`/`>Bᵐ`
  (mirror image for `<B`, and for `<Bx` output as left functor), type-raised
  categories are never the primary functor of application, no TR of TR. The
  constraint is relaxed exactly when the alternative derivation would exceed
  `max_composition_degree`.
* **Viterbi** — chart cells keep the best item per (category, NF class); the
  score is the sum of supertag log-probabilities (rules cost 0 unless you set
  `unary_penalty`). Score ties prefer high punctuation attachment. The root
  must match `root_categories` (default `S` = any `S[f]`). Optional
  per-cell beam and CCGbank "seen rules" filter.
* **Threads** — `parse_batch` parses many sentences on all cores; the category
  store and rule caches are shared and lock-protected.

`k` and `β` are independent: a token keeps categories in decreasing probability
until the cumulative mass reaches `β` **or** `k` categories are kept, whichever
comes first (`k=1` is the 1-best check).

## Setup (Windows, conda env `ccg`)

The `ccg` env already has `torch` (CUDA), `transformers`, `holccg` and the
cached `ryosuke-yamaki/hol-ccg-roberta-large` bundle. Build tools used: VS 2026
(MSVC 14.51), CMake ≥ 3.20, `pybind11`, `scikit-build-core`, `ninja` (pip).

```powershell
conda activate ccg
pip install pybind11 scikit-build-core ninja      # once
pip install -e . --no-build-isolation             # builds cpp/ -> stsupport/_ccgcky.pyd
pytest -q
```

If `holccg` is not pip-installed in your env, set `tagger.hol_ccg_src` to the
hol-ccg checkout (the `src/` dir is put on `sys.path`).

## Usage

```powershell
# TinyStories sample (2000 sentences, data/tinystories_sample.txt), k=8, beta=0.95
python -m stsupport run --config configs/default.yaml --limit 500
python -m stsupport run --k 16 --beta 0.99 --set search.strategy=progressive --set output.name=k16

# tag once, compare several (k, beta) settings
python -m stsupport sweep --k 1,2,4,8,16,32 --beta 0.9,0.99 --limit 2000

# prepare a sentence-per-line corpus from a raw TinyStories dump
python -m stsupport prepare-corpus "E:\...\TinyStories-validation.txt" data\tinystories_valid.txt --limit 20000

# hand-written lattice: word|CAT:p|CAT:p ...
python -m stsupport check "Lily|N:0.9|NP:0.1" "found|S[dcl]\NP:0.6|(S[dcl]\NP)/NP:0.3" "a|NP[nb]/N:0.9" "ball|N:0.9" ".|.:0.99"
```

Outputs in `output/<name>.*`:

* `.jsonl` — one record per sentence: `status` (`onebest` / `recovered` / `fail` /
  `skipped`), `onebest_tags`, `final_tags`, `changed_positions`, `logprob`,
  `lattice_sizes`, `k_used`, `beta_used`, `auto`, and for failures the labels
  present in the full-span cell (`root_cell_cats`).
* `.auto` — CCGbank-style derivations (`ID=... STATUS=...` header lines).
* `.summary.json` — counts, `onebest_rate`, `parse_rate`, timing.

### Python API

```python
from stsupport import CKYEngine, Validator, load_config
from stsupport.supertagger import HolCCGSupertagger

cfg = load_config("configs/default.yaml")
tagger = HolCCGSupertagger(cfg.tagger.model, device="cuda")
engine = CKYEngine(cfg.grammar)
validator = Validator(engine, tagger.categories, cfg.search, exclude_ids=tagger.excluded_ids)

tokens = ["Mia", "found", "the", "ball", "."]
probs = tagger.tag([tokens])[0]          # (5, 1341) softmax over supertags
res = validator.validate(tokens, probs)  # SearchResult
print(res.status, res.final_tags, res.auto)
```

`engine.combine(left, right)`, `engine.unary(cat)`, `engine.normalize(cat)` expose
the grammar for debugging.

## Global lexicon experiments

Build a fixed word-to-category dictionary from a source corpus, then parse any
target corpus without loading the supertagger:

```powershell
python -m stsupport build-lexicon --corpus data/tinystories_sample.txt --format txt --lexicon output/tinystories.lexicon.json --k 8 --beta 0.99
python -m stsupport eval-lexicon --lexicon output/tinystories.lexicon.json --corpus "D:/corpora/my_test.txt" --format txt --set output.name=my_test
```

The lexicon stores full soft probability counts separately from the union of
per-occurrence candidates. Evaluation uses the entire dictionary candidate set,
strict OOV failures, and reports lexical coverage and parse rates separately.
See [the experiment guide](docs/global_lexicon_experiment.md) for probability
aggregation, custom TXT/JSONL corpora, metrics, and controlled comparisons.
The completed [Brown news experiment](docs/brown_news_experiment_2026-10-15.md)
includes the full target subset, an independent contextual-supertagging reference,
data checksums, failure diagnostics, and reproduction commands.
The [full TinyStories workflow](docs/tinystories_full_run.md) uses a pinned full
training split, transactional SQLite checkpoints, and automatic Brown news evaluation.
For a fresh cloud host, see [data download and cloud usage](docs/data_download_and_cloud.md).
The download-only modes require just Python's standard library:

```bash
python scripts/download_tinystories.py --split all --output-dir /mnt/datasets/raw
python scripts/prepare_brown_news.py --data-dir /mnt/datasets --download-only
```

## Results on the TinyStories sample

`data/tinystories_sample.txt` → 2226 sentences, `ryosuke-yamaki/hol-ccg-roberta-large`,
default grammar, RTX 3070 Laptop (tagging 16 s; parsing times are for the whole corpus
on all cores):

| k | β | 1-best derives S | recovered by lattice | fail | parse rate | cands/token | parse s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | – | 1912 | 0 | 314 | 85.9 % | 1.00 | 5.5 |
| 2 | 0.99 | 1912 | 213 | 101 | 95.5 % | 1.06 | 6.4 |
| 4 | 0.99 | 1912 | 257 | 57 | 97.4 % | 1.13 | 7.5 |
| 8 | 0.95 | 1912 | 222 | 92 | 95.9 % | 1.08 | 6.4 |
| 8 | 0.99 | 1912 | 270 | 44 | 98.0 % | 1.23 | 9.9 |
| 16 | 0.99 | 1912 | 275 | 39 | 98.2 % | 1.35 | 16.8 |
| 32 | 0.99 | 1912 | 276 | 38 | 98.3 % | 1.49 | 40.0 |

Recovered sentences change 1.2 tokens on average. Typical repairs:
`PP/NP ↔ (S/S)/NP` (*Once upon a time*), `((S\NP)\(S\NP))/NP → PP/NP` (*with her mom*),
`(S[dcl]\NP)/S[wq] → (S[dcl]\NP)/S[q]`. Remaining failures are mostly imperatives
(CCGbank root `S[b]\NP`), un-quoted direct speech (`S[qem]` expected, `S[q]` found)
and fragments (`Nice to meet you !`).

## Layout

```
cpp/            category.{hpp,cpp}  grammar.{hpp,cpp}  parser.{hpp,cpp}  bindings.cpp
src/stsupport/  config.py  engine.py  lattice.py  supertagger.py  pipeline.py  corpus.py  cli.py
configs/        default.yaml
data/           tinystories_sample.txt
tests/          test_engine.py  test_lattice.py
```

## Notes

* Probabilities are the hol-ccg word-classifier softmax with `<unk>` and the
  candc-unlicensed chain categories masked out (`tagger.grammar_mask`), so `β`
  refers to licensed categories only.
* CCGbank analyses imperatives (`Let 's play !`) as `S[b]\NP` at the root, not
  as `S`; with the default `root_categories: ["S"]` they count as failures. Add
  `"S[b]\\NP"` to `grammar.root_categories` if you want them accepted.
* `use_seen_rules: true` restricts binary combinations to child pairs seen in
  CCGbank 02–21 (hol-ccg's `grammar_rule_counter.json`); it makes derivations
  more CCGbank-like but fewer sentences reach `S`.
* vcpkg is not required; pybind11 comes from pip.
