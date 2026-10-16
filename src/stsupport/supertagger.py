"""hol-ccg as a supertagger: per-token probability distributions over categories.

Only the encoder + word-category classifier of a HolCCG bundle is used; the
neural span/phrase classifiers and hol-ccg's own CKY are not involved.
"""

from __future__ import annotations

import logging
import contextlib
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _import_holccg(hol_ccg_src: str | None):
    try:
        import holccg  # noqa: F401
    except ImportError:
        if hol_ccg_src is None:
            raise
        src = Path(hol_ccg_src)
        if (src / "src").is_dir():
            src = src / "src"
        sys.path.insert(0, str(src))
        import holccg  # noqa: F401


def resolve_bundle(model: str) -> Path:
    """Local bundle directory for a path or HF repo id; cached snapshots are
    used without contacting the Hub, a download happens only on a cache miss."""

    path = Path(model)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(model, local_files_only=True))
    except Exception:
        logger.info("model %s not in the local HF cache; downloading", model)
        return Path(snapshot_download(model))


class HolCCGSupertagger:
    """Wraps a HolCCG bundle and exposes ``tag(sentences) -> list[(L, C) probs]``."""

    def __init__(
        self,
        model: str = "ryosuke-yamaki/hol-ccg-roberta-large",
        *,
        device: str = "cuda",
        grammar_mask: bool = True,
        hol_ccg_src: str | None = None,
        autocast_bf16: bool = False,
    ) -> None:
        _import_holccg(hol_ccg_src)
        import torch
        from holccg.parsing.inference import HolCCGParser, InferenceOptions

        self.torch = torch
        self.bundle_dir: Path = resolve_bundle(model)
        options = InferenceOptions(grammar="candc" if grammar_mask else "treebank", autocast_bf16=autocast_bf16)
        self.parser = HolCCGParser.from_pretrained(self.bundle_dir, options=options, device=device)
        self.model = self.parser.model
        self.device = self.parser.device
        self.autocast_bf16 = autocast_bf16

        index2cat = self.model.word_index2category
        self.categories: list[str] = [c if c is not None else "<none>" for c in index2cat]
        unk = self.model.word_category2index.get("<unk>", 0)
        excluded = {int(unk)}
        if grammar_mask and getattr(self.model, "_word_mask_ids", None):
            excluded.update(int(i) for i in self.model._word_mask_ids)
        self.excluded_ids: list[int] = sorted(excluded)
        self._mask_tensor = None

    # ------------------------------------------------------------------
    def _mask(self, logits):
        torch = self.torch
        if not self.excluded_ids:
            return logits
        if self._mask_tensor is None or self._mask_tensor.device != logits.device:
            self._mask_tensor = torch.as_tensor(self.excluded_ids, dtype=torch.long, device=logits.device)
        return logits.index_fill(-1, self._mask_tensor, float("-inf"))

    def _encode_vectorized(self, sentences):
        """Equivalent subword means and normalization, with batched GPU operations."""
        torch = self.torch
        model = self.model
        encoding = model.tokenizer(sentences, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        width = encoding["input_ids"].shape[1]
        lengths = [len(s) for s in sentences]
        source_ids, word_ids, base = [], [], 0
        for row, length in enumerate(lengths):
            alignment = encoding.word_ids(row)
            present = set()
            for col, word in enumerate(alignment):
                if word is not None:
                    present.add(word)
                    source_ids.append(row * width + col)
                    word_ids.append(base + word)
            if len(present) != length:
                raise ValueError("sentence exceeds model subword capacity; split it explicitly")
            base += length
        device = next(model.encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in encoding.items()}
        autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if self.autocast_bf16 else contextlib.nullcontext()
        with autocast:
            hidden = model.encoder(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), return_dict=True).last_hidden_state
        hidden = hidden.float()
        source = torch.as_tensor(source_ids, device=device, dtype=torch.long)
        target = torch.as_tensor(word_ids, device=device, dtype=torch.long)
        pooled = torch.zeros((base, hidden.shape[-1]), device=device, dtype=hidden.dtype)
        pooled.index_add_(0, target, hidden.reshape(-1, hidden.shape[-1]).index_select(0, source))
        pooled /= torch.bincount(target, minlength=base).to(pooled.dtype).unsqueeze(-1)
        vectors = model.projection(pooled)
        if model.config.normalization_type == "complex":
            vectors = model.complex_normalize(vectors)
        else:
            vectors = torch.nn.functional.normalize(vectors, dim=-1) * model.vector_norm
        return list(vectors.split(lengths))

    def tag(self, sentences: Sequence[Sequence[str]], batch_size: int = 32, *, vectorized_pooling: bool = False) -> list[np.ndarray]:
        """Return one ``(len(tokens), n_categories)`` float32 probability matrix per sentence.

        Excluded categories (``<unk>`` and, with ``grammar_mask``, unlicensed
        chains) receive probability 0 before the softmax, so the remaining mass
        sums to 1 and ``beta`` refers to licensed categories only.
        """

        torch = self.torch
        from holccg.utils.ccg_normalize import normalize_ptb_tokens

        results: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(sentences), max(1, batch_size)):
                chunk = [list(map(str, s)) for s in sentences[start : start + batch_size]]
                enc = [normalize_ptb_tokens(words) for words in chunk]
                if vectorized_pooling:
                    vecs = self._encode_vectorized(enc)
                else:
                    try:
                        vecs = self.model._encode_words_batch(enc, autocast_bf16=self.autocast_bf16)
                    except Exception as error:  # non-fast tokenizer or alignment issue: per sentence
                        logger.debug("batched encode failed (%s); falling back per sentence", error)
                        vecs = [self.model._encode_words(words, autocast_bf16=self.autocast_bf16) for words in enc]
                lengths = [v.shape[0] for v in vecs]
                flat = torch.cat(vecs, dim=0)
                logits = self._mask(self.model.word_category_classifier(flat))
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                offset = 0
                for length in lengths:
                    results.append(probs[offset : offset + length])
                    offset += length
        return results

    def onebest(self, probs: np.ndarray) -> list[str]:
        return [self.categories[int(i)] for i in probs.argmax(axis=-1)]
