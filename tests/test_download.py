from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


@pytest.fixture
def downloader(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    import download_tinystories
    return download_tinystories


class Response(io.BytesIO):
    def __init__(self, payload, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {}


def install_responses(monkeypatch, module, responses):
    requests = []
    responses = iter(responses)
    def open_response(request, **kwargs):
        requests.append(request)
        return next(responses)
    monkeypatch.setattr(module.urllib.request, "urlopen", open_response)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return requests


def test_resume_verified_payload(tmp_path, downloader, monkeypatch):
    payload = b"hello cloud corpus"
    target = tmp_path / "train.txt"
    target.with_suffix(".txt.part").write_bytes(payload[:5])
    requests = install_responses(monkeypatch, downloader, [Response(payload[5:], 206, {"Content-Range": f"bytes 5-{len(payload)-1}/{len(payload)}"})])
    downloader.download_file("https://example.test/data", target, hashlib.sha256(payload).hexdigest(), len(payload))
    assert requests[0].get_header("Range") == "bytes=5-"
    assert target.read_bytes() == payload
    assert not target.with_suffix(".txt.part").exists()


def test_server_ignoring_range_restarts_without_duplicate_bytes(tmp_path, downloader, monkeypatch):
    payload = b"complete data"
    target = tmp_path / "data.txt"
    target.with_suffix(".txt.part").write_bytes(payload[:3])
    install_responses(monkeypatch, downloader, [Response(payload)])
    downloader.download_file("https://example.test/data", target, hashlib.sha256(payload).hexdigest(), len(payload))
    assert target.read_bytes() == payload


def test_short_response_retries_from_saved_offset(tmp_path, downloader, monkeypatch):
    payload = b"whole dataset"
    target = tmp_path / "data.txt"
    requests = install_responses(monkeypatch, downloader, [Response(payload[:4]), Response(payload[4:], 206, {"Content-Range": f"bytes 4-{len(payload)-1}/{len(payload)}"})])
    downloader.download_file("https://example.test/data", target, hashlib.sha256(payload).hexdigest(), len(payload), retries=1)
    assert [r.get_header("Range") for r in requests] == [None, "bytes=4-"]
    assert target.read_bytes() == payload


def test_bad_range_does_not_modify_partial(tmp_path, downloader, monkeypatch):
    target = tmp_path / "data.txt"
    partial = target.with_suffix(".txt.part")
    partial.write_bytes(b"ab")
    install_responses(monkeypatch, downloader, [Response(b"cd", 206, {"Content-Range": "bytes 1-3/4"})])
    with pytest.raises(ValueError, match="Content-Range"):
        downloader.download_file("https://example.test/data", target, "0" * 64, 4)
    assert partial.read_bytes() == b"ab"
    assert not target.exists()


def test_hash_mismatch_is_never_published(tmp_path, downloader, monkeypatch):
    target = tmp_path / "data.txt"
    install_responses(monkeypatch, downloader, [Response(b"wrong")])
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        downloader.download_file("https://example.test/data", target, "0" * 64, 5)
    assert not target.exists()
    assert target.with_suffix(".txt.part").read_bytes() == b"wrong"


def test_complete_partial_and_offline_validation_need_no_network(tmp_path, downloader, monkeypatch):
    payload = b"offline"
    target = tmp_path / "data.txt"
    target.with_suffix(".txt.part").write_bytes(payload)
    def no_network(*args, **kwargs):
        raise AssertionError("network should not be used")
    monkeypatch.setattr(downloader.urllib.request, "urlopen", no_network)
    sha = hashlib.sha256(payload).hexdigest()
    downloader.download_file("https://example.test/data", target, sha, len(payload))
    downloader.download_file("https://example.test/data", target, sha, len(payload), verify_only=True)
    with pytest.raises(FileNotFoundError):
        downloader.download_file("https://example.test/data", tmp_path / "missing", sha, len(payload), verify_only=True)
    target.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        downloader.download_file("https://example.test/data", target, sha, len(payload))
    assert target.read_bytes() == b"corrupt"


def test_cli_split_all_and_custom_directory(tmp_path, downloader, monkeypatch):
    calls = []
    monkeypatch.setattr(downloader, "download", lambda root, **kwargs: calls.append(kwargs))
    assert downloader.main(["--split", "all", "--output-dir", str(tmp_path), "--verify-only"]) == 0
    assert [c["split"] for c in calls] == ["train", "validation"]
    assert all(c["output_dir"] == tmp_path and c["verify_only"] for c in calls)


def test_cloud_report_accepts_missing_local_baselines(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location("full_report", scripts / "report_full_tinystories_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "output"
    out.mkdir()
    (tmp_path / "docs").mkdir()
    corpus = tmp_path / "external_data" / "brown_news.jsonl"
    corpus.parent.mkdir()
    corpus.write_text('{"tokens":["Hello","."]}\n', encoding="utf-8")
    corpus.with_suffix(".manifest.json").write_text(json.dumps({"output_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest()}))
    source = {"complete": True, "stories": 1, "sentences": 1, "tokens": 2, "word_types": 2,
              "category_pairs": 2, "elapsed_seconds": 10, "metadata": {"source": {"sha256": "hash"}}}
    full = {"sentences": 1, "covered_sentences": 1, "parsed": 1, "seconds": 1,
            "target_corpus": {"path": str(corpus)}, "avg_candidates_per_available_token": 1,
            **{k: 1 for k in ("token_candidate_coverage", "word_type_coverage", "sentence_candidate_coverage", "parse_rate_all", "parse_rate_covered_eligible")}}
    (out / "tinystories_full.lexicon.summary.json").write_text(json.dumps(source))
    (out / "brown_news_full_global_lexicon.summary.json").write_text(json.dumps(full))
    module.main(tmp_path)
    report = (tmp_path / "docs/tinystories_full_experiment.md").read_text(encoding="utf-8")
    assert "未运行" in report and "100.00%" in report
