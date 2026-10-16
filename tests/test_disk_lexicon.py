from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from stsupport.disk_lexicon import LexiconStore
from stsupport.lexicon import GlobalLexicon


def test_transactional_resume_matches_in_memory_aggregation(tmp_path):
    categories = ["N", "NP", "S"]
    metadata = {"lowercase": False, "source": {"drop_quotes": True}}
    path = tmp_path / "lex.sqlite"
    tokens = ["duck", "duck", "other"]
    p = np.array([[0.9, 0.09, 0.01], [0.2, 0.79, 0.01], [0.1, 0.2, 0.7]])
    expected = GlobalLexicon(categories)
    expected.add(tokens, p, k=1)
    first = GlobalLexicon(categories)
    first.add(tokens[:1], p[:1], k=1)
    store = LexiconStore(path, categories, metadata)
    store.commit(first, {"offset": 100})
    store.close()
    resumed = LexiconStore(path, categories, metadata)
    assert resumed.progress == {"offset": 100}
    assert resumed.summary()["tokens"] == 1
    rest = GlobalLexicon(categories)
    rest.add(tokens[1:], p[1:], k=1)
    resumed.commit(rest, {"offset": 200, "complete": True})
    assert resumed.summary() == {"word_types": 2, "tokens": 3, "category_pairs": 3}
    resumed.export_map(tmp_path / "map.json")
    resumed.close()
    actual = GlobalLexicon.load(path)
    assert actual.counts == expected.counts
    assert actual.category_map() == expected.category_map()
    for word in expected.counts:
        np.testing.assert_allclose(actual.distribution(word), expected.distribution(word))
        assert actual.candidates(word) == expected.candidates(word)
    with pytest.raises(RuntimeError, match="read-only"):
        actual.add(tokens, p)
    actual.close()
    with pytest.raises(ValueError, match="settings differ"):
        LexiconStore(path, categories, {"different": True})


def test_story_offsets_resume_inline_markers_and_utf8(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("full_runner", scripts / "run_full_tinystories_experiment.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    finally:
        sys.path.remove(str(scripts))
    path = tmp_path / "stories.txt"
    path.write_bytes("First café.<|endoftext|>Second story.\r\n<|endoftext|>\nThird.".encode("utf-8"))
    records = list(runner.stories(path))
    assert [text for text, _ in records] == ["First café.", "Second story.", "Third."]
    for index, (_, offset) in enumerate(records):
        assert list(runner.stories(path, offset)) == records[index + 1:]


def test_child_progress_is_visible_and_logged(tmp_path, monkeypatch, capsys):
    import subprocess
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    import run_full_tinystories_experiment as runner
    log = tmp_path / "cpu.log"
    runner.run_logged([sys.executable, "-c", "import sys; sys.stderr.write('CPU: 0%\\rCPU: 100%\\n'); print('summary')"], log)
    terminal = capsys.readouterr().err
    assert "CPU: 100%" in terminal and "summary" in terminal
    assert b"0%\rCPU: 100%" in log.read_bytes()
    with pytest.raises(subprocess.CalledProcessError):
        runner.run_logged([sys.executable, "-c", "import sys; print('parser failed'); sys.exit(2)"], tmp_path / "failed.log")
    assert "parser failed" in (tmp_path / "failed.log").read_text()
