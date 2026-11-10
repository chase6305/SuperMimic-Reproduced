import json

import pytest

from supermimic.eval.benchmark import ClipResult, save_results, summarize


def test_short_failure_has_separate_executed_and_full_clip_scores(tmp_path):
    failed = ClipResult("failed", 101, 10, 0.8, True, 10)
    complete = ClipResult("complete", 51, 50, 0.6, False, -1)
    result = summarize([failed, complete])
    assert result["completed_clips"] == 1
    assert result["mean_norm_return"] == pytest.approx(0.7)
    assert result["mean_full_clip_zero_padded_return"] == pytest.approx(0.34)
    assert result["mean_clip_coverage"] == pytest.approx(0.55)
    save_results([failed, complete], tmp_path / "evaluation.json")
    saved = json.loads((tmp_path / "evaluation.json").read_text())
    assert saved["clips"][0]["requested_steps"] == 100
    assert saved["clips"][0]["full_clip_zero_padded_return"] == pytest.approx(0.08)
