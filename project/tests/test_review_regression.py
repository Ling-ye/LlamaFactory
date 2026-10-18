# Copyright 2026. Licensed under the Apache License, Version 2.0.

import pytest

from lf_tools.artifacts import read_json, seal, write_rows
from lf_tools.calibration import calibrate
from lf_tools.regression import score_regression


def test_regression_distinguishes_wrong_answer_and_format():
    refs = [{"id": str(i), "category": "reasoning", "answers": ["8"]} for i in range(3)]
    result = score_regression(["8。", "9", "答案是8"], refs)
    assert result["accuracy"] == pytest.approx(100 / 3)
    assert [r["passed"] for r in result["details"]] == [True, False, False]
    with pytest.raises(ValueError):
        score_regression([], refs)


def reviews(directory, reviewer_type, checksum="same", winner="A"):
    directory.mkdir()
    dims = {"identity": 3, "style": 4, "facts": None, "naturalness": 4}
    write_rows(
        directory / "reviews.jsonl",
        [
            {
                "case_id": "one",
                "case_sha256": checksum,
                "reviewer_type": reviewer_type,
                "winner": winner,
                "A": dims,
                "B": dims,
            }
        ],
    )
    seal(directory, {"kind": "reviews"})
    return directory


def test_calibration_uses_only_paired_evidence(tmp_path):
    model = reviews(tmp_path / "model", "model")
    human = reviews(tmp_path / "human", "human", winner="B")
    output = calibrate(model, human, tmp_path / "result")
    result = read_json(output / "calibration.json")
    assert result["paired_cases"] == 1
    assert result["winner_agreement"] == 0
    assert result["mean_absolute_score_difference"]["facts"] is None


def test_calibration_rejects_different_answers(tmp_path):
    model = reviews(tmp_path / "model", "model")
    human = reviews(tmp_path / "human", "human", checksum="changed")
    with pytest.raises(ValueError, match="different anonymous answers"):
        calibrate(model, human, tmp_path / "result")
