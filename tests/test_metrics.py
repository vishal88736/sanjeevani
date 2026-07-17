"""Unit tests for evaluation/metrics.py (WER/CER computation)."""

from __future__ import annotations

import pytest

from evaluation.metrics import compute_cer, compute_wer


def test_wer_perfect_match_is_zero():
    refs = ["mujhe bukhar hai", "sar dard ho raha hai"]
    hyps = ["mujhe bukhar hai", "sar dard ho raha hai"]
    assert compute_wer(refs, hyps) == 0.0


def test_wer_single_substitution():
    refs = ["mujhe bukhar hai"]
    hyps = ["mujhe sardi hai"]
    # 1 substitution out of 3 reference words
    assert compute_wer(refs, hyps) == pytest.approx(1 / 3)


def test_wer_insertion_and_deletion():
    refs = ["one two three"]
    hyps = ["one two three four"]  # one insertion
    assert compute_wer(refs, hyps) == pytest.approx(1 / 3)

    refs2 = ["one two three four"]
    hyps2 = ["one two three"]  # one deletion
    assert compute_wer(refs2, hyps2) == pytest.approx(1 / 4)


def test_cer_perfect_match_is_zero():
    refs = ["hello"]
    hyps = ["hello"]
    assert compute_cer(refs, hyps) == 0.0


def test_cer_partial_match():
    refs = ["hello"]
    hyps = ["hallo"]  # 1 char substitution
    assert compute_cer(refs, hyps) == pytest.approx(1 / 5)


def test_wer_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        compute_wer(["one"], ["one", "two"])


def test_wer_detailed_result_has_breakdown():
    refs = ["a b c"]
    hyps = ["a x c d"]
    result = compute_wer(refs, hyps, detailed=True)
    assert result.substitutions + result.deletions + result.insertions > 0
    assert result.total_ref_units == 3
    assert len(result.per_example) == 1


def test_empty_hypothesis_gives_full_deletion_wer():
    refs = ["a b c"]
    hyps = [""]
    assert compute_wer(refs, hyps) == pytest.approx(1.0)
