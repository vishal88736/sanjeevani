"""
Word Error Rate (WER) and Character Error Rate (CER) computation via
Levenshtein edit distance, plus basic per-language breakdown and
substitution/insertion/deletion confusion statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _edit_distance_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """Standard DP edit distance. Returns (substitutions, deletions, insertions, ref_length)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrack to classify operation types.
    i, j = n, m
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return subs, dels, ins, n


@dataclass
class ErrorRateResult:
    error_rate: float
    substitutions: int
    deletions: int
    insertions: int
    total_ref_units: int
    per_example: list[float] = field(default_factory=list)


def _normalize(text: str) -> str:
    return " ".join(text.strip().split())


def compute_wer(references: list[str], hypotheses: list[str], detailed: bool = False):
    """Corpus-level WER = total edit distance / total reference word count.

    Set `detailed=True` to receive an ErrorRateResult with per-example
    rates and substitution/deletion/insertion breakdowns; otherwise
    returns just the float WER for convenience in the training loop.
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must be the same length")

    total_subs = total_dels = total_ins = total_ref = 0
    per_example = []

    for ref, hyp in zip(references, hypotheses):
        ref_words = _normalize(ref).split()
        hyp_words = _normalize(hyp).split()
        subs, dels, ins, ref_len = _edit_distance_ops(ref_words, hyp_words)
        total_subs += subs
        total_dels += dels
        total_ins += ins
        total_ref += ref_len
        per_example.append((subs + dels + ins) / max(ref_len, 1))

    error_rate = (total_subs + total_dels + total_ins) / max(total_ref, 1)

    if not detailed:
        return error_rate

    return ErrorRateResult(
        error_rate=error_rate,
        substitutions=total_subs,
        deletions=total_dels,
        insertions=total_ins,
        total_ref_units=total_ref,
        per_example=per_example,
    )


def compute_cer(references: list[str], hypotheses: list[str], detailed: bool = False):
    """Corpus-level CER = total character edit distance / total reference char count."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must be the same length")

    total_subs = total_dels = total_ins = total_ref = 0
    per_example = []

    for ref, hyp in zip(references, hypotheses):
        ref_chars = list(_normalize(ref).replace(" ", ""))
        hyp_chars = list(_normalize(hyp).replace(" ", ""))
        subs, dels, ins, ref_len = _edit_distance_ops(ref_chars, hyp_chars)
        total_subs += subs
        total_dels += dels
        total_ins += ins
        total_ref += ref_len
        per_example.append((subs + dels + ins) / max(ref_len, 1))

    error_rate = (total_subs + total_dels + total_ins) / max(total_ref, 1)

    if not detailed:
        return error_rate

    return ErrorRateResult(
        error_rate=error_rate,
        substitutions=total_subs,
        deletions=total_dels,
        insertions=total_ins,
        total_ref_units=total_ref,
        per_example=per_example,
    )
