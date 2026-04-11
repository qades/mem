"""
Evaluation metrics for context management strategies.
"""

from typing import List, Dict, Any, Optional
import math
from difflib import SequenceMatcher


def calculate_metrics(
    generated_responses: List[str], reference_responses: List[str]
) -> Dict[str, float]:
    """Calculate comprehensive metrics for response quality."""
    if not generated_responses or not reference_responses:
        return {
            "bleu": 0.0,
            "rouge_l": 0.0,
            "exact_match": 0.0,
            "semantic_similarity": 0.0,
        }

    metrics = {}

    # BLEU score (simplified)
    metrics["bleu"] = _calculate_bleu(generated_responses, reference_responses)

    # ROUGE-L score
    metrics["rouge_l"] = _calculate_rouge_l(generated_responses, reference_responses)

    # Exact match
    metrics["exact_match"] = _calculate_exact_match(
        generated_responses, reference_responses
    )

    # Semantic similarity (placeholder)
    metrics["semantic_similarity"] = _calculate_semantic_similarity(
        generated_responses, reference_responses
    )

    return metrics


def _calculate_bleu(hypotheses: List[str], references: List[str]) -> float:
    """Calculate BLEU score (simplified unigram implementation)."""
    if len(hypotheses) != len(references):
        return 0.0

    total_precision = 0.0

    for hyp, ref in zip(hypotheses, references):
        hyp_words = hyp.lower().split()
        ref_words = ref.lower().split()

        if not hyp_words:
            continue

        # Count matching unigrams
        matches = sum(1 for w in hyp_words if w in ref_words)
        precision = matches / len(hyp_words)
        total_precision += precision

    return total_precision / len(hypotheses) if hypotheses else 0.0


def _calculate_rouge_l(hypotheses: List[str], references: List[str]) -> float:
    """Calculate ROUGE-L score (simplified LCS implementation)."""
    if len(hypotheses) != len(references):
        return 0.0

    total_score = 0.0

    for hyp, ref in zip(hypotheses, references):
        lcs_length = _longest_common_subsequence(hyp.lower(), ref.lower())

        if not hyp or not ref:
            continue

        precision = lcs_length / len(hyp)
        recall = lcs_length / len(ref)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            total_score += f1

    return total_score / len(hypotheses) if hypotheses else 0.0


def _longest_common_subsequence(s1: str, s2: str) -> int:
    """Calculate LCS length."""
    m, n = len(s1), len(s2)

    if m == 0 or n == 0:
        return 0

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def _calculate_exact_match(hypotheses: List[str], references: List[str]) -> float:
    """Calculate exact match percentage."""
    if len(hypotheses) != len(references):
        return 0.0

    matches = sum(1 for h, r in zip(hypotheses, references) if h.strip() == r.strip())
    return matches / len(hypotheses) if hypotheses else 0.0


def _calculate_semantic_similarity(
    hypotheses: List[str], references: List[str]
) -> float:
    """Calculate semantic similarity (placeholder)."""
    if len(hypotheses) != len(references):
        return 0.0

    total_similarity = 0.0

    for hyp, ref in zip(hypotheses, references):
        similarity = SequenceMatcher(None, hyp.lower(), ref.lower()).ratio()
        total_similarity += similarity

    return total_similarity / len(hypotheses) if hypotheses else 0.0


def calculate_efficiency_metrics(
    context_sizes: List[int], response_times: List[float]
) -> Dict[str, float]:
    """Calculate efficiency metrics."""
    if not context_sizes or not response_times:
        return {}

    return {
        "avg_context_size": sum(context_sizes) / len(context_sizes),
        "max_context_size": max(context_sizes),
        "min_context_size": min(context_sizes),
        "avg_response_time_ms": sum(response_times) / len(response_times),
        "total_tokens": sum(context_sizes),
    }
