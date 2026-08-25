"""Unit tests for app/engine/scoring.py."""

import pytest
from app.engine.fuzzy import FuzzyCandidate
from app.engine.scoring import (
    ScoredCandidate,
    ScoringResult,
    calculate_amount_score,
    calculate_confidence,
    calculate_date_score,
    classify_confidence,
    score_fuzzy_candidates,
)


def create_fuzzy_candidate(
    left_id: str = "PAY_001",
    right_id: str = "BANK_001",
    text_similarity: float = 95.0,
    amount_difference_minor: int = 0,
    date_difference_days: int = 0,
    match_type: str = "PAYMENT_BANK_FUZZY",
) -> FuzzyCandidate:
    """Helper factory for building valid FuzzyCandidate instances in tests."""
    return FuzzyCandidate(
        left_id=left_id,
        right_id=right_id,
        match_type=match_type,
        text_similarity=text_similarity,
        amount_difference_minor=amount_difference_minor,
        date_difference_days=date_difference_days,
        evidence=["Fuzzy text similarity match"],
    )


# --- 1. High-Confidence Candidate Test ---

def test_high_confidence_candidate_auto_matches():
    cand = create_fuzzy_candidate(
        text_similarity=96.0,
        amount_difference_minor=0,
        date_difference_days=0,
    )
    result = score_fuzzy_candidates([cand])

    assert len(result.auto_matches) == 1
    assert len(result.review_candidates) == 0
    assert len(result.rejected_candidates) == 0

    scored = result.auto_matches[0]
    assert scored.decision == "AUTO_MATCH"
    # Weighted calculation: (96 * 0.5) + (100 * 0.3) + (100 * 0.2) = 48 + 30 + 20 = 98.0
    assert pytest.approx(scored.confidence_score, abs=1e-4) == 98.0
    assert scored.text_score == 96.0
    assert scored.amount_score == 100.0
    assert scored.date_score == 100.0


# --- 2. Medium-Confidence Candidate Test ---

def test_medium_confidence_candidate_goes_to_review():
    cand = create_fuzzy_candidate(
        text_similarity=75.0,
        amount_difference_minor=20,  # score = 80 with max 100
        date_difference_days=2,      # score ≈ 71.43 with max 7
    )
    # (75 * 0.5) + (80 * 0.3) + (71.4285 * 0.2) = 37.5 + 24.0 + 14.2857 = 75.7857
    result = score_fuzzy_candidates([cand])

    assert len(result.auto_matches) == 0
    assert len(result.review_candidates) == 1
    assert len(result.rejected_candidates) == 0

    scored = result.review_candidates[0]
    assert scored.decision == "REVIEW"
    assert 70.0 <= scored.confidence_score < 90.0


# --- 3. Low-Confidence Candidate Test ---

def test_low_confidence_candidate_is_rejected():
    cand = create_fuzzy_candidate(
        text_similarity=40.0,
        amount_difference_minor=80,  # score = 20 with max 100
        date_difference_days=6,      # score ≈ 14.28 with max 7
    )
    # (40 * 0.5) + (20 * 0.3) + (14.2857 * 0.2) = 20.0 + 6.0 + 2.857 = 28.857
    result = score_fuzzy_candidates([cand])

    assert len(result.auto_matches) == 0
    assert len(result.review_candidates) == 0
    assert len(result.rejected_candidates) == 1

    scored = result.rejected_candidates[0]
    assert scored.decision == "REJECT"
    assert scored.confidence_score < 70.0


# --- 4. Ambiguity Margin Protection Test ---

def test_ambiguity_margin_downgrades_competing_auto_matches():
    # Both candidates pass auto_match threshold (90.0), but margin (2.0) < min_margin (5.0)
    cand_a = create_fuzzy_candidate(
        left_id="PAY_001",
        right_id="BANK_A",
        text_similarity=92.0,
        amount_difference_minor=0,
        date_difference_days=0,
    )  # Score = 96.0
    cand_b = create_fuzzy_candidate(
        left_id="PAY_001",
        right_id="BANK_B",
        text_similarity=88.0,
        amount_difference_minor=0,
        date_difference_days=0,
    )  # Score = 94.0

    result = score_fuzzy_candidates(
        [cand_a, cand_b],
        auto_match_threshold=90.0,
        review_threshold=70.0,
        minimum_score_margin=5.0,
    )

    # Ensure zero auto matches remain due to ambiguity
    assert len(result.auto_matches) == 0
    assert len(result.review_candidates) == 2

    scored_a = next(c for c in result.scored_candidates if c.candidate.right_id == "BANK_A")
    scored_b = next(c for c in result.scored_candidates if c.candidate.right_id == "BANK_B")

    assert scored_a.decision == "REVIEW"
    assert scored_b.decision == "REVIEW"
    assert any("Ambiguous" in r for r in scored_a.reasons)
    assert any("Competing candidate" in r for r in scored_b.reasons)


# --- 5. Zero Amount Tolerance Test ---

def test_zero_amount_tolerance():
    assert calculate_amount_score(0, max_amount_difference_minor=0) == 100.0
    assert calculate_amount_score(1, max_amount_difference_minor=0) == 0.0
    assert calculate_amount_score(100, max_amount_difference_minor=0) == 0.0


# --- 6. Zero Date Tolerance Test ---

def test_zero_date_tolerance():
    assert calculate_date_score(0, max_date_difference_days=0) == 100.0
    assert calculate_date_score(1, max_date_difference_days=0) == 0.0
    assert calculate_date_score(7, max_date_difference_days=0) == 0.0


# --- 7. Maximum Tolerance Boundaries Test ---

def test_max_tolerance_boundary_returns_zero_and_valid_sub_scores():
    # Exact boundary difference must return 0.0
    assert calculate_amount_score(100, max_amount_difference_minor=100) == 0.0
    assert calculate_date_score(7, max_date_difference_days=7) == 0.0

    # Values exceeding max tolerance clamp to 0.0
    assert calculate_amount_score(150, max_amount_difference_minor=100) == 0.0
    assert calculate_date_score(10, max_date_difference_days=7) == 0.0

    # Just below boundary should return valid score strictly between 0 and 100
    sub_max_amount = calculate_amount_score(99, max_amount_difference_minor=100)
    sub_max_date = calculate_date_score(6, max_date_difference_days=7)

    assert 0.0 < sub_max_amount < 100.0
    assert pytest.approx(sub_max_amount, abs=1e-4) == 1.0
    assert 0.0 < sub_max_date < 100.0


# --- 8. Invalid Weights Validation Test ---

def test_invalid_weights_raises_value_error():
    cand = create_fuzzy_candidate()

    # Negative weights
    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_confidence(cand, text_weight=-0.5, amount_weight=1.0, date_weight=0.5)

    # Weights that do not sum to 1.0
    with pytest.raises(ValueError, match="Weights must sum to 1.0"):
        calculate_confidence(cand, text_weight=0.5, amount_weight=0.5, date_weight=0.5)

    with pytest.raises(ValueError, match="Weights must sum to 1.0"):
        score_fuzzy_candidates([cand], text_weight=0.4, amount_weight=0.4, date_weight=0.4)


# --- 9. Invalid Thresholds Validation Test ---

def test_invalid_thresholds_raises_value_error():
    # Direct function test
    with pytest.raises(ValueError, match="cannot be greater than"):
        classify_confidence(80.0, auto_match_threshold=70.0, review_threshold=90.0)

    # Main scoring engine function test
    cand = create_fuzzy_candidate()
    with pytest.raises(ValueError, match="cannot exceed"):
        score_fuzzy_candidates([cand], auto_match_threshold=70.0, review_threshold=90.0)


# --- 10. Empty Candidate List Test ---

def test_score_fuzzy_candidates_empty_input():
    result = score_fuzzy_candidates([])
    assert isinstance(result, ScoringResult)
    assert result.scored_candidates == []
    assert result.auto_matches == []
    assert result.review_candidates == []
    assert result.rejected_candidates == []


# --- Additional Edge Cases ---

def test_negative_differences_raise_value_error():
    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_amount_score(-10, max_amount_difference_minor=100)

    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_date_score(-1, max_date_difference_days=7)


def test_exact_threshold_boundaries():
    assert classify_confidence(90.0, auto_match_threshold=90.0, review_threshold=70.0) == "AUTO_MATCH"
    assert classify_confidence(70.0, auto_match_threshold=90.0, review_threshold=70.0) == "REVIEW"
    assert classify_confidence(69.999, auto_match_threshold=90.0, review_threshold=70.0) == "REJECT"


def test_weight_tolerance_acceptance():
    cand = create_fuzzy_candidate()
    # Weights sum to 1.00001 (within default weight_tolerance=1e-4)
    conf, _, _, _ = calculate_confidence(
        cand,
        text_weight=0.500005,
        amount_weight=0.300000,
        date_weight=0.199999,
    )
    assert isinstance(conf, float)