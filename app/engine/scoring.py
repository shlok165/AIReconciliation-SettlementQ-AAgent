"""Scoring engine for financial reconciliation fuzzy candidates.

Calculates multi-signal confidence scores (0.0 to 100.0) from text similarity,
amount proximity, and date window signals. Applies decision thresholds and ambiguity
margin checks to categorize candidates into AUTO_MATCH, REVIEW, or REJECT.
"""

from dataclasses import dataclass, field
from typing import Collection, Dict, List, Optional, Tuple
import math

from app.engine.fuzzy import FuzzyCandidate


# --- Data Structures ---

@dataclass(frozen=True)
class ScoredCandidate:
    """Represents a fuzzy candidate with computed confidence scores and decision."""
    candidate: FuzzyCandidate
    confidence_score: float
    text_score: float
    amount_score: float
    date_score: float
    decision: str  # AUTO_MATCH, REVIEW, REJECT
    reasons: List[str] = field(default_factory=list)


@dataclass
class ScoringResult:
    """Container holding all scored candidates and decision partitions."""
    scored_candidates: List[ScoredCandidate]
    auto_matches: List[ScoredCandidate]
    review_candidates: List[ScoredCandidate]
    rejected_candidates: List[ScoredCandidate]


# --- Sub-score Calculation Helpers ---

def calculate_amount_score(
    amount_difference_minor: int,
    max_amount_difference_minor: int,
) -> float:
    """Calculate normalized amount proximity score from 0.0 to 100.0.
    
    Raises:
        ValueError: If amount_difference_minor is negative or max threshold is negative.
    """
    if amount_difference_minor < 0:
        raise ValueError(f"amount_difference_minor cannot be negative: {amount_difference_minor}")
    if max_amount_difference_minor < 0:
        raise ValueError(f"max_amount_difference_minor cannot be negative: {max_amount_difference_minor}")

    if max_amount_difference_minor == 0:
        return 100.0 if amount_difference_minor == 0 else 0.0

    if amount_difference_minor >= max_amount_difference_minor:
        return 0.0

    score = 100.0 * (1.0 - (amount_difference_minor / max_amount_difference_minor))
    return float(max(0.0, min(100.0, score)))


def calculate_date_score(
    date_difference_days: int,
    max_date_difference_days: int,
) -> float:
    """Calculate normalized date proximity score from 0.0 to 100.0.
    
    Raises:
        ValueError: If date_difference_days is negative or max threshold is negative.
    """
    if date_difference_days < 0:
        raise ValueError(f"date_difference_days cannot be negative: {date_difference_days}")
    if max_date_difference_days < 0:
        raise ValueError(f"max_date_difference_days cannot be negative: {max_date_difference_days}")

    if max_date_difference_days == 0:
        return 100.0 if date_difference_days == 0 else 0.0

    if date_difference_days >= max_date_difference_days:
        return 0.0

    score = 100.0 * (1.0 - (date_difference_days / max_date_difference_days))
    return float(max(0.0, min(100.0, score)))


def calculate_confidence(
    candidate: FuzzyCandidate,
    *,
    text_weight: float = 0.50,
    amount_weight: float = 0.30,
    date_weight: float = 0.20,
    max_amount_difference_minor: int = 100,
    max_date_difference_days: int = 7,
    weight_tolerance: float = 1e-4,
) -> Tuple[float, float, float, float]:
    """Calculate overall weighted confidence score and individual sub-scores."""
    if text_weight < 0 or amount_weight < 0 or date_weight < 0:
        raise ValueError("Weights cannot be negative.")

    total_weight = text_weight + amount_weight + date_weight
    if not math.isclose(total_weight, 1.0, abs_tol=weight_tolerance):
        raise ValueError(
            f"Weights must sum to 1.0 within tolerance {weight_tolerance}. "
            f"Got: text={text_weight}, amount={amount_weight}, date={date_weight} (sum={total_weight})"
        )

    # Text similarity is already on a 0.0 - 100.0 scale from FuzzyCandidate
    text_score = max(0.0, min(100.0, candidate.text_similarity))

    amount_score = calculate_amount_score(
        candidate.amount_difference_minor,
        max_amount_difference_minor,
    )
    date_score = calculate_date_score(
        candidate.date_difference_days,
        max_date_difference_days,
    )

    confidence = (
        (text_score * text_weight)
        + (amount_score * amount_weight)
        + (date_score * date_weight)
    )

    return (
        float(max(0.0, min(100.0, confidence))),
        float(text_score),
        float(amount_score),
        float(date_score),
    )


def classify_confidence(
    confidence_score: float,
    auto_match_threshold: float = 90.0,
    review_threshold: float = 70.0,
) -> str:
    """Classify confidence score into AUTO_MATCH, REVIEW, or REJECT."""
    if review_threshold > auto_match_threshold:
        raise ValueError(
            f"review_threshold ({review_threshold}) cannot be greater than "
            f"auto_match_threshold ({auto_match_threshold})"
        )

    if confidence_score >= auto_match_threshold:
        return "AUTO_MATCH"
    elif confidence_score >= review_threshold:
        return "REVIEW"
    else:
        return "REJECT"


# --- Main Engine Function ---

def score_fuzzy_candidates(
    candidates: Collection[FuzzyCandidate],
    *,
    text_weight: float = 0.50,
    amount_weight: float = 0.30,
    date_weight: float = 0.20,
    max_amount_difference_minor: int = 100,
    max_date_difference_days: int = 7,
    auto_match_threshold: float = 90.0,
    review_threshold: float = 70.0,
    minimum_score_margin: float = 5.0,
) -> ScoringResult:
    """Score fuzzy candidates and categorize decisions with ambiguity protection."""
    if not candidates:
        return ScoringResult([], [], [], [])

    if review_threshold > auto_match_threshold:
        raise ValueError(
            f"review_threshold ({review_threshold}) cannot exceed auto_match_threshold ({auto_match_threshold})"
        )

    # 1. Initial Scoring Pass
    initially_scored: List[ScoredCandidate] = []
    grouped_by_left_id: Dict[str, List[ScoredCandidate]] = {}

    for cand in candidates:
        conf, t_score, a_score, d_score = calculate_confidence(
            cand,
            text_weight=text_weight,
            amount_weight=amount_weight,
            date_weight=date_weight,
            max_amount_difference_minor=max_amount_difference_minor,
            max_date_difference_days=max_date_difference_days,
        )

        initial_decision = classify_confidence(
            conf,
            auto_match_threshold=auto_match_threshold,
            review_threshold=review_threshold,
        )

        reasons: List[str] = []
        if t_score > 0:
            reasons.append(f"Text similarity score: {t_score:.1f}")
        if cand.amount_difference_minor == 0:
            reasons.append("Exact amount match (0 minor units diff)")
        else:
            reasons.append(f"Amount diff: {cand.amount_difference_minor} minor units (score: {a_score:.1f})")

        if cand.date_difference_days == 0:
            reasons.append("Exact date match (0 days diff)")
        else:
            reasons.append(f"Date diff: {cand.date_difference_days} days (score: {d_score:.1f})")

        scored_item = ScoredCandidate(
            candidate=cand,
            confidence_score=conf,
            text_score=t_score,
            amount_score=a_score,
            date_score=d_score,
            decision=initial_decision,
            reasons=reasons,
        )

        initially_scored.append(scored_item)
        grouped_by_left_id.setdefault(cand.left_id, []).append(scored_item)

    # 2. Ambiguity & Margin Analysis Pass
    final_scored: List[ScoredCandidate] = []
    auto_matches: List[ScoredCandidate] = []
    review_candidates: List[ScoredCandidate] = []
    rejected_candidates: List[ScoredCandidate] = []

    for left_id, group in grouped_by_left_id.items():
        # Sort candidates for the same left_id descending by confidence
        sorted_group = sorted(group, key=lambda x: x.confidence_score, reverse=True)

        best = sorted_group[0]
        downgrade_best_for_ambiguity = False
        ambiguity_reason = ""

        # Check score margin if multiple candidates exist for the same source record
        if len(sorted_group) > 1:
            second_best = sorted_group[1]
            margin = best.confidence_score - second_best.confidence_score

            if best.decision == "AUTO_MATCH" and margin < minimum_score_margin:
                downgrade_best_for_ambiguity = True
                ambiguity_reason = (
                    f"Ambiguous: Top candidate confidence ({best.confidence_score:.1f}) "
                    f"is only {margin:.1f} points above second-best ({second_best.confidence_score:.1f}), "
                    f"below required margin of {minimum_score_margin:.1f}."
                )

        # Build final candidates with adjusted decisions
        # Build final candidates with adjusted decisions
        for idx, item in enumerate(sorted_group):
            item_reasons = list(item.reasons)
            item_decision = item.decision

            # If the group is flagged as ambiguous, ensure NO candidate retains AUTO_MATCH
            if downgrade_best_for_ambiguity and item_decision == "AUTO_MATCH":
                item_decision = "REVIEW"
                if idx == 0:
                    item_reasons.append(ambiguity_reason)
                else:
                    item_reasons.append(
                        f"Competing candidate for ambiguous source record {left_id}; "
                        "another candidate has a score within the required margin."
                    )

            final_item = ScoredCandidate(
                candidate=item.candidate,
                confidence_score=item.confidence_score,
                text_score=item.text_score,
                amount_score=item.amount_score,
                date_score=item.date_score,
                decision=item_decision,
                reasons=item_reasons,
            )

            final_scored.append(final_item)

            if item_decision == "AUTO_MATCH":
                auto_matches.append(final_item)
            elif item_decision == "REVIEW":
                review_candidates.append(final_item)
            else:
                rejected_candidates.append(final_item)

    return ScoringResult(
        scored_candidates=final_scored,
        auto_matches=auto_matches,
        review_candidates=review_candidates,
        rejected_candidates=rejected_candidates,
    )