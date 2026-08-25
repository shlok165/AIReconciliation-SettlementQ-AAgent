"""Fuzzy matching engine for financial reconciliation.

Generates ranked candidate match pairs for records unresolved by deterministic matching.
Calculates text similarity, amount proximity, and date window signals without making
final reconciliation decisions.
"""

from dataclasses import dataclass, field
from typing import Any, Collection, Dict, List, Optional, Set, Tuple
import pandas as pd

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None


# --- Data Structures ---

@dataclass(frozen=True)
class FuzzyCandidate:
    """Represents a potential match pair evaluated by fuzzy criteria."""
    left_id: str
    right_id: str
    match_type: str
    text_similarity: float
    amount_difference_minor: int
    date_difference_days: int
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FuzzyMatchResult:
    """Container holding ranked fuzzy candidate pairs and remaining unresolved IDs."""
    payment_bank_candidates: List[FuzzyCandidate]
    invoice_payment_candidates: List[FuzzyCandidate]
    unresolved_payment_ids: List[str]
    unresolved_bank_transaction_ids: List[str]
    unresolved_invoice_ids: List[str]


# --- Helper & Validation Logic ---

def _validate_rapidfuzz_installed() -> None:
    """Ensure rapidfuzz is available in the Python environment."""
    if fuzz is None:
        raise ImportError(
            "The 'rapidfuzz' library is required for fuzzy string matching. "
            "Please install it using: pip install rapidfuzz"
        )


def _validate_columns(df: pd.DataFrame, required_cols: List[str], dataset_name: str) -> None:
    """Validate that required normalized columns exist in the DataFrame."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required normalized columns for {dataset_name} in fuzzy matching: {missing}"
        )


def _normalize_text(value: Any) -> str:
    """Clean text values for similarity scoring, handling NaNs and non-string inputs."""
    if pd.isna(value) or value is None:
        return ""
    return str(value).strip().lower()


def _compute_text_similarity(
    left_str: str, right_str: str
) -> Tuple[float, Dict[str, float]]:
    """Compute multiple fuzzy similarity metrics and return the best score alongside breakdown."""
    left_clean = _normalize_text(left_str)
    right_clean = _normalize_text(right_str)

    if not left_clean or not right_clean:
        return 0.0, {"ratio": 0.0, "token_sort_ratio": 0.0, "token_set_ratio": 0.0}

    r_score = float(fuzz.ratio(left_clean, right_clean))
    ts_score = float(fuzz.token_sort_ratio(left_clean, right_clean))
    tset_score = float(fuzz.token_set_ratio(left_clean, right_clean))

    best_score = max(r_score, ts_score, tset_score)
    scores_breakdown = {
        "ratio": r_score,
        "token_sort_ratio": ts_score,
        "token_set_ratio": tset_score,
    }

    return best_score, scores_breakdown


# --- Candidate Generation ---

def generate_payment_bank_candidates(
    payments: pd.DataFrame,
    bank_transactions: pd.DataFrame,
    unresolved_payment_ids: Optional[Collection[str]] = None,
    unresolved_bank_transaction_ids: Optional[Collection[str]] = None,
    max_amount_difference_minor: int = 100,
    max_date_difference_days: int = 7,
    min_text_similarity: float = 0.0,
) -> List[FuzzyCandidate]:
    """Generate and rank candidate pairs between unresolved payments and bank transactions."""
    _validate_rapidfuzz_installed()

    required_payment_cols = [
        "payment_id_normalized",
        "net_settled_amount_normalized",
        "settlement_date_normalized",
    ]
    required_bank_cols = [
        "transaction_id_normalized",
        "amount_normalized",
        "date_normalized",
    ]

    _validate_columns(payments, required_payment_cols, "Payments")
    _validate_columns(bank_transactions, required_bank_cols, "Bank Transactions")

    # Filter to unresolved records if specified
    pay_df = payments.copy()
    if unresolved_payment_ids is not None:
        pay_df = pay_df[pay_df["payment_id_normalized"].isin(unresolved_payment_ids)]

    bank_df = bank_transactions.copy()
    if unresolved_bank_transaction_ids is not None:
        bank_df = bank_df[
            bank_df["transaction_id_normalized"].isin(unresolved_bank_transaction_ids)
        ]

    if pay_df.empty or bank_df.empty:
        return []

    candidates: List[FuzzyCandidate] = []

    # Iterate candidate pairs using amount and date blocking
    for _, pay in pay_df.iterrows():
        p_id = str(pay["payment_id_normalized"])
        p_amount = pay["net_settled_amount_normalized"]
        p_date = pd.to_datetime(pay["settlement_date_normalized"])

        for _, bank in bank_df.iterrows():
            b_id = str(bank["transaction_id_normalized"])
            b_amount = bank["amount_normalized"]
            b_date = pd.to_datetime(bank["date_normalized"])

            # 1. Amount Proximity Filter
            amount_diff = abs(int(p_amount) - int(b_amount))
            if amount_diff > max_amount_difference_minor:
                continue

            # 2. Date Window Filter
            if pd.isna(p_date) or pd.isna(b_date):
                continue
            date_diff = abs((p_date - b_date).days)
            if date_diff > max_date_difference_days:
                continue

            # 3. Fuzzy Text Similarity Signals
            b_ref = bank.get("reference_no_normalized", "")
            b_desc = bank.get("description_normalized", "")

            sim_p_id_vs_ref, ref_breakdown = _compute_text_similarity(p_id, b_ref)
            sim_p_id_vs_desc, desc_breakdown = _compute_text_similarity(p_id, b_desc)

            best_similarity = max(sim_p_id_vs_ref, sim_p_id_vs_desc)

            if best_similarity < min_text_similarity:
                continue

            evidence = {
                "payment_id_vs_bank_reference": sim_p_id_vs_ref,
                "payment_id_vs_bank_reference_breakdown": ref_breakdown,
                "payment_id_vs_bank_description": sim_p_id_vs_desc,
                "payment_id_vs_bank_description_breakdown": desc_breakdown,
                "best_text_signal": (
                    "payment_id_vs_bank_reference"
                    if sim_p_id_vs_ref >= sim_p_id_vs_desc
                    else "payment_id_vs_bank_description"
                ),
            }

            candidates.append(
                FuzzyCandidate(
                    left_id=p_id,
                    right_id=b_id,
                    match_type="PAYMENT_BANK_FUZZY_CANDIDATE",
                    text_similarity=best_similarity,
                    amount_difference_minor=amount_diff,
                    date_difference_days=date_diff,
                    evidence=evidence,
                )
            )

    # Rank Candidates: 1. Higher Similarity, 2. Lower Amount Diff, 3. Lower Date Diff
    candidates.sort(
        key=lambda c: (
            -c.text_similarity,
            c.amount_difference_minor,
            c.date_difference_days,
        )
    )

    return candidates


# --- Pipeline Orchestrator ---

def run_fuzzy_matching(
    invoices: pd.DataFrame,
    payments: pd.DataFrame,
    bank_transactions: pd.DataFrame,
    deterministic_result: Optional[Any] = None,
    max_amount_difference_minor: int = 100,
    max_date_difference_days: int = 7,
    min_text_similarity: float = 0.0,
    include_ambiguous_as_unresolved: bool = True,
) -> FuzzyMatchResult:
    """Execute fuzzy candidate generation on unresolved records from deterministic phase."""

    # 1. Determine unresolved record IDs
    if deterministic_result is not None:
        unresolved_payments: Set[str] = set(deterministic_result.unmatched_payment_ids)
        unresolved_bank_txns: Set[str] = set(deterministic_result.unmatched_bank_transaction_ids)
        unresolved_invoices: Set[str] = set(deterministic_result.unmatched_invoice_ids)

        # Explicitly inspect ambiguous match relationship types
        # Explicitly inspect ambiguous match relationship types
        if include_ambiguous_as_unresolved and hasattr(deterministic_result, "ambiguous_matches"):
            for amb in deterministic_result.ambiguous_matches:
                match_type = getattr(amb, "match_type", "")
                source_id = getattr(amb, "source_id", None)
                candidate_ids = getattr(amb, "candidate_ids", [])

                if "PAYMENT_BANK" in match_type:
                    if source_id:
                        unresolved_payments.add(source_id)
                    unresolved_bank_txns.update(candidate_ids)
                elif "INVOICE_PAYMENT" in match_type:
                    if source_id:
                        unresolved_invoices.add(source_id)
                    unresolved_payments.update(candidate_ids)
                else:
                    continue
    else:
        unresolved_payments = set(
            payments["payment_id_normalized"].dropna().astype(str)
        ) if "payment_id_normalized" in payments.columns else set()

        unresolved_bank_txns = set(
            bank_transactions["transaction_id_normalized"].dropna().astype(str)
        ) if "transaction_id_normalized" in bank_transactions.columns else set()

        unresolved_invoices = set(
            invoices["invoice_id_normalized"].dropna().astype(str)
        ) if "invoice_id_normalized" in invoices.columns else set()

    # 2. Generate candidates for Payment <-> Bank
    pb_candidates = generate_payment_bank_candidates(
        payments=payments,
        bank_transactions=bank_transactions,
        unresolved_payment_ids=unresolved_payments,
        unresolved_bank_transaction_ids=unresolved_bank_txns,
        max_amount_difference_minor=max_amount_difference_minor,
        max_date_difference_days=max_date_difference_days,
        min_text_similarity=min_text_similarity,
    )

    return FuzzyMatchResult(
        payment_bank_candidates=pb_candidates,
        invoice_payment_candidates=[],
        unresolved_payment_ids=sorted(list(unresolved_payments)),
        unresolved_bank_transaction_ids=sorted(list(unresolved_bank_txns)),
        unresolved_invoice_ids=sorted(list(unresolved_invoices)),
    )