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


def generate_invoice_payment_candidates(
    invoices: pd.DataFrame,
    payments: pd.DataFrame,
    bank_transactions: Optional[pd.DataFrame] = None,
    deterministic_result: Optional[Any] = None,
    unresolved_invoice_ids: Optional[Collection[str]] = None,
    unresolved_payment_ids: Optional[Collection[str]] = None,
    max_amount_difference_minor: int = 100,
    max_date_difference_days: int = 7,
    min_text_similarity: float = 0.0,
) -> List[FuzzyCandidate]:
    """Generate and rank candidate pairs between unresolved invoices and payments."""
    _validate_rapidfuzz_installed()

    required_inv_cols = [
        "invoice_id_normalized",
        "expected_amount_normalized",
        "invoice_date_normalized",
    ]
    required_pay_cols = [
        "payment_id_normalized",
        "gross_amount_normalized",
        "settlement_date_normalized",
    ]

    _validate_columns(invoices, required_inv_cols, "Invoices")
    _validate_columns(payments, required_pay_cols, "Payments")

    inv_df = invoices.copy()
    if unresolved_invoice_ids is not None:
        inv_df = inv_df[inv_df["invoice_id_normalized"].isin(unresolved_invoice_ids)]

    pay_df = payments.copy()
    if unresolved_payment_ids is not None:
        pay_df = pay_df[pay_df["payment_id_normalized"].isin(unresolved_payment_ids)]

    if inv_df.empty or pay_df.empty:
        return []

    # Map payment IDs to bank transactions if deterministic matches are present
    pay_to_bank: Dict[str, str] = {}
    if deterministic_result and hasattr(deterministic_result, "payment_bank_matches"):
        for m in deterministic_result.payment_bank_matches:
            pay_to_bank[m.left_id] = m.right_id

    bank_lookup: Dict[str, pd.Series] = {}
    if bank_transactions is not None and "transaction_id_normalized" in bank_transactions.columns:
        bank_lookup = {
            str(r["transaction_id_normalized"]): r for _, r in bank_transactions.iterrows()
        }

    candidates: List[FuzzyCandidate] = []

    for _, inv in inv_df.iterrows():
        inv_id = str(inv["invoice_id_normalized"])
        inv_amt = inv["expected_amount_normalized"]
        inv_date = pd.to_datetime(inv["invoice_date_normalized"])
        inv_desc = str(inv.get("description_normalized", "")).lower()
        inv_id_clean = inv_id.lower().replace("-", "")

        for _, pay in pay_df.iterrows():
            p_id = str(pay["payment_id_normalized"])
            p_amt = pay["gross_amount_normalized"]
            if pd.isna(p_amt):
                p_amt = pay.get("net_settled_amount_normalized")
            p_date = pd.to_datetime(pay["settlement_date_normalized"])

            if pd.isna(inv_amt) or pd.isna(p_amt):
                continue

            # 1. Amount proximity
            amt_diff = abs(int(inv_amt) - int(p_amt))
            if amt_diff > max_amount_difference_minor:
                continue

            # 2. Date window
            if pd.isna(inv_date) or pd.isna(p_date):
                continue
            date_diff = abs((inv_date - p_date).days)
            if date_diff > max_date_difference_days:
                continue

            # 3. Multi-source Text Similarity (comparing invoice against bank memo or payment references)
            bank_id = pay_to_bank.get(p_id)
            b_ref, b_desc = "", ""
            if bank_id and bank_id in bank_lookup:
                b_ref = str(bank_lookup[bank_id].get("reference_no_normalized", "")).lower()
                b_desc = str(bank_lookup[bank_id].get("description_normalized", "")).lower()

            sim_desc_vs_desc = float(fuzz.token_set_ratio(inv_desc, b_desc)) if b_desc and inv_desc else 0.0
            sim_desc_vs_ref = float(fuzz.token_set_ratio(inv_desc, b_ref)) if b_ref and inv_desc else 0.0
            sim_id_vs_ref = float(fuzz.token_set_ratio(inv_id.lower(), b_ref)) if b_ref else 0.0
            sim_id_vs_desc = float(fuzz.token_set_ratio(inv_id.lower(), b_desc)) if b_desc else 0.0

            # Substring exact check (e.g. "INV-0026" or "0026" in bank reference)
            if (b_ref and (inv_id.lower() in b_ref or inv_id_clean in b_ref.replace("-", ""))) or \
               (b_desc and (inv_id.lower() in b_desc or inv_id_clean in b_desc.replace("-", ""))):
                sim_id_vs_ref = max(sim_id_vs_ref, 100.0)

            best_sim = max(sim_desc_vs_desc, sim_desc_vs_ref, sim_id_vs_ref, sim_id_vs_desc)

            if best_sim < min_text_similarity:
                continue

            evidence = {
                "invoice_description_vs_bank_description": sim_desc_vs_desc,
                "invoice_description_vs_bank_reference": sim_desc_vs_ref,
                "invoice_id_vs_bank_reference": sim_id_vs_ref,
                "best_similarity": best_sim,
                "associated_bank_id": bank_id,
            }

            candidates.append(
                FuzzyCandidate(
                    left_id=p_id,
                    right_id=inv_id,
                    match_type="INVOICE_PAYMENT_FUZZY_CANDIDATE",
                    text_similarity=best_sim,
                    amount_difference_minor=amt_diff,
                    date_difference_days=date_diff,
                    evidence=evidence,
                )
            )

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

    # 3. Generate candidates for Invoice <-> Payment
    ip_candidates = generate_invoice_payment_candidates(
        invoices=invoices,
        payments=payments,
        bank_transactions=bank_transactions,
        deterministic_result=deterministic_result,
        unresolved_invoice_ids=unresolved_invoices,
        unresolved_payment_ids=None,
        max_amount_difference_minor=max_amount_difference_minor,
        max_date_difference_days=max_date_difference_days,
        min_text_similarity=min_text_similarity,
    )

    return FuzzyMatchResult(
        payment_bank_candidates=pb_candidates,
        invoice_payment_candidates=ip_candidates,
        unresolved_payment_ids=sorted(list(unresolved_payments)),
        unresolved_bank_transaction_ids=sorted(list(unresolved_bank_txns)),
        unresolved_invoice_ids=sorted(list(unresolved_invoices)),
    )