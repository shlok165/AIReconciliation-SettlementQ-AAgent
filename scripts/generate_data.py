"""Extreme realistic synthetic data generator for multi-source financial reconciliation benchmark.

Generates complex multi-source financial records across Invoices, Payments, and Bank Transactions,
incorporating real-world enterprise ledger challenges:
1. Clean exact matches (baseline)
2. Unstructured reference memos (empty linked_invoice_id, embedded references)
3. OCR noise, character confusions, abbreviations, and customer name typos
4. Payment gateway interchange fees and net settlement deductions
5. Multi-day banking cutoff delays and weekend settlement shifts
6. Partial payment installments (multiple payments settling a single invoice)
7. AI tie-breaker semantic ambiguity (competing identical candidate invoices)
8. Genuine exceptions (orphans, extreme date lag, severe amount mismatches)
"""

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

from faker import Faker
import pandas as pd

# Define Project Paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_GT_DIR = PROJECT_ROOT / "data" / "ground_truth"


class IDGenerator:
    """Sequential identifier generator for deterministic test case labeling."""

    def __init__(self):
        self.invoice_seq = 1
        self.txn_seq = 1
        self.payment_seq = 1
        self.case_seq = 1

    def next_invoice_id(self) -> str:
        res = f"INV-{self.invoice_seq:04d}"
        self.invoice_seq += 1
        return res

    def next_txn_id(self) -> str:
        res = f"TXN-{self.txn_seq:04d}"
        self.txn_seq += 1
        return res

    def next_payment_id(self) -> str:
        res = f"PAY-{self.payment_seq:04d}"
        self.payment_seq += 1
        return res

    def next_case_id(self) -> str:
        res = f"CASE-{self.case_seq:04d}"
        self.case_seq += 1
        return res


def init_environment(seed: int = 42):
    """Set random seeds and ensure output directories exist."""
    random.seed(seed)
    Faker.seed(seed)

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_GT_DIR.mkdir(parents=True, exist_ok=True)


# --- Noise & Entity Formatting Helpers ---

def inject_ocr_and_typos(text: str, corruption_rate: float = 0.25) -> str:
    """Inject realistic OCR character substitutions and typos into a text string."""
    if not text or random.random() > corruption_rate:
        return text

    subs = {
        "O": "0", "0": "O",
        "I": "1", "1": "I", "l": "1",
        "S": "5", "5": "S",
        "B": "8", "8": "B",
        "Z": "2", "2": "Z",
    }

    chars = list(text)
    # Apply 1-2 character transformations
    for _ in range(random.randint(1, 2)):
        idx = random.randint(0, len(chars) - 1)
        ch = chars[idx]
        if ch in subs:
            chars[idx] = subs[ch]
        elif ch.isalpha() and random.random() < 0.3:
            # Random double letter or dropped letter
            if random.random() < 0.5:
                chars[idx] = ch + ch
            elif len(chars) > 3:
                chars.pop(idx)
                break

    # Abbreviations
    res = "".join(chars)
    replacements = [
        ("Corporation", "Corp"),
        ("Company", "Co"),
        ("Technologies", "Tech"),
        ("International", "Intl"),
        ("Logistics", "Log"),
        ("Solutions", "Sol"),
        ("Services", "Serv"),
        ("Incorporated", "Inc"),
        ("Limited", "Ltd"),
    ]
    for old, new in replacements:
        if old in res and random.random() < 0.5:
            res = res.replace(old, new)

    return res


def format_unstructured_memo(inv_id: str, cust_name: str, base_desc: str) -> Tuple[str, str]:
    """Generate realistic unstructured bank memos and payment descriptions."""
    clean_cust = cust_name.replace(",", "").replace(".", "").upper()
    raw_num = inv_id.replace("INV-", "")

    pay_templates = [
        f"Payment for {inv_id} - {cust_name}",
        f"{clean_cust} / INV#{raw_num} settlement",
        f"Direct debit {inv_id} {base_desc}",
        f"ACH PMT {clean_cust} REF:{inv_id}",
        f"Online transfer {base_desc} (Invoice {inv_id})",
        f"{inv_id} / {clean_cust}",
    ]

    bank_templates = [
        f"ACH CR {random.randint(100000, 999999)} {clean_cust} INV#{raw_num}",
        f"WIRE TFR: {inv_id} / {clean_cust}",
        f"SQ *{clean_cust[:12]} REF:{inv_id}",
        f"SETTLE-{inv_id}-{random.randint(100, 999)}",
        f"EDI PYMT {clean_cust} {inv_id}",
        f"DEP {clean_cust} - {inv_id}",
    ]

    return random.choice(pay_templates), random.choice(bank_templates)


# --- Category Generators ---

def generate_clean_cases(fake: Faker, id_gen: IDGenerator, count: int = 25):
    """Category 1: Clean exact matches (baseline)."""
    invoices, txns, payments, ground_truths = [], [], [], []

    for _ in range(count):
        case_id = id_gen.next_case_id()
        inv_id = id_gen.next_invoice_id()
        txn_id = id_gen.next_txn_id()
        pay_id = id_gen.next_payment_id()

        company = fake.company()
        cust_ref = f"CUST-{random.randint(1000, 9999)}"
        amount = round(random.uniform(200.0, 8000.0), 2)
        base_date = fake.date_between(start_date="-60d", end_date="-5d")
        date_str = base_date.strftime("%Y-%m-%d")

        desc = f"Standard invoice for {company}"

        invoices.append({
            "invoice_id": inv_id,
            "expected_amount": f"{amount:.2f}",
            "status": "PAID",
            "invoice_date": date_str,
            "customer_ref": cust_ref,
            "description": desc,
        })

        payments.append({
            "payment_id": pay_id,
            "gross_amount": f"{amount:.2f}",
            "fee": "0.00",
            "net_settled_amount": f"{amount:.2f}",
            "settlement_date": date_str,
            "linked_invoice_id": inv_id,
        })

        txns.append({
            "transaction_id": txn_id,
            "amount": f"{amount:.2f}",
            "date": date_str,
            "description": desc,
            "reference_no": f"REF-{inv_id}",
        })

        ground_truths.append({
            "case_id": case_id,
            "invoice_id": inv_id,
            "transaction_id": txn_id,
            "payment_id": pay_id,
            "expected_result": "MATCH",
            "exception_reason": "",
            "category": "CLEAN_EXACT",
        })

    return invoices, txns, payments, ground_truths


def generate_unstructured_memo_cases(fake: Faker, id_gen: IDGenerator, count: int = 20):
    """Category 2: Unstructured text references with missing linked_invoice_id foreign keys."""
    invoices, txns, payments, ground_truths = [], [], [], []

    for _ in range(count):
        case_id = id_gen.next_case_id()
        inv_id = id_gen.next_invoice_id()
        txn_id = id_gen.next_txn_id()
        pay_id = id_gen.next_payment_id()

        company = fake.company()
        cust_ref = f"CUST-{random.randint(1000, 9999)}"
        amount = round(random.uniform(150.0, 9500.0), 2)
        base_date = fake.date_between(start_date="-50d", end_date="-5d")
        date_str = base_date.strftime("%Y-%m-%d")

        service_desc = fake.bs()
        inv_desc = f"{service_desc} for {company}"
        pay_memo, bank_memo = format_unstructured_memo(inv_id, company, service_desc)

        invoices.append({
            "invoice_id": inv_id,
            "expected_amount": f"{amount:.2f}",
            "status": "PAID",
            "invoice_date": date_str,
            "customer_ref": cust_ref,
            "description": inv_desc,
        })

        # Notice: linked_invoice_id is intentionally omitted (empty string)
        payments.append({
            "payment_id": pay_id,
            "gross_amount": f"{amount:.2f}",
            "fee": "0.00",
            "net_settled_amount": f"{amount:.2f}",
            "settlement_date": date_str,
            "linked_invoice_id": "",
        })

        txns.append({
            "transaction_id": txn_id,
            "amount": f"{amount:.2f}",
            "date": date_str,
            "description": pay_memo,
            "reference_no": bank_memo,
        })

        ground_truths.append({
            "case_id": case_id,
            "invoice_id": inv_id,
            "transaction_id": txn_id,
            "payment_id": pay_id,
            "expected_result": "MATCH",
            "exception_reason": "",
            "category": "UNSTRUCTURED_MEMO",
        })

    return invoices, txns, payments, ground_truths


def generate_ocr_typo_cases(fake: Faker, id_gen: IDGenerator, count: int = 15):
    """Category 3: OCR errors, character substitutions, abbreviations, and name typos."""
    invoices, txns, payments, ground_truths = [], [], [], []

    for _ in range(count):
        case_id = id_gen.next_case_id()
        inv_id = id_gen.next_invoice_id()
        txn_id = id_gen.next_txn_id()
        pay_id = id_gen.next_payment_id()

        company = fake.company()
        noisy_company = inject_ocr_and_typos(company, corruption_rate=0.9)
        noisy_inv_ref = inject_ocr_and_typos(inv_id, corruption_rate=0.7)

        amount = round(random.uniform(300.0, 7500.0), 2)
        base_date = fake.date_between(start_date="-45d", end_date="-4d")
        date_str = base_date.strftime("%Y-%m-%d")

        service_desc = fake.catch_phrase()
        inv_desc = f"{service_desc} - {company}"

        invoices.append({
            "invoice_id": inv_id,
            "expected_amount": f"{amount:.2f}",
            "status": "PAID",
            "invoice_date": date_str,
            "customer_ref": f"CUST-{random.randint(1000, 9999)}",
            "description": inv_desc,
        })

        payments.append({
            "payment_id": pay_id,
            "gross_amount": f"{amount:.2f}",
            "fee": "0.00",
            "net_settled_amount": f"{amount:.2f}",
            "settlement_date": date_str,
            "linked_invoice_id": "",
        })

        txns.append({
            "transaction_id": txn_id,
            "amount": f"{amount:.2f}",
            "date": date_str,
            "description": f"PAYMENT {noisy_company} {service_desc[:15]}",
            "reference_no": f"REF-{noisy_inv_ref}",
        })

        ground_truths.append({
            "case_id": case_id,
            "invoice_id": inv_id,
            "transaction_id": txn_id,
            "payment_id": pay_id,
            "expected_result": "MATCH",
            "exception_reason": "",
            "category": "OCR_TYPO_NOISE",
        })

    return invoices, txns, payments, ground_truths


def generate_fee_and_fx_cases(fake: Faker, id_gen: IDGenerator, count: int = 10):
    """Category 4: Payment processor gateway fees and net settlement deductions."""
    invoices, txns, payments, ground_truths = [], [], [], []

    for _ in range(count):
        case_id = id_gen.next_case_id()
        inv_id = id_gen.next_invoice_id()
        txn_id = id_gen.next_txn_id()
        pay_id = id_gen.next_payment_id()

        gross = round(random.uniform(500.0, 12000.0), 2)
        # 2.9% + $0.30 standard gateway fee, or 1.5% interchange
        if random.random() < 0.5:
            fee = round(gross * 0.029 + 0.30, 2)
        else:
            fee = round(gross * 0.015, 2)
        net = round(gross - fee, 2)

        base_date = fake.date_between(start_date="-50d", end_date="-5d")
        date_str = base_date.strftime("%Y-%m-%d")
        company = fake.company()

        invoices.append({
            "invoice_id": inv_id,
            "expected_amount": f"{gross:.2f}",
            "status": "PAID",
            "invoice_date": date_str,
            "customer_ref": f"CUST-{random.randint(1000, 9999)}",
            "description": f"Invoice for {company}",
        })

        payments.append({
            "payment_id": pay_id,
            "gross_amount": f"{gross:.2f}",
            "fee": f"{fee:.2f}",
            "net_settled_amount": f"{net:.2f}",
            "settlement_date": date_str,
            "linked_invoice_id": inv_id,
        })

        txns.append({
            "transaction_id": txn_id,
            "amount": f"{net:.2f}",
            "date": date_str,
            "description": f"Merchant payout {company} (Net after fees)",
            "reference_no": f"STRIPE-SETTLE-{pay_id}",
        })

        ground_truths.append({
            "case_id": case_id,
            "invoice_id": inv_id,
            "transaction_id": txn_id,
            "payment_id": pay_id,
            "expected_result": "MATCH",
            "exception_reason": "",
            "category": "GATEWAY_FEES",
        })

    return invoices, txns, payments, ground_truths


def generate_banking_delay_cases(fake: Faker, id_gen: IDGenerator, count: int = 10):
    """Category 5: Multi-day banking cutoff delays and weekend settlement shifts."""
    invoices, txns, payments, ground_truths = [], [], [], []

    for _ in range(count):
        case_id = id_gen.next_case_id()
        inv_id = id_gen.next_invoice_id()
        txn_id = id_gen.next_txn_id()
        pay_id = id_gen.next_payment_id()

        amount = round(random.uniform(400.0, 6000.0), 2)
        inv_date = fake.date_between(start_date="-55d", end_date="-10d")
        
        # Payment 1-3 days after invoice
        pay_date = inv_date + timedelta(days=random.randint(1, 3))
        # Bank clearance 2-4 days after payment (total shift 3-6 days)
        bank_date = pay_date + timedelta(days=random.randint(2, 4))

        company = fake.company()

        invoices.append({
            "invoice_id": inv_id,
            "expected_amount": f"{amount:.2f}",
            "status": "PAID",
            "invoice_date": inv_date.strftime("%Y-%m-%d"),
            "customer_ref": f"CUST-{random.randint(1000, 9999)}",
            "description": f"Enterprise services for {company}",
        })

        payments.append({
            "payment_id": pay_id,
            "gross_amount": f"{amount:.2f}",
            "fee": "0.00",
            "net_settled_amount": f"{amount:.2f}",
            "settlement_date": pay_date.strftime("%Y-%m-%d"),
            "linked_invoice_id": inv_id,
        })

        txns.append({
            "transaction_id": txn_id,
            "amount": f"{amount:.2f}",
            "date": bank_date.strftime("%Y-%m-%d"),
            "description": f"ACH Credit - {company}",
            "reference_no": f"ACH-{random.randint(100000, 999999)}",
        })

        ground_truths.append({
            "case_id": case_id,
            "invoice_id": inv_id,
            "transaction_id": txn_id,
            "payment_id": pay_id,
            "expected_result": "MATCH",
            "exception_reason": "",
            "category": "BANKING_DELAY",
        })

    return invoices, txns, payments, ground_truths


def generate_partial_payment_cases(fake: Faker, id_gen: IDGenerator, count: int = 5):
    """Category 6: Single invoice settled by multiple partial payments."""
    invoices, txns, payments, ground_truths = [], [], [], []

    for _ in range(count):
        case_id = id_gen.next_case_id()
        inv_id = id_gen.next_invoice_id()

        total_amount = round(random.uniform(3000.0, 10000.0), 2)
        split_ratio = random.choice([0.5, 0.6, 0.7])
        p1_amt = round(total_amount * split_ratio, 2)
        p2_amt = round(total_amount - p1_amt, 2)

        base_date = fake.date_between(start_date="-40d", end_date="-15d")
        pay1_date = base_date + timedelta(days=1)
        pay2_date = base_date + timedelta(days=5)

        company = fake.company()

        invoices.append({
            "invoice_id": inv_id,
            "expected_amount": f"{total_amount:.2f}",
            "status": "PAID",
            "invoice_date": base_date.strftime("%Y-%m-%d"),
            "customer_ref": f"CUST-{random.randint(1000, 9999)}",
            "description": f"Multi-installment project milestone for {company}",
        })

        # Payment 1
        p1_id = id_gen.next_payment_id()
        t1_id = id_gen.next_txn_id()
        payments.append({
            "payment_id": p1_id,
            "gross_amount": f"{p1_amt:.2f}",
            "fee": "0.00",
            "net_settled_amount": f"{p1_amt:.2f}",
            "settlement_date": pay1_date.strftime("%Y-%m-%d"),
            "linked_invoice_id": inv_id,
        })
        txns.append({
            "transaction_id": t1_id,
            "amount": f"{p1_amt:.2f}",
            "date": pay1_date.strftime("%Y-%m-%d"),
            "description": f"Installment 1/2 for {inv_id}",
            "reference_no": f"INST-1-{inv_id}",
        })

        # Payment 2
        p2_id = id_gen.next_payment_id()
        t2_id = id_gen.next_txn_id()
        payments.append({
            "payment_id": p2_id,
            "gross_amount": f"{p2_amt:.2f}",
            "fee": "0.00",
            "net_settled_amount": f"{p2_amt:.2f}",
            "settlement_date": pay2_date.strftime("%Y-%m-%d"),
            "linked_invoice_id": inv_id,
        })
        txns.append({
            "transaction_id": t2_id,
            "amount": f"{p2_amt:.2f}",
            "date": pay2_date.strftime("%Y-%m-%d"),
            "description": f"Installment 2/2 for {inv_id}",
            "reference_no": f"INST-2-{inv_id}",
        })

        # We record both sub-relationships in ground truth for precise pairwise tracking
        ground_truths.append({
            "case_id": case_id,
            "invoice_id": inv_id,
            "transaction_id": t1_id,
            "payment_id": p1_id,
            "expected_result": "MATCH",
            "exception_reason": "",
            "category": "PARTIAL_PAYMENTS",
        })
        ground_truths.append({
            "case_id": case_id,
            "invoice_id": inv_id,
            "transaction_id": t2_id,
            "payment_id": p2_id,
            "expected_result": "MATCH",
            "exception_reason": "",
            "category": "PARTIAL_PAYMENTS",
        })

    return invoices, txns, payments, ground_truths


def generate_ai_tie_breaker_cases(fake: Faker, id_gen: IDGenerator, count: int = 5):
    """Category 7: Semantic ambiguity where competing invoices have identical amounts and dates."""
    invoices, txns, payments, ground_truths = [], [], [], []

    services_pairs = [
        ("Cloud Infrastructure Hosting (AWS Q3)", "Cybersecurity Penetration Testing & Audit"),
        ("Custom Software Engineering Services", "Database High-Availability Optimization"),
        ("Financial ERP Integration Consulting", "Corporate Payroll System Migration"),
        ("Monthly Dedicated Server Maintenance", "Disaster Recovery Standby Site Hosting"),
        ("Executive Leadership Coaching Program", "Annual Corporate Compliance Training"),
    ]

    for i in range(count):
        case_id = id_gen.next_case_id()
        inv1_id = id_gen.next_invoice_id()
        inv2_id = id_gen.next_invoice_id()
        txn_id = id_gen.next_txn_id()
        pay_id = id_gen.next_payment_id()

        shared_amount = round(random.uniform(1500.0, 8500.0), 2)
        base_date = fake.date_between(start_date="-35d", end_date="-10d")
        date_str = base_date.strftime("%Y-%m-%d")

        company = fake.company()
        cust_ref = f"CUST-{random.randint(1000, 9999)}"

        s1, s2 = services_pairs[i % len(services_pairs)]

        # Target matched invoice is s1
        invoices.append({
            "invoice_id": inv1_id,
            "expected_amount": f"{shared_amount:.2f}",
            "status": "PAID",
            "invoice_date": date_str,
            "customer_ref": cust_ref,
            "description": s1,
        })

        # Competing invoice is s2
        invoices.append({
            "invoice_id": inv2_id,
            "expected_amount": f"{shared_amount:.2f}",
            "status": "OPEN",
            "invoice_date": date_str,
            "customer_ref": cust_ref,
            "description": s2,
        })

        # Payment references the semantic text of s1 without exact ID link
        payments.append({
            "payment_id": pay_id,
            "gross_amount": f"{shared_amount:.2f}",
            "fee": "0.00",
            "net_settled_amount": f"{shared_amount:.2f}",
            "settlement_date": date_str,
            "linked_invoice_id": "",
        })

        txns.append({
            "transaction_id": txn_id,
            "amount": f"{shared_amount:.2f}",
            "date": date_str,
            "description": f"Payment for {s1} - {company}",
            "reference_no": f"TX-{random.randint(10000, 99999)}",
        })

        ground_truths.append({
            "case_id": case_id,
            "invoice_id": inv1_id,
            "transaction_id": txn_id,
            "payment_id": pay_id,
            "expected_result": "MATCH",
            "exception_reason": "AI_TIE_BREAKER_REQUIRED",
            "category": "AI_SEMANTIC_AMBIGUITY",
        })

    return invoices, txns, payments, ground_truths


def generate_complex_exception_cases(fake: Faker, id_gen: IDGenerator, count: int = 10):
    """Category 8: Genuine exceptions, orphans, severe date lags, and amount shortfalls."""
    invoices, txns, payments, ground_truths = [], [], [], []

    reasons = [
        "ORPHAN_INVOICE",
        "ORPHAN_BANK_TRANSACTION",
        "ORPHAN_PAYMENT",
        "SEVERE_DATE_LAG",
        "AMOUNT_SHORTFALL",
        "ORPHAN_INVOICE",
        "ORPHAN_BANK_TRANSACTION",
        "ORPHAN_PAYMENT",
        "SEVERE_DATE_LAG",
        "AMOUNT_SHORTFALL",
    ]

    for i in range(count):
        case_id = id_gen.next_case_id()
        reason = reasons[i % len(reasons)]

        amount = round(random.uniform(500.0, 6000.0), 2)
        base_date = fake.date_between(start_date="-60d", end_date="-20d")
        date_str = base_date.strftime("%Y-%m-%d")
        company = fake.company()

        inv_id, txn_id, pay_id = "", "", ""

        if reason == "ORPHAN_INVOICE":
            inv_id = id_gen.next_invoice_id()
            invoices.append({
                "invoice_id": inv_id,
                "expected_amount": f"{amount:.2f}",
                "status": "UNPAID",
                "invoice_date": date_str,
                "customer_ref": f"CUST-{random.randint(1000, 9999)}",
                "description": f"Unpaid abandoned order {company}",
            })

        elif reason == "ORPHAN_BANK_TRANSACTION":
            txn_id = id_gen.next_txn_id()
            txns.append({
                "transaction_id": txn_id,
                "amount": f"{amount:.2f}",
                "date": date_str,
                "description": f"Unidentified direct credit deposit {company}",
                "reference_no": f"UNKNOWN-DEP-{random.randint(1000, 9999)}",
            })

        elif reason == "ORPHAN_PAYMENT":
            pay_id = id_gen.next_payment_id()
            payments.append({
                "payment_id": pay_id,
                "gross_amount": f"{amount:.2f}",
                "fee": "0.00",
                "net_settled_amount": f"{amount:.2f}",
                "settlement_date": date_str,
                "linked_invoice_id": "",
            })

        elif reason == "SEVERE_DATE_LAG":
            inv_id = id_gen.next_invoice_id()
            txn_id = id_gen.next_txn_id()
            pay_id = id_gen.next_payment_id()
            lagged_date = (base_date + timedelta(days=95)).strftime("%Y-%m-%d")

            invoices.append({
                "invoice_id": inv_id,
                "expected_amount": f"{amount:.2f}",
                "status": "OVERDUE",
                "invoice_date": date_str,
                "customer_ref": f"CUST-{random.randint(1000, 9999)}",
                "description": f"Contract services with 90+ day delay for {company}",
            })
            payments.append({
                "payment_id": pay_id,
                "gross_amount": f"{amount:.2f}",
                "fee": "0.00",
                "net_settled_amount": f"{amount:.2f}",
                "settlement_date": lagged_date,
                "linked_invoice_id": inv_id,
            })
            txns.append({
                "transaction_id": txn_id,
                "amount": f"{amount:.2f}",
                "date": lagged_date,
                "description": f"Severely delayed settlement {company}",
                "reference_no": f"DELAYED-PAY-{random.randint(100, 999)}",
            })

        elif reason == "AMOUNT_SHORTFALL":
            inv_id = id_gen.next_invoice_id()
            txn_id = id_gen.next_txn_id()
            pay_id = id_gen.next_payment_id()
            underpayment = round(amount - random.uniform(100.0, 400.0), 2)

            invoices.append({
                "invoice_id": inv_id,
                "expected_amount": f"{amount:.2f}",
                "status": "SHORT_PAID",
                "invoice_date": date_str,
                "customer_ref": f"CUST-{random.randint(1000, 9999)}",
                "description": f"Disputed invoice underpayment {company}",
            })
            payments.append({
                "payment_id": pay_id,
                "gross_amount": f"{underpayment:.2f}",
                "fee": "0.00",
                "net_settled_amount": f"{underpayment:.2f}",
                "settlement_date": date_str,
                "linked_invoice_id": inv_id,
            })
            txns.append({
                "transaction_id": txn_id,
                "amount": f"{underpayment:.2f}",
                "date": date_str,
                "description": f"Partial unauthorized deduction for {inv_id}",
                "reference_no": f"SHORT-PMT-{inv_id}",
            })

        ground_truths.append({
            "case_id": case_id,
            "invoice_id": inv_id,
            "transaction_id": txn_id,
            "payment_id": pay_id,
            "expected_result": "EXCEPTION",
            "exception_reason": reason,
            "category": "GENUINE_EXCEPTION",
        })

    return invoices, txns, payments, ground_truths


def save_data(invoices, txns, payments, ground_truths):
    """Write generated records to CSV files."""
    df_inv = pd.DataFrame(invoices)
    df_txn = pd.DataFrame(txns)
    df_pay = pd.DataFrame(payments)
    df_gt = pd.DataFrame(ground_truths)

    inv_path = DATA_RAW_DIR / "invoices.csv"
    txn_path = DATA_RAW_DIR / "bank_transactions.csv"
    pay_path = DATA_RAW_DIR / "payments.csv"
    gt_path = DATA_GT_DIR / "ground_truth.csv"

    df_inv.to_csv(inv_path, index=False)
    df_txn.to_csv(txn_path, index=False)
    df_pay.to_csv(pay_path, index=False)
    df_gt.to_csv(gt_path, index=False)

    return len(df_inv), len(df_txn), len(df_pay), len(df_gt)


def main():
    init_environment(seed=42)
    fake = Faker()
    id_gen = IDGenerator()

    inv_all, txn_all, pay_all, gt_all = [], [], [], []

    generators = [
        generate_clean_cases(fake, id_gen, count=25),
        generate_unstructured_memo_cases(fake, id_gen, count=20),
        generate_ocr_typo_cases(fake, id_gen, count=15),
        generate_fee_and_fx_cases(fake, id_gen, count=10),
        generate_banking_delay_cases(fake, id_gen, count=10),
        generate_partial_payment_cases(fake, id_gen, count=5),
        generate_ai_tie_breaker_cases(fake, id_gen, count=5),
        generate_complex_exception_cases(fake, id_gen, count=10),
    ]

    for inv, txn, pay, gt in generators:
        inv_all.extend(inv)
        txn_all.extend(txn)
        pay_all.extend(pay)
        gt_all.extend(gt)

    count_inv, count_txn, count_pay, count_gt = save_data(
        inv_all, txn_all, pay_all, gt_all
    )

    print("=== EXTREME FINANCIAL RECONCILIATION DATASET GENERATED ===")
    print(f"Total Underlying Cases : {count_gt}")
    print(f"Total Invoices         : {count_inv}")
    print(f"Total Payments         : {count_pay}")
    print(f"Total Bank Transactions: {count_txn}")
    print(f"Output Directory       : {PROJECT_ROOT / 'data'}")


if __name__ == "__main__":
    main()