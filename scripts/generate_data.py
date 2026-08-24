import random
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker
import pandas as pd

# Define Project Paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_GT_DIR = PROJECT_ROOT / "data" / "ground_truth"


class IDGenerator:

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


def generate_clean_cases(fake: Faker, id_gen: IDGenerator, count: int = 30):
    """Generate exact match scenarios across all sources."""
    invoices, txns, payments, ground_truths = [], [], [], []

    for _ in range(count):
        case_id = id_gen.next_case_id()
        inv_id = id_gen.next_invoice_id()
        txn_id = id_gen.next_txn_id()
        pay_id = id_gen.next_payment_id()

        cust_ref = f"CUST-{random.randint(1000, 9999)}"
        amount = round(random.uniform(100.0, 10000.0), 2)
        base_date = fake.date_between(
            start_date="-60d", end_date="today"
        )
        date_str = base_date.strftime("%Y-%m-%d")

        desc = f"Payment for {fake.bs()}"

        invoices.append(
            {
                "invoice_id": inv_id,
                "expected_amount": f"{amount:.2f}",
                "status": "PAID",
                "invoice_date": date_str,
                "customer_ref": cust_ref,
                "description": desc,
            }
        )

        txns.append(
            {
                "transaction_id": txn_id,
                "amount": f"{amount:.2f}",
                "date": date_str,
                "description": desc,
                "reference_no": f"REF-{inv_id}",
            }
        )

        payments.append(
            {
                "payment_id": pay_id,
                "gross_amount": f"{amount:.2f}",
                "fee": "0.00",
                "net_settled_amount": f"{amount:.2f}",
                "settlement_date": date_str,
                "linked_invoice_id": inv_id,
            }
        )

        ground_truths.append(
            {
                "case_id": case_id,
                "invoice_id": inv_id,
                "transaction_id": txn_id,
                "payment_id": pay_id,
                "expected_result": "MATCH",
                "exception_reason": "",
            }
        )

    return invoices, txns, payments, ground_truths


def generate_fuzzy_cases(fake: Faker, id_gen: IDGenerator, count: int = 10):
    """Generate cases with slight date offsets and description variations."""
    invoices, txns, payments, ground_truths = [], [], [], []

    for _ in range(count):
        case_id = id_gen.next_case_id()
        inv_id = id_gen.next_invoice_id()
        txn_id = id_gen.next_txn_id()
        pay_id = id_gen.next_payment_id()

        cust_ref = f"CUST-{random.randint(1000, 9999)}"
        amount = round(random.uniform(100.0, 5000.0), 2)
        base_date = fake.date_between(start_date="-60d", end_date="-10d")

        service_name = fake.bs()
        inv_desc = f"Services payment for {service_name}"
        txn_desc = f"Payment - {service_name}"

        inv_date = base_date.strftime("%Y-%m-%d")
        pay_date = (base_date + timedelta(days=random.randint(1, 2))).strftime(
            "%Y-%m-%d"
        )
        txn_date = (base_date + timedelta(days=random.randint(2, 3))).strftime(
            "%Y-%m-%d"
        )

        invoices.append(
            {
                "invoice_id": inv_id,
                "expected_amount": f"{amount:.2f}",
                "status": "PAID",
                "invoice_date": inv_date,
                "customer_ref": cust_ref,
                "description": inv_desc,
            }
        )

        txns.append(
            {
                "transaction_id": txn_id,
                "amount": f"{amount:.2f}",
                "date": txn_date,
                "description": txn_desc,
                "reference_no": f"TRANSFER-{random.randint(10000, 99999)}",
            }
        )

        payments.append(
            {
                "payment_id": pay_id,
                "gross_amount": f"{amount:.2f}",
                "fee": "0.00",
                "net_settled_amount": f"{amount:.2f}",
                "settlement_date": pay_date,
                "linked_invoice_id": inv_id,
            }
        )

        ground_truths.append(
            {
                "case_id": case_id,
                "invoice_id": inv_id,
                "transaction_id": txn_id,
                "payment_id": pay_id,
                "expected_result": "MATCH",
                "exception_reason": "",
            }
        )

    return invoices, txns, payments, ground_truths


def generate_fee_cases(fake: Faker, id_gen: IDGenerator, count: int = 5):
    """Generate cases where payment gateway fees affect net settlement."""
    invoices, txns, payments, ground_truths = [], [], [], []

    for _ in range(count):
        case_id = id_gen.next_case_id()
        inv_id = id_gen.next_invoice_id()
        txn_id = id_gen.next_txn_id()
        pay_id = id_gen.next_payment_id()

        gross = round(random.uniform(500.0, 10000.0), 2)
        fee = round(gross * random.uniform(0.01, 0.03), 2)
        net = round(gross - fee, 2)

        base_date = fake.date_between(start_date="-60d", end_date="-5d")
        date_str = base_date.strftime("%Y-%m-%d")

        invoices.append(
            {
                "invoice_id": inv_id,
                "expected_amount": f"{gross:.2f}",
                "status": "PAID",
                "invoice_date": date_str,
                "customer_ref": f"CUST-{random.randint(1000, 9999)}",
                "description": f"Invoice for {fake.company()}",
            }
        )

        payments.append(
            {
                "payment_id": pay_id,
                "gross_amount": f"{gross:.2f}",
                "fee": f"{fee:.2f}",
                "net_settled_amount": f"{net:.2f}",
                "settlement_date": date_str,
                "linked_invoice_id": inv_id,
            }
        )

        txns.append(
            {
                "transaction_id": txn_id,
                "amount": f"{net:.2f}",
                "date": date_str,
                "description": f"Settlement payout {pay_id}",
                "reference_no": f"SETTLE-{pay_id}",
            }
        )

        ground_truths.append(
            {
                "case_id": case_id,
                "invoice_id": inv_id,
                "transaction_id": txn_id,
                "payment_id": pay_id,
                "expected_result": "MATCH",
                "exception_reason": "",
            }
        )

    return invoices, txns, payments, ground_truths


def generate_ambiguous_cases(fake: Faker, id_gen: IDGenerator, count: int = 5):
    """Generate genuine unresolved ambiguous cases.

    Two competing candidate invoices are generated with identical amounts, dates,
    customer references, and descriptions. Neither invoice is a designated "true"
    match, making resolution impossible without human intervention.
    """
    invoices, txns, payments, ground_truths = [], [], [], []

    for _ in range(count):
        case_id = id_gen.next_case_id()
        inv_id1 = id_gen.next_invoice_id()
        inv_id2 = id_gen.next_invoice_id()
        txn_id = id_gen.next_txn_id()
        pay_id = id_gen.next_payment_id()

        shared_amount = round(random.uniform(200.0, 1500.0), 2)
        base_date = fake.date_between(start_date="-40d", end_date="-10d")
        date_str = base_date.strftime("%Y-%m-%d")

        cust_ref = f"CUST-{random.randint(1000, 9999)}"
        desc = "Monthly subscription renewal"

        # Candidate Invoice 1
        invoices.append(
            {
                "invoice_id": inv_id1,
                "expected_amount": f"{shared_amount:.2f}",
                "status": "PENDING",
                "invoice_date": date_str,
                "customer_ref": cust_ref,
                "description": desc,
            }
        )

        # Candidate Invoice 2 (Equally plausible candidate)
        invoices.append(
            {
                "invoice_id": inv_id2,
                "expected_amount": f"{shared_amount:.2f}",
                "status": "PENDING",
                "invoice_date": date_str,
                "customer_ref": cust_ref,
                "description": desc,
            }
        )

        txns.append(
            {
                "transaction_id": txn_id,
                "amount": f"{shared_amount:.2f}",
                "date": date_str,
                "description": "Subscription payment",
                "reference_no": "PAYMENT-RECURRING",
            }
        )

        payments.append(
            {
                "payment_id": pay_id,
                "gross_amount": f"{shared_amount:.2f}",
                "fee": "0.00",
                "net_settled_amount": f"{shared_amount:.2f}",
                "settlement_date": date_str,
                "linked_invoice_id": "",
            }
        )

        ground_truths.append(
            {
                "case_id": case_id,
                "invoice_id": "",
                "transaction_id": txn_id,
                "payment_id": pay_id,
                "expected_result": "EXCEPTION",
                "exception_reason": "AMBIGUOUS_MATCH",
            }
        )

    return invoices, txns, payments, ground_truths


def generate_exception_cases(fake: Faker, id_gen: IDGenerator, count: int = 5):
    """Generate orphan records and severe mismatches."""
    invoices, txns, payments, ground_truths = [], [], [], []

    reasons = [
        "ORPHAN_INVOICE",
        "ORPHAN_BANK_TRANSACTION",
        "ORPHAN_PAYMENT",
        "DATE_MISMATCH",
        "NO_CANDIDATE",
    ]

    for i in range(count):
        case_id = id_gen.next_case_id()
        reason = reasons[i % len(reasons)]

        amount = round(random.uniform(100.0, 3000.0), 2)
        base_date = fake.date_between(start_date="-60d", end_date="-20d")
        date_str = base_date.strftime("%Y-%m-%d")

        inv_id, txn_id, pay_id = "", "", ""

        if reason == "ORPHAN_INVOICE":
            inv_id = id_gen.next_invoice_id()
            invoices.append(
                {
                    "invoice_id": inv_id,
                    "expected_amount": f"{amount:.2f}",
                    "status": "PENDING",
                    "invoice_date": date_str,
                    "customer_ref": f"CUST-{random.randint(1000, 9999)}",
                    "description": "Unpaid invoice orphan",
                }
            )

        elif reason == "ORPHAN_BANK_TRANSACTION":
            txn_id = id_gen.next_txn_id()
            txns.append(
                {
                    "transaction_id": txn_id,
                    "amount": f"{amount:.2f}",
                    "date": date_str,
                    "description": "Unknown bank deposit",
                    "reference_no": "UNKNOWN-DEP",
                }
            )

        elif reason == "ORPHAN_PAYMENT":
            pay_id = id_gen.next_payment_id()
            payments.append(
                {
                    "payment_id": pay_id,
                    "gross_amount": f"{amount:.2f}",
                    "fee": "0.00",
                    "net_settled_amount": f"{amount:.2f}",
                    "settlement_date": date_str,
                    "linked_invoice_id": "",
                }
            )

        elif reason == "DATE_MISMATCH":
            inv_id = id_gen.next_invoice_id()
            txn_id = id_gen.next_txn_id()
            pay_id = id_gen.next_payment_id()

            far_date = (base_date + timedelta(days=90)).strftime("%Y-%m-%d")

            invoices.append(
                {
                    "invoice_id": inv_id,
                    "expected_amount": f"{amount:.2f}",
                    "status": "PAID",
                    "invoice_date": date_str,
                    "customer_ref": f"CUST-{random.randint(1000, 9999)}",
                    "description": "Services subject to severe date lag",
                }
            )

            # Bank transaction uses a generic reference to prevent direct ID bypass
            txns.append(
                {
                    "transaction_id": txn_id,
                    "amount": f"{amount:.2f}",
                    "date": far_date,
                    "description": "Services payment",
                    "reference_no": f"WIRE-{random.randint(10000, 99999)}",
                }
            )

            payments.append(
                {
                    "payment_id": pay_id,
                    "gross_amount": f"{amount:.2f}",
                    "fee": "0.00",
                    "net_settled_amount": f"{amount:.2f}",
                    "settlement_date": far_date,
                    "linked_invoice_id": inv_id,
                }
            )

        elif reason == "NO_CANDIDATE":
            inv_id = id_gen.next_invoice_id()
            invoices.append(
                {
                    "invoice_id": inv_id,
                    "expected_amount": f"{amount:.2f}",
                    "status": "FAILED",
                    "invoice_date": date_str,
                    "customer_ref": f"CUST-{random.randint(1000, 9999)}",
                    "description": "Cancelled order",
                }
            )

        ground_truths.append(
            {
                "case_id": case_id,
                "invoice_id": inv_id,
                "transaction_id": txn_id,
                "payment_id": pay_id,
                "expected_result": "EXCEPTION",
                "exception_reason": reason,
            }
        )

    return invoices, txns, payments, ground_truths


def generate_amount_mismatch_cases(
    fake: Faker, id_gen: IDGenerator, count: int = 5
):
    """Generate cases where invoice amount differs from bank settlement."""
    invoices, txns, payments, ground_truths = [], [], [], []

    for _ in range(count):
        case_id = id_gen.next_case_id()
        inv_id = id_gen.next_invoice_id()
        txn_id = id_gen.next_txn_id()
        pay_id = id_gen.next_payment_id()

        expected_amount = round(random.uniform(1000.0, 5000.0), 2)
        paid_amount = round(expected_amount - random.uniform(50.0, 200.0), 2)

        base_date = fake.date_between(start_date="-40d", end_date="-5d")
        date_str = base_date.strftime("%Y-%m-%d")

        invoices.append(
            {
                "invoice_id": inv_id,
                "expected_amount": f"{expected_amount:.2f}",
                "status": "PARTIAL",
                "invoice_date": date_str,
                "customer_ref": f"CUST-{random.randint(1000, 9999)}",
                "description": f"Partial payment invoice {fake.company()}",
            }
        )

        txns.append(
            {
                "transaction_id": txn_id,
                "amount": f"{paid_amount:.2f}",
                "date": date_str,
                "description": f"Underpayment for {inv_id}",
                "reference_no": f"REF-{inv_id}",
            }
        )

        payments.append(
            {
                "payment_id": pay_id,
                "gross_amount": f"{paid_amount:.2f}",
                "fee": "0.00",
                "net_settled_amount": f"{paid_amount:.2f}",
                "settlement_date": date_str,
                "linked_invoice_id": inv_id,
            }
        )

        ground_truths.append(
            {
                "case_id": case_id,
                "invoice_id": inv_id,
                "transaction_id": txn_id,
                "payment_id": pay_id,
                "expected_result": "EXCEPTION",
                "exception_reason": "AMOUNT_MISMATCH",
            }
        )

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
        generate_clean_cases(fake, id_gen, count=30),
        generate_fuzzy_cases(fake, id_gen, count=10),
        generate_fee_cases(fake, id_gen, count=5),
        generate_ambiguous_cases(fake, id_gen, count=5),
        generate_exception_cases(fake, id_gen, count=5),
        generate_amount_mismatch_cases(fake, id_gen, count=5),
    ]

    for inv, txn, pay, gt in generators:
        inv_all.extend(inv)
        txn_all.extend(txn)
        pay_all.extend(pay)
        gt_all.extend(gt)

    count_inv, count_txn, count_pay, count_gt = save_data(
        inv_all, txn_all, pay_all, gt_all
    )

    print("Dataset generated successfully.\n")
    print(f"Invoices: {count_inv}")
    print(f"Bank transactions: {count_txn}")
    print(f"Payments: {count_pay}")
    print(f"Ground truth cases: {count_gt}\n")
    print(f"Output directory: {PROJECT_ROOT / 'data'}")


if __name__ == "__main__":
    main()