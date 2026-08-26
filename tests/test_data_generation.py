from pathlib import Path

from scripts.generate_data import generate_dataset


def test_generate_dataset_respects_requested_size_and_writes_outputs(tmp_path):
    result = generate_dataset(40, output_dir=tmp_path)

    assert result["size_requested"] == 40
    assert result["total_invoices"] + result["total_payments"] + result["total_bank_transactions"] > 0
    assert (tmp_path / "raw" / "invoices.csv").exists()
    assert (tmp_path / "raw" / "payments.csv").exists()
    assert (tmp_path / "raw" / "bank_transactions.csv").exists()
    assert (tmp_path / "ground_truth" / "ground_truth.csv").exists()
    assert result["total_records"] >= 40
