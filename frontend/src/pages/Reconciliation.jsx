import MetricCard from '../components/MetricCard'
import DataTable from '../components/DataTable'

const invoiceColumns = [
  { key: 'invoice_id', label: 'Invoice ID' },
  { key: 'customer_id', label: 'Customer' },
  { key: 'invoice_date', label: 'Date' },
  { key: 'expected_amount', label: 'Amount' },
  { key: 'status', label: 'Status' },
  { key: 'description', label: 'Description' },
]

const paymentColumns = [
  { key: 'payment_id', label: 'Payment ID' },
  { key: 'linked_invoice_id', label: 'Linked Invoice' },
  { key: 'settlement_date', label: 'Date' },
  { key: 'gross_amount', label: 'Gross' },
  { key: 'fee', label: 'Fee' },
  { key: 'net_settled_amount', label: 'Net' },
  { key: 'description', label: 'Description' },
]

const bankColumns = [
  { key: 'transaction_id', label: 'Transaction ID' },
  { key: 'date', label: 'Date' },
  { key: 'amount', label: 'Amount' },
  { key: 'description', label: 'Description' },
  { key: 'reference_no', label: 'Reference' },
]

export default function Reconciliation({ dataset }) {
  if (!dataset) {
    return <div className="loading"><span className="spinner" /> Loading dataset…</div>
  }

  return (
    <>
      <div className="metrics">
        <MetricCard label="Invoices" value={dataset.invoice_count} note="invoice records" tone="blue" />
        <MetricCard label="Payments" value={dataset.payment_count} note="payment records" tone="green" />
        <MetricCard label="Bank transactions" value={dataset.bank_transaction_count} note="bank transaction records" tone="orange" />
      </div>
      <DataTable title="Invoices" subtitle="All invoice records in the current dataset" rows={dataset.invoices} columns={invoiceColumns} />
      <DataTable title="Payments" subtitle="All payment records including fees and net settlement" rows={dataset.payments} columns={paymentColumns} />
      <DataTable title="Bank transactions" subtitle="All bank transaction records" rows={dataset.bank_transactions} columns={bankColumns} />
    </>
  )
}
