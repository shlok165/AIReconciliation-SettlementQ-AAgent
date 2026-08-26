import { useEffect, useState } from 'react'
import { CircleAlert, LoaderCircle, RefreshCw } from 'lucide-react'
import {
  askQuestion,
  generateDataset,
  generateReport,
  getDataset,
  getExceptions,
  getMetrics,
  getUnresolved,
  runReconciliation,
} from './services/api'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import Reconciliation from './pages/Reconciliation'
import Exceptions from './pages/Exceptions'
import Assistant from './pages/Assistant'
import About from './pages/About'
import './App.css'

const VIEW_TITLES = {
  overview: 'Settlement overview',
  dataset: 'Generated dataset',
  exceptions: 'Exception review',
  assistant: 'Settlement Q&A',
  about: 'About the generator',
}

export default function App() {
  const [view, setView] = useState('overview')
  const [metrics, setMetrics] = useState(null)
  const [exceptions, setExceptions] = useState([])
  const [loading, setLoading] = useState(true)
  const [working, setWorking] = useState(false)
  const [error, setError] = useState('')
  const [question, setQuestion] = useState('')
  const [requestStatus, setRequestStatus] = useState('')
  const [datasetSize, setDatasetSize] = useState(100)
  const [datasetStatus, setDatasetStatus] = useState('')
  const [unresolved, setUnresolved] = useState(null)
  const [dataset, setDataset] = useState(null)
  const [messages, setMessages] = useState([
    { role: 'assistant', text: 'Ask about a payment, invoice, bank transaction, gateway fee, exception, or reconciliation metric.' },
  ])

  const refresh = async () => {
    setLoading(true)
    setError('')
    try {
      const [m, e] = await Promise.all([getMetrics(), getExceptions()])
      setMetrics(m)
      setExceptions(e)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Could not connect to FastAPI. Start it on port 8000 and try again.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
    getUnresolved().then(setUnresolved).catch(() => {})
    getDataset().then(setDataset).catch(() => {})
  }, [])

  const submit = async (event) => {
    event.preventDefault()
    const text = question.trim()
    if (!text || working) return

    setMessages((items) => [...items, { role: 'user', text }])
    setQuestion('')
    setWorking(true)
    setRequestStatus('Sending request to the Settlement Q&A agent…')

    try {
      const response = await askQuestion(text)
      setMessages((items) => [...items, { role: 'assistant', text: response.answer, trace: response.tool_trace }])
      const toolCount = response.tool_trace?.length || 0
      setRequestStatus(`Request completed${toolCount ? ` with ${toolCount} tool call${toolCount === 1 ? '' : 's'}` : ''}.`)
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || 'The assistant could not complete that request.'
      setMessages((items) => [...items, { role: 'assistant', text: detail, failed: true }])
      setRequestStatus('Request failed. See the assistant response for details.')
    } finally {
      setWorking(false)
    }
  }

  const generateData = async () => {
    setWorking(true)
    setError('')
    setDatasetStatus('Generating dataset, running reconciliation, and exporting report…')
    try {
      const result = await generateDataset(Number(datasetSize))
      setDatasetStatus(`Generated ${result.total_records} records. Running reconciliation and LLM evaluation…`)
      await runReconciliation()
      await generateReport()
      setDatasetStatus(`Done. ${result.total_records} records generated, reconciled, and evaluated.`)
      await refresh()
      getUnresolved().then(setUnresolved).catch(() => {})
      getDataset().then(setDataset).catch(() => {})
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Could not generate the dataset.')
      setDatasetStatus('')
    } finally {
      setWorking(false)
    }
  }

  return (
    <main className="shell">
      <Sidebar view={view} setView={setView} exceptionCount={exceptions.length} />

      <section className="content">
        <header>
          <div>
            <p className="eyebrow">AI RECONCILIATION ENGINE</p>
            <h1>{VIEW_TITLES[view]}</h1>
            <p className="subtitle">Current local dataset · deterministic-first workflow</p>
          </div>
          <button className="refresh" onClick={refresh} title="Refresh data">
            <RefreshCw size={18} className={loading ? 'spin' : ''} />
          </button>
        </header>

        {error && <div className="error"><CircleAlert size={18} />{error}</div>}

        {loading ? (
          <div className="loading"><span className="spinner" /> Loading current reconciliation results…</div>
        ) : (
          <>
            {view === 'overview' && (
              <Dashboard
                metrics={metrics}
                unresolved={unresolved}
                datasetStatus={datasetStatus}
                datasetSize={datasetSize}
                setDatasetSize={setDatasetSize}
                working={working}
                onGenerate={generateData}
              />
            )}
            {view === 'dataset' && <Reconciliation dataset={dataset} />}
            {view === 'exceptions' && <Exceptions exceptions={exceptions} />}
            {view === 'assistant' && (
              <Assistant
                messages={messages}
                question={question}
                setQuestion={setQuestion}
                working={working}
                requestStatus={requestStatus}
                dataset={dataset}
                onSubmit={submit}
              />
            )}
            {view === 'about' && <About />}
          </>
        )}
      </section>
    </main>
  )
}
