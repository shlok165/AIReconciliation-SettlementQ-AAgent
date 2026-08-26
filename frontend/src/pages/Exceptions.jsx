import ExceptionTable from '../components/ExceptionTable'

export default function Exceptions({ exceptions }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Exception log</h2>
        <p>Records that need validation or follow-up</p>
      </div>
      <ExceptionTable rows={exceptions} />
    </div>
  )
}
