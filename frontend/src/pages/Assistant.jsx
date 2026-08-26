import ChatBox from '../components/ChatBox'

export default function Assistant({ messages, question, setQuestion, working, requestStatus, dataset, onSubmit }) {
  return (
    <ChatBox
      messages={messages}
      question={question}
      setQuestion={setQuestion}
      working={working}
      requestStatus={requestStatus}
      dataset={dataset}
      onSubmit={onSubmit}
    />
  )
}
