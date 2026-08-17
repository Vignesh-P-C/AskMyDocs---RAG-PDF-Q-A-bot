import { useState, useRef, useEffect } from 'react'
import { askQuestion } from '../api.js'

export default function ReadingView({ session, onReset }) {
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)
  const [error, setError] = useState(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function handleAsk() {
    const q = question.trim()
    if (!q || asking) return

    setMessages((prev) => [...prev, { role: 'question', content: q }])
    setQuestion('')
    setAsking(true)
    setError(null)

    try {
      const data = await askQuestion(session.sessionId, q)
      setMessages((prev) => [...prev, { role: 'answer', content: data.answer }])
    } catch (e) {
      setError(e.message)
    } finally {
      setAsking(false)
    }
  }

  return (
    <div className="reading-view">
      <aside className="doc-panel">
        <div className="brand">
          <h1>AskMyDocs</h1>
          <p>Talk to your PDFs</p>
        </div>
        <div className="doc-info">
          <p className="doc-filename">{session.filename}</p>
          <p className="doc-meta">{session.pageCount} pages · {session.chunkCount} chunks</p>
        </div>
        <div className="doc-status">
          <span className="status-label">Status</span>
          <span className="status-value">Indexed, ready</span>
        </div>
        <button className="reset-btn" onClick={onReset}>Start over</button>
      </aside>

      <main className="conversation">
        <div className="spine" />
        <div className="thread">
          {messages.length === 0 && (
            <p className="thread-empty">Ask something about this document to get started.</p>
          )}
          {messages.map((m, i) =>
            m.role === 'question' ? (
              <div key={i} className="note note-question">
                <span className="note-label">Asked</span>
                <p className="note-text">{m.content}</p>
              </div>
            ) : (
              <div key={i} className="note note-answer">
                <span className="note-label">Note</span>
                <p className="note-text">{m.content}</p>
              </div>
            )
          )}
          {asking && (
            <div className="note note-answer">
              <span className="note-label">Note</span>
              <p className="note-text note-loading">Thinking...</p>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {error && <p className="error-text">{error}</p>}

        <div className="ask-row">
          <input
            type="text"
            placeholder="Ask a question about the document..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleAsk()}
            disabled={asking}
          />
          <button onClick={handleAsk} disabled={asking || !question.trim()}>Ask</button>
        </div>
      </main>
    </div>
  )
}