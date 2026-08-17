import { useState, useRef } from 'react'
import { uploadDocument } from '../api.js'

export default function UploadView({ onIndexed }) {
  const [dragOver, setDragOver] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  async function handleFile(file) {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Only PDF files are supported.')
      return
    }
    setError(null)
    setLoading(true)
    try {
      const data = await uploadDocument(file)
      onIndexed({
        sessionId: data.session_id,
        filename: data.filename,
        pageCount: data.page_count,
        chunkCount: data.chunk_count,
      })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="upload-view">
      <div className="upload-title">
        <h1>AskMyDocs</h1>
        <p>Upload a PDF, ask it anything. Answers come straight from your document.</p>
      </div>

      <div
        className={`dropzone ${dragOver ? 'dropzone-active' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          handleFile(e.dataTransfer.files[0])
        }}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          hidden
          onChange={(e) => handleFile(e.target.files[0])}
        />
        {loading ? (
          <p>Indexing your document...</p>
        ) : (
          <>
            <p className="dropzone-label">Drop a PDF here, or click to choose one</p>
            <p className="dropzone-hint">PDF only</p>
          </>
        )}
      </div>

      {error && <p className="error-text">{error}</p>}
    </div>
  )
}