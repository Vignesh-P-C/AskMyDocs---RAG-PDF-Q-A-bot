import { useState } from 'react'
import UploadView from './components/UploadView.jsx'
import ReadingView from './components/ReadingView.jsx'

export default function App() {
  const [session, setSession] = useState(null)

  return (
    <div className="app-shell">
      {session ? (
        <ReadingView session={session} onReset={() => setSession(null)} />
      ) : (
        <UploadView onIndexed={setSession} />
      )}
    </div>
  )
}