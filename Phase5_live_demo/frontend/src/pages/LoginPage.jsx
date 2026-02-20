import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

const CAT_DISPLAY = {
  sports: 'Sports', health: 'Health', technology: 'Technology',
  politics: 'Politics', finance: 'Finance', entertainment: 'Entertainment',
  travel: 'Travel', science: 'Science', foodanddrink: 'Food & Drink',
  lifestyle: 'Lifestyle', autos: 'Autos', weather: 'Weather',
}

function today() {
  return new Date().toLocaleDateString('en-US', {
    weekday: 'long', year: 'numeric', month: 'long', day: 'numeric',
  })
}

export default function LoginPage() {
  const navigate = useNavigate()
  const [users, setUsers] = useState([])
  const [selectedUser, setSelectedUser] = useState(null)
  const [customUserId, setCustomUserId] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activePanel, setActivePanel] = useState(null)   // null | 'signin' | 'new'

  // Effective user = typed ID (if non-empty) or the selected profile
  const effectiveUser = customUserId.trim() || selectedUser

  useEffect(() => {
    fetch('/api/users')
      .then(r => r.json())
      .then(d => setUsers(d.users || []))
      .catch(() => setUsers([]))
  }, [])

  function togglePanel(panel) {
    setActivePanel(prev => prev === panel ? null : panel)
    setError(null)
  }

  async function handleLogin() {
    if (!effectiveUser) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: effectiveUser }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Login failed (${res.status})`)
      }
      const data = await res.json()
      localStorage.setItem('sessionId', data.session_id)
      localStorage.setItem('displayName', data.display_name || effectiveUser)
      if (data.is_new_user) {
        // Unknown user ID — send them through the quiz first
        localStorage.removeItem('quizCompleted')
        localStorage.removeItem('initialRecs')
        navigate('/quiz')
      } else {
        localStorage.setItem('quizCompleted', 'true')
        localStorage.setItem('initialRecs', JSON.stringify(data))
        localStorage.setItem('quizStyle', 'balanced')
        navigate('/feed')
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function handleNewReader() {
    localStorage.removeItem('quizCompleted')
    localStorage.removeItem('initialRecs')
    localStorage.removeItem('sessionId')
    localStorage.removeItem('displayName')
    navigate('/quiz')
  }

  return (
    <div className="min-h-screen bg-paper flex flex-col">

      {/* ── Masthead ── */}
      <header className="border-b-4 border-ink pt-6 pb-4 px-6 text-center">
        <div className="flex items-center justify-center gap-3 mb-1">
          <div className="h-px flex-1 bg-ink max-w-24" />
          <span className="text-xs font-bold uppercase tracking-[0.35em] text-ink-light">
            Est. 2025 · Vol. I
          </span>
          <div className="h-px flex-1 bg-ink max-w-24" />
        </div>
        <h1 className="font-headline text-5xl md:text-6xl font-bold text-ink tracking-tight leading-none">
          The NewsLens
        </h1>
        <div className="flex items-center justify-center gap-3 mt-1">
          <div className="h-px flex-1 bg-ink max-w-24" />
          <span className="text-xs uppercase tracking-[0.3em] text-ink-light">
            Diversity-Aware Personalised News
          </span>
          <div className="h-px flex-1 bg-ink max-w-24" />
        </div>
        <p className="text-xs text-ink-light mt-2">{today()}</p>
      </header>

      <main className="flex-1 max-w-3xl mx-auto w-full px-4 py-10">

        {/* ── Two-panel entry ── */}
        <div className="grid md:grid-cols-[1fr_auto_1fr] gap-0 items-start">

          {/* ── Left panel: Sign In ── */}
          <div className="border border-rule bg-white flex flex-col overflow-hidden">
            {/* Clickable header */}
            <button
              className={`w-full text-left px-6 py-5 flex items-center justify-between transition-colors
                ${activePanel === 'signin' ? 'bg-masthead' : 'bg-masthead/80 hover:bg-masthead'}`}
              onClick={() => togglePanel('signin')}
            >
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.3em] text-paper/60 mb-0.5">
                  Returning Reader
                </p>
                <h2 className="font-headline text-2xl font-bold text-paper leading-tight">
                  Sign In
                </h2>
              </div>
              <span className={`text-paper/70 text-lg transition-transform duration-300 ${
                activePanel === 'signin' ? 'rotate-180' : ''
              }`}>
                ▾
              </span>
            </button>

            {/* Expandable body */}
            <div className={`transition-all duration-300 ease-in-out overflow-hidden ${
              activePanel === 'signin' ? 'max-h-[800px] opacity-100' : 'max-h-0 opacity-0'
            }`}>
              <div className="px-6 py-5 flex flex-col">

                {/* User ID input — first */}
                <p className="text-xs text-ink-light mb-2 uppercase tracking-wide font-medium">
                  Enter your user ID:
                </p>
                <input
                  type="text"
                  value={customUserId}
                  onChange={e => { setCustomUserId(e.target.value); setSelectedUser(null) }}
                  placeholder="e.g. U12345"
                  className="w-full border border-rule bg-paper px-3 py-2 text-sm text-ink
                             placeholder:text-ink-light/50 focus:outline-none focus:border-masthead mb-5"
                />

                {/* Suggested profiles */}
                <div className="border-t border-rule pt-4 mb-5">
                  <p className="text-xs text-ink-light mb-3 uppercase tracking-wide font-medium">
                    Or choose a suggested profile:
                  </p>
                  <div className="space-y-2">
                    {users.length === 0 && (
                      <p className="text-sm text-ink-light/50 animate-pulse py-2 text-center">
                        Loading reader profiles…
                      </p>
                    )}
                    {users.map(u => (
                      <button
                        key={u.user_id}
                        onClick={() => { setSelectedUser(u.user_id); setCustomUserId('') }}
                        className={`w-full text-left border px-4 py-3 transition-all ${
                          selectedUser === u.user_id
                            ? 'border-masthead bg-masthead/5'
                            : 'border-rule bg-paper hover:border-ink-light'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <span className={`mt-0.5 w-3.5 h-3.5 flex-shrink-0 rounded-full border-2 transition-colors ${
                            selectedUser === u.user_id
                              ? 'border-masthead bg-masthead'
                              : 'border-rule'
                          }`} />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between gap-2">
                              <span className="text-sm font-semibold text-ink truncate">
                                {u.display_name}
                              </span>
                              <span className="text-xs text-ink-light flex-shrink-0">
                                {u.history_count} read
                              </span>
                            </div>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {u.top_categories.slice(0, 3).map(c => (
                                <span key={c} className="text-xs px-1.5 py-0.5 bg-paper border border-rule text-ink-light uppercase tracking-wide">
                                  {CAT_DISPLAY[c] ?? c}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                {error && (
                  <p className="text-xs text-red-600 mb-3 border-l-2 border-red-400 pl-2">{error}</p>
                )}

                <button
                  onClick={handleLogin}
                  disabled={!effectiveUser || loading}
                  className="btn-primary w-full justify-center py-3 text-sm font-semibold"
                >
                  {loading ? 'Loading your edition…' : 'Continue to My Feed →'}
                </button>
              </div>
            </div>
          </div>

          {/* ── OR divider ── */}
          <div className="hidden md:flex flex-col items-center justify-center px-6 gap-3">
            <div className="flex-1 w-px bg-rule" />
            <span className="text-xs font-bold text-ink-light uppercase tracking-widest bg-paper px-2 py-1">
              or
            </span>
            <div className="flex-1 w-px bg-rule" />
          </div>
          <div className="md:hidden thin-rule my-4" />

          {/* ── Right panel: New Reader ── */}
          <div className="border border-rule bg-white flex flex-col overflow-hidden">
            {/* Clickable header */}
            <button
              className={`w-full text-left px-6 py-5 flex items-center justify-between transition-colors
                ${activePanel === 'new'
                  ? 'bg-paper border-b border-rule'
                  : 'bg-paper hover:bg-paper/80'}`}
              onClick={() => togglePanel('new')}
            >
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.3em] text-ink-light mb-0.5">
                  First Visit
                </p>
                <h2 className="font-headline text-2xl font-bold text-ink leading-tight">
                  Create Profile
                </h2>
              </div>
              <span className={`text-ink-light text-lg transition-transform duration-300 ${
                activePanel === 'new' ? 'rotate-180' : ''
              }`}>
                ▾
              </span>
            </button>

            {/* Expandable body */}
            <div className={`transition-all duration-300 ease-in-out overflow-hidden ${
              activePanel === 'new' ? 'max-h-[520px] opacity-100' : 'max-h-0 opacity-0'
            }`}>
              <div className="px-6 py-5 flex flex-col">
                <p className="text-sm text-ink-light mb-5">
                  Answer a short quiz to set your interests. Our AI will build a personalised front page tailored to you.
                </p>

                {/* Steps */}
                <div className="space-y-4 mb-6">
                  {[
                    { step: '01', label: 'Choose your topics of interest', sub: 'Sports, Tech, Politics, and more' },
                    { step: '02', label: 'Set your reading style', sub: 'Accurate, Balanced, or Exploratory' },
                    { step: '03', label: 'Get your personalised front page', sub: 'Curated by our diversity-aware AI' },
                  ].map(item => (
                    <div key={item.step} className="flex items-start gap-4">
                      <span className="text-2xl font-bold text-rule font-headline leading-none flex-shrink-0 mt-0.5">
                        {item.step}
                      </span>
                      <div>
                        <p className="text-sm font-semibold text-ink">{item.label}</p>
                        <p className="text-xs text-ink-light mt-0.5">{item.sub}</p>
                      </div>
                    </div>
                  ))}
                </div>

                <button
                  onClick={handleNewReader}
                  className="btn-ghost w-full justify-center py-3 text-sm font-semibold border-ink text-ink hover:bg-ink hover:text-paper"
                >
                  Start the Quiz →
                </button>
              </div>
            </div>
          </div>

        </div>
      </main>

      {/* ── Footer ── */}
      <footer className="border-t border-rule py-3 text-center">
        <p className="text-xs text-rule uppercase tracking-widest">
          Capstone Project · Diversity-Aware Recommender System
        </p>
      </footer>
    </div>
  )
}
