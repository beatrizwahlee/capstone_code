import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

const TOPICS = [
  { id: 'sports',        label: 'Sports',       emoji: '⚽', desc: 'NFL, NBA, soccer, baseball, tennis, golf, Olympics & more' },
  { id: 'news',          label: 'News',         emoji: '📰', desc: 'Politics, world affairs, science, technology & breaking news' },
  { id: 'health',        label: 'Health',       emoji: '🏥', desc: 'Medicine, fitness, nutrition, mental health & wellness' },
  { id: 'finance',       label: 'Finance',      emoji: '📈', desc: 'Markets, investing, personal finance, crypto & real estate' },
  { id: 'entertainment', label: 'Entertain.',   emoji: '🎬', desc: 'Celebrity news, awards, TV, pop culture & viral trends' },
  { id: 'travel',        label: 'Travel',       emoji: '✈️', desc: 'Destinations, tips, hotels, airlines & adventure guides' },
  { id: 'foodanddrink',  label: 'Food & Drink', emoji: '🍳', desc: 'Recipes, restaurants, nutrition, wine & cooking techniques' },
  { id: 'lifestyle',     label: 'Lifestyle',    emoji: '🌿', desc: 'Fashion, home decor, relationships, beauty & wellness' },
  { id: 'autos',         label: 'Autos',        emoji: '🚗', desc: 'Car reviews, EVs, racing, buying guides & auto news' },
  { id: 'weather',       label: 'Weather',      emoji: '⛅', desc: 'Forecasts, storms, climate, seasonal alerts & extreme events' },
  { id: 'music',         label: 'Music',        emoji: '🎵', desc: 'Charts, album reviews, artists, concerts & music industry' },
  { id: 'movies',        label: 'Movies',       emoji: '🎥', desc: 'Box office, reviews, film festivals, trailers & streaming picks' },
]

const STYLES = [
  {
    id: 'accurate',
    label: 'Accurate',
    desc: 'Stick to my topics — pure relevance ranking, no surprises.',
    note: 'Baseline model only',
  },
  {
    id: 'balanced',
    label: 'Balanced',
    desc: 'Mix relevance with discovery — a bit of everything.',
    note: 'Composite reranker',
  },
  {
    id: 'explore',
    label: 'Explore',
    desc: 'Surprise me — venture well outside my usual categories.',
    note: 'Composite, max diversity',
  },
]

async function ensureSession() {
  const existing = localStorage.getItem('sessionId')
  const res = await fetch('/api/session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: existing }),
  })
  if (!res.ok) throw new Error(`Session error ${res.status}`)
  const data = await res.json()
  localStorage.setItem('sessionId', data.session_id)
  return data.session_id
}

async function postQuiz(sessionId, topics, style) {
  const res = await fetch('/api/quiz', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, preferences: { topics, style } }),
  })
  if (!res.ok) throw new Error(`Quiz error ${res.status}`)
  return res.json()
}

export default function QuizPage() {
  const navigate = useNavigate()
  const [step, setStep] = useState(1)
  const [selectedTopics, setSelectedTopics] = useState([])
  const [selectedStyle, setSelectedStyle] = useState(null)
  const [error, setError] = useState(null)

  function toggleTopic(id) {
    setSelectedTopics(prev =>
      prev.includes(id) ? prev.filter(t => t !== id) : [...prev, id]
    )
  }

  async function handleFinish() {
    setStep(3)
    setError(null)
    try {
      const sessionId = await ensureSession()
      const data = await postQuiz(sessionId, selectedTopics, selectedStyle)
      // Persist all auth state
      localStorage.setItem('quizCompleted', 'true')
      localStorage.setItem('quizStyle', selectedStyle)        // ← was missing
      localStorage.setItem('displayName', 'Guest Reader')
      localStorage.setItem('initialRecs', JSON.stringify(data))
      await new Promise(r => setTimeout(r, 1200))
      navigate('/feed')
    } catch (e) {
      setError(e.message)
      setStep(2)
    }
  }

  function today() {
    return new Date().toLocaleDateString('en-US', {
      weekday: 'long', year: 'numeric', month: 'long', day: 'numeric',
    })
  }

  return (
    <div className="min-h-screen bg-paper flex flex-col">
      {/* Masthead */}
      <header className="border-b-4 border-ink pt-5 pb-2 px-4 text-center">
        <h1 className="font-headline text-4xl font-bold text-ink tracking-tight leading-none">
          The NewsLens
        </h1>
        <p className="text-xs text-ink-light mt-1 uppercase tracking-[0.3em]">
          Reader Setup · {today()}
        </p>
      </header>

      {/* Progress */}
      <div className="max-w-lg mx-auto w-full px-4 mt-6">
        <div className="flex gap-1 mb-1">
          {[1, 2, 3].map(n => (
            <div
              key={n}
              className={`h-1 flex-1 transition-all duration-500 ${
                step >= n ? 'bg-ink' : 'bg-rule'
              }`}
            />
          ))}
        </div>
        <p className="text-xs text-ink-light text-right">
          {step < 3 ? `Step ${step} of 2` : 'Preparing your edition…'}
        </p>
      </div>

      <main className="flex-1 flex items-start justify-center px-4 py-6">
        <div className="w-full max-w-lg">

          {/* Step 1: Topics */}
          {step === 1 && (
            <div className="fade-slide-in border border-rule bg-white p-6">
              <div className="section-rule mb-3" />
              <h2 className="font-headline text-2xl font-bold text-ink mb-1">
                What interests you?
              </h2>
              <p className="text-sm text-ink-light mb-5">
                Select all topics that apply — this seeds your personalised edition.
              </p>
              <div className="grid grid-cols-3 gap-2 mb-6">
                {TOPICS.map(t => (
                  <div key={t.id} className="relative group">
                    <button
                      onClick={() => toggleTopic(t.id)}
                      className={`w-full flex flex-col items-center gap-1 py-3 border text-xs font-semibold
                                  uppercase tracking-wide transition-all ${
                        selectedTopics.includes(t.id)
                          ? 'border-ink bg-ink text-paper'
                          : 'border-rule bg-paper text-ink hover:border-ink'
                      }`}
                    >
                      <span className="text-lg">{t.emoji}</span>
                      <span>{t.label}</span>
                    </button>
                    {/* Tooltip */}
                    <div className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-10
                                    w-44 bg-ink text-paper text-xs leading-snug px-2.5 py-2 rounded
                                    opacity-0 group-hover:opacity-100 transition-opacity duration-150">
                      {t.desc}
                      <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-ink" />
                    </div>
                  </div>
                ))}
              </div>
              {selectedTopics.length === 0 && (
                <p className="text-xs text-ink-light text-center mb-3">
                  Select at least one topic to continue
                </p>
              )}
              <button
                className="btn-primary w-full justify-center py-2.5"
                disabled={selectedTopics.length === 0}
                onClick={() => setStep(2)}
              >
                Continue →
              </button>
              <button
                onClick={() => navigate('/')}
                className="w-full text-center text-xs text-ink-light mt-3 hover:underline"
              >
                ← Back to sign in
              </button>
            </div>
          )}

          {/* Step 2: Reading Style */}
          {step === 2 && (
            <div className="fade-slide-in border border-rule bg-white p-6">
              <div className="section-rule mb-3" />
              <h2 className="font-headline text-2xl font-bold text-ink mb-1">
                How do you like to read?
              </h2>
              <p className="text-sm text-ink-light mb-5">
                This sets which recommendation model runs your feed.
              </p>
              <div className="space-y-2 mb-6">
                {STYLES.map(s => (
                  <button
                    key={s.id}
                    onClick={() => setSelectedStyle(s.id)}
                    className={`w-full text-left border p-4 transition-all ${
                      selectedStyle === s.id
                        ? 'border-ink bg-ink text-paper'
                        : 'border-rule bg-white hover:border-ink'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className={`font-headline text-base font-bold ${
                        selectedStyle === s.id ? 'text-paper' : 'text-ink'
                      }`}>
                        {s.label}
                      </span>
                      <span className={`text-xs uppercase tracking-wide ${
                        selectedStyle === s.id ? 'text-paper/70' : 'text-rule'
                      }`}>
                        {s.note}
                      </span>
                    </div>
                    <p className={`text-sm leading-snug ${
                      selectedStyle === s.id ? 'text-paper/80' : 'text-ink-light'
                    }`}>
                      {s.desc}
                    </p>
                  </button>
                ))}
              </div>
              {error && (
                <p className="text-sm text-red-600 border-l-2 border-red-600 pl-2 mb-4">
                  {error}
                </p>
              )}
              <div className="flex gap-2">
                <button className="btn-ghost flex-1 justify-center" onClick={() => setStep(1)}>
                  ← Back
                </button>
                <button
                  className="btn-primary flex-1 justify-center py-2.5"
                  disabled={!selectedStyle}
                  onClick={handleFinish}
                >
                  Build My Edition →
                </button>
              </div>
            </div>
          )}

          {/* Step 3: Loading */}
          {step === 3 && (
            <div className="fade-slide-in border border-rule bg-white p-10 text-center">
              <div className="section-rule mb-6" />
              <div className="flex justify-center mb-5">
                <div className="relative w-12 h-12">
                  <div className="absolute inset-0 border-2 border-rule rounded-full" />
                  <div className="absolute inset-0 border-2 border-ink rounded-full border-t-transparent animate-spin" />
                </div>
              </div>
              <h2 className="font-headline text-2xl font-bold text-ink mb-2">
                Composing Your Edition…
              </h2>
              <p className="text-sm text-ink-light mb-5">
                Our AI is selecting diverse, relevant articles.
              </p>
              <div className="space-y-2 text-sm text-ink-light">
                {['Analysing your preferences', 'Curating diverse articles', 'Calculating relevance scores'].map((msg, i) => (
                  <div key={msg} className="flex items-center gap-2 justify-center">
                    <span className="text-ink">✓</span>
                    <span className="fade-slide-in" style={{ animationDelay: `${i * 0.4}s` }}>
                      {msg}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>

      <footer className="border-t border-rule py-3 text-center">
        <p className="text-xs text-rule uppercase tracking-widest">
          Capstone Project · Diversity-Aware Recommender System
        </p>
      </footer>
    </div>
  )
}
