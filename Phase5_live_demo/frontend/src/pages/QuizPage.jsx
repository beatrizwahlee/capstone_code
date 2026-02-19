import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

const TOPICS = [
  { id: 'sports',        label: 'Sports',         emoji: '⚽' },
  { id: 'technology',    label: 'Technology',      emoji: '💻' },
  { id: 'health',        label: 'Health',          emoji: '🏥' },
  { id: 'politics',      label: 'Politics',        emoji: '🏛️' },
  { id: 'finance',       label: 'Finance',         emoji: '📈' },
  { id: 'entertainment', label: 'Entertainment',   emoji: '🎬' },
  { id: 'travel',        label: 'Travel',          emoji: '✈️' },
  { id: 'science',       label: 'Science',         emoji: '🔬' },
  { id: 'foodanddrink',  label: 'Food & Drink',    emoji: '🍳' },
  { id: 'lifestyle',     label: 'Lifestyle',       emoji: '🌿' },
  { id: 'autos',         label: 'Autos',           emoji: '🚗' },
  { id: 'weather',       label: 'Weather',         emoji: '⛅' },
]

const STYLES = [
  {
    id: 'accurate',
    emoji: '🎯',
    label: 'Accurate',
    desc: 'Stick to my topics — keep it relevant',
  },
  {
    id: 'balanced',
    emoji: '🔀',
    label: 'Balanced',
    desc: 'Mix relevance with discovery',
  },
  {
    id: 'explore',
    emoji: '🎲',
    label: 'Explore',
    desc: 'Surprise me with new perspectives',
  },
]

async function postQuiz(sessionId, topics, style) {
  const res = await fetch('/api/quiz', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      preferences: { topics, style },
    }),
  })
  if (!res.ok) throw new Error(`Quiz API error ${res.status}`)
  return res.json()
}

async function ensureSession() {
  let sessionId = localStorage.getItem('sessionId')
  const res = await fetch('/api/session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  })
  const data = await res.json()
  localStorage.setItem('sessionId', data.session_id)
  return data.session_id
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
      localStorage.setItem('quizCompleted', 'true')
      localStorage.setItem('initialRecs', JSON.stringify(data))
      // Short delay to show loading animation
      await new Promise(r => setTimeout(r, 1400))
      navigate('/feed')
    } catch (e) {
      setError(e.message)
      setStep(2)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 flex flex-col items-center justify-center px-4 py-12">
      {/* Logo */}
      <div className="mb-8 text-center">
        <span className="text-4xl">📰</span>
        <h1 className="mt-2 text-3xl font-bold text-gray-900 tracking-tight">NewsLens</h1>
        <p className="text-gray-500 mt-1 text-sm">Personalized news powered by diversity AI</p>
      </div>

      {/* Progress bar */}
      <div className="w-full max-w-lg mb-8">
        <div className="flex gap-2">
          {[1, 2, 3].map(n => (
            <div
              key={n}
              className={`h-1.5 flex-1 rounded-full transition-all duration-500 ${
                step >= n ? 'bg-blue-600' : 'bg-gray-200'
              }`}
            />
          ))}
        </div>
        <p className="text-xs text-gray-400 mt-2 text-right">
          {step < 3 ? `Step ${step} of 2` : 'Setting up your feed…'}
        </p>
      </div>

      {/* Step 1: Topics */}
      {step === 1 && (
        <div className="w-full max-w-lg card p-8 fade-slide-in">
          <h2 className="text-xl font-semibold text-gray-900 mb-1">What topics interest you?</h2>
          <p className="text-sm text-gray-500 mb-6">Select all that apply — you can always change this later.</p>
          <div className="grid grid-cols-3 gap-3 mb-8">
            {TOPICS.map(t => (
              <button
                key={t.id}
                onClick={() => toggleTopic(t.id)}
                className={`flex flex-col items-center gap-1.5 px-3 py-3 rounded-xl border-2 text-sm font-medium transition-all ${
                  selectedTopics.includes(t.id)
                    ? 'border-blue-500 bg-blue-50 text-blue-700'
                    : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300 hover:bg-gray-50'
                }`}
              >
                <span className="text-xl">{t.emoji}</span>
                <span>{t.label}</span>
              </button>
            ))}
          </div>
          <button
            className="btn-primary w-full justify-center py-3"
            disabled={selectedTopics.length === 0}
            onClick={() => setStep(2)}
          >
            Continue →
          </button>
          {selectedTopics.length === 0 && (
            <p className="text-xs text-gray-400 text-center mt-2">Select at least one topic</p>
          )}
        </div>
      )}

      {/* Step 2: Reading Style */}
      {step === 2 && (
        <div className="w-full max-w-lg card p-8 fade-slide-in">
          <h2 className="text-xl font-semibold text-gray-900 mb-1">How do you like to read?</h2>
          <p className="text-sm text-gray-500 mb-6">This sets your default diversity preference.</p>
          <div className="grid grid-cols-3 gap-4 mb-8">
            {STYLES.map(s => (
              <button
                key={s.id}
                onClick={() => setSelectedStyle(s.id)}
                className={`flex flex-col items-center gap-2 px-4 py-5 rounded-xl border-2 text-sm transition-all ${
                  selectedStyle === s.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50'
                }`}
              >
                <span className="text-3xl">{s.emoji}</span>
                <span className="font-semibold text-gray-900">{s.label}</span>
                <span className="text-xs text-gray-500 text-center leading-snug">{s.desc}</span>
              </button>
            ))}
          </div>
          {error && (
            <p className="text-sm text-red-600 bg-red-50 rounded-lg px-3 py-2 mb-4">{error}</p>
          )}
          <div className="flex gap-3">
            <button className="btn-ghost flex-1 justify-center" onClick={() => setStep(1)}>
              ← Back
            </button>
            <button
              className="btn-primary flex-1 justify-center py-3"
              disabled={!selectedStyle}
              onClick={handleFinish}
            >
              Build My Feed →
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Loading */}
      {step === 3 && (
        <div className="w-full max-w-lg card p-12 fade-slide-in text-center">
          <div className="flex justify-center mb-6">
            <div className="relative w-16 h-16">
              <div className="absolute inset-0 border-4 border-blue-200 rounded-full" />
              <div className="absolute inset-0 border-4 border-blue-600 rounded-full border-t-transparent animate-spin" />
            </div>
          </div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Building your personalized feed…</h2>
          <p className="text-sm text-gray-500">Our AI is selecting diverse, relevant articles just for you.</p>
          <div className="mt-6 flex flex-col gap-2">
            {['Analyzing your preferences', 'Selecting diverse articles', 'Calculating relevance scores'].map((msg, i) => (
              <div key={msg} className="flex items-center gap-2 text-sm text-gray-400 justify-center">
                <span className="text-green-500">✓</span>
                <span style={{ animationDelay: `${i * 0.4}s` }} className="fade-slide-in">{msg}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
