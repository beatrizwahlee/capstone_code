import { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import ArticleCard from '../components/ArticleCard.jsx'
import DiversitySidebar from '../components/DiversitySidebar.jsx'
import MetricsDashboard from '../components/MetricsDashboard.jsx'

const DEFAULT_SLIDERS = {
  main_diversity: 0.5,
  calibration: 0.3,
  serendipity: 0.2,
  fairness: 0.2,
}

async function apiRerank(sessionId, sliders) {
  const res = await fetch('/api/rerank', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, sliders }),
  })
  if (!res.ok) throw new Error(`Rerank error ${res.status}`)
  return res.json()
}

async function apiClick(sessionId, newsId, sliders) {
  const res = await fetch('/api/click', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, news_id: newsId, sliders }),
  })
  if (!res.ok) throw new Error(`Click error ${res.status}`)
  return res.json()
}

export default function FeedPage() {
  const navigate = useNavigate()
  const sessionId = localStorage.getItem('sessionId')

  const [recommendations, setRecommendations] = useState([])
  const [metrics, setMetrics] = useState(null)
  const [categoryDist, setCategoryDist] = useState({})
  const [activeMethod, setActiveMethod] = useState('mmr')
  const [historyCount, setHistoryCount] = useState(0)
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [clickedIds, setClickedIds] = useState(new Set())
  const [sliders, setSliders] = useState(DEFAULT_SLIDERS)
  const [error, setError] = useState(null)

  // Debounce timer ref for slider changes
  const debounceRef = useRef(null)

  // Load initial recs from localStorage (set by QuizPage)
  useEffect(() => {
    if (!sessionId) {
      navigate('/quiz')
      return
    }
    const stored = localStorage.getItem('initialRecs')
    if (stored) {
      try {
        const data = JSON.parse(stored)
        applyResponse(data)
        // Restore sliders from quiz prefs if available
        const quizStyle = localStorage.getItem('quizStyle')
        if (quizStyle) {
          const styleSliders = styleToSliders(quizStyle)
          setSliders(styleSliders)
        }
      } catch {
        // ignore parse error
      }
    } else {
      // Fallback: fetch fresh recs
      handleRerank(DEFAULT_SLIDERS, true)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  function styleToSliders(style) {
    if (style === 'accurate') return { main_diversity: 0.1, calibration: 0.5, serendipity: 0.1, fairness: 0.1 }
    if (style === 'explore') return { main_diversity: 0.8, calibration: 0.1, serendipity: 0.6, fairness: 0.2 }
    return DEFAULT_SLIDERS
  }

  function applyResponse(data) {
    if (data.recommendations) setRecommendations(data.recommendations)
    if (data.metrics) setMetrics(data.metrics)
    if (data.category_distribution) setCategoryDist(data.category_distribution)
    if (data.active_method) setActiveMethod(data.active_method)
    if (typeof data.history_count === 'number') setHistoryCount(data.history_count)
  }

  const handleRerank = useCallback(async (newSliders, immediate = false) => {
    if (!sessionId) return
    if (debounceRef.current) clearTimeout(debounceRef.current)

    const run = async () => {
      setLoading(true)
      setError(null)
      try {
        const data = await apiRerank(sessionId, newSliders)
        applyResponse(data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }

    if (immediate) {
      await run()
    } else {
      debounceRef.current = setTimeout(run, 400)
    }
  }, [sessionId])

  function handleSliderChange(newSliders) {
    setSliders(newSliders)
    handleRerank(newSliders, false)
  }

  async function handleRead(newsId) {
    if (!sessionId) return
    setClickedIds(prev => new Set([...prev, newsId]))
    setRefreshing(true)
    setError(null)
    try {
      const data = await apiClick(sessionId, newsId, sliders)
      applyResponse(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setRefreshing(false)
    }
  }

  async function handleRefresh() {
    await handleRerank(sliders, true)
  }

  function handleRetakeQuiz() {
    localStorage.removeItem('quizCompleted')
    localStorage.removeItem('initialRecs')
    localStorage.removeItem('quizStyle')
    navigate('/quiz')
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-2xl">📰</span>
            <span className="text-lg font-bold text-gray-900 tracking-tight">NewsLens</span>
            {!import.meta.env.PROD && (
              <span className="hidden sm:inline badge bg-amber-100 text-amber-700 ml-2">DEMO</span>
            )}
          </div>
          <div className="flex items-center gap-3">
            <span className="text-sm text-gray-500 hidden sm:inline">
              history: <span className="font-semibold text-gray-800">{historyCount}</span>
            </span>
            <button
              onClick={handleRefresh}
              disabled={loading || refreshing}
              className="btn-ghost text-xs py-1.5 px-3"
              title="Refresh recommendations"
            >
              <span className={loading || refreshing ? 'animate-spin inline-block' : ''}>🔄</span>
              <span className="hidden sm:inline ml-1">Refresh</span>
            </button>
            <button
              onClick={handleRetakeQuiz}
              className="btn-ghost text-xs py-1.5 px-3"
              title="Retake quiz"
            >
              ✏️ <span className="hidden sm:inline ml-1">Retake Quiz</span>
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {error && (
          <div className="mb-4 bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-lg">
            {error}
          </div>
        )}

        <div className="flex gap-6">
          {/* Articles column */}
          <main className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-4">
              <h1 className="text-lg font-semibold text-gray-900">
                Your Personalized Feed
              </h1>
              <span className="text-sm text-gray-400">
                {recommendations.length} articles
              </span>
            </div>

            {loading && recommendations.length === 0 ? (
              <div className="space-y-4">
                {Array.from({ length: 5 }).map((_, i) => (
                  <div key={i} className="card p-5 animate-pulse">
                    <div className="h-3 bg-gray-200 rounded w-16 mb-3" />
                    <div className="h-5 bg-gray-200 rounded w-3/4 mb-2" />
                    <div className="h-3 bg-gray-200 rounded w-full mb-1" />
                    <div className="h-3 bg-gray-200 rounded w-5/6" />
                  </div>
                ))}
              </div>
            ) : (
              <div className={`space-y-4 transition-opacity duration-200 ${(loading || refreshing) ? 'opacity-60' : 'opacity-100'}`}>
                {recommendations.map((article, idx) => (
                  <ArticleCard
                    key={article.news_id}
                    article={article}
                    rank={idx + 1}
                    alreadyRead={clickedIds.has(article.news_id)}
                    onRead={handleRead}
                  />
                ))}
                {recommendations.length === 0 && (
                  <div className="card p-12 text-center">
                    <p className="text-gray-400 text-sm">No recommendations yet. Try refreshing.</p>
                    <button className="btn-primary mt-4" onClick={handleRefresh}>Refresh</button>
                  </div>
                )}
              </div>
            )}
          </main>

          {/* Sidebar */}
          <aside className="w-72 flex-shrink-0 hidden lg:block">
            <div className="sticky top-20 space-y-4">
              <DiversitySidebar
                sliders={sliders}
                activeMethod={activeMethod}
                onChange={handleSliderChange}
              />
              {metrics && (
                <MetricsDashboard
                  metrics={metrics}
                  categoryDist={categoryDist}
                  historyCount={historyCount}
                />
              )}
            </div>
          </aside>
        </div>
      </div>
    </div>
  )
}
