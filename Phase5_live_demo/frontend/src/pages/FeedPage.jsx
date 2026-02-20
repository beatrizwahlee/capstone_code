import { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import ArticleCard from '../components/ArticleCard.jsx'
import ArticleModal from '../components/ArticleModal.jsx'
import DiversitySidebar from '../components/DiversitySidebar.jsx'
import MetricsDashboard from '../components/MetricsDashboard.jsx'

const DEFAULT_SLIDERS = {
  main_diversity: 0.5,
  calibration: 0.3,
  serendipity: 0.2,
  fairness: 0.2,
}

function styleToSliders(style) {
  if (style === 'accurate') return { main_diversity: 0.0, calibration: 0.0, serendipity: 0.0, fairness: 0.0 }
  if (style === 'explore')  return { main_diversity: 0.9, calibration: 0.1, serendipity: 0.8, fairness: 0.2 }
  return DEFAULT_SLIDERS
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

function today() {
  return new Date().toLocaleDateString('en-US', {
    weekday: 'long', year: 'numeric', month: 'long', day: 'numeric',
  })
}

export default function FeedPage() {
  const navigate = useNavigate()
  const sessionId = localStorage.getItem('sessionId')
  const displayName = localStorage.getItem('displayName') || 'Guest Reader'

  const [recommendations, setRecommendations] = useState([])
  const [metrics, setMetrics] = useState(null)
  const [categoryDist, setCategoryDist] = useState({})
  const [activeMethod, setActiveMethod] = useState('mmr')
  const [historyCount, setHistoryCount] = useState(0)
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [clickedIds, setClickedIds] = useState(new Set())
  const [readArticles, setReadArticles] = useState([])     // article objects for history panel
  const [sliders, setSliders] = useState(DEFAULT_SLIDERS)
  const [error, setError] = useState(null)
  const [modalArticle, setModalArticle] = useState(null)
  const [echoDismissed, setEchoDismissed] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [historyArticles, setHistoryArticles] = useState([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [diversityPreference, setDiversityPreference] = useState(null)

  const debounceRef = useRef(null)

  function applyResponse(data) {
    if (data.recommendations) setRecommendations(data.recommendations)
    if (data.metrics)         setMetrics(data.metrics)
    if (data.category_distribution) setCategoryDist(data.category_distribution)
    if (data.active_method)   setActiveMethod(data.active_method)
    if (typeof data.history_count === 'number') setHistoryCount(data.history_count)
    if (typeof data.diversity_preference === 'number') setDiversityPreference(data.diversity_preference)
  }

  // On mount: restore initial recs from localStorage (set by quiz or login)
  useEffect(() => {
    if (!sessionId) { navigate('/'); return }

    const stored = localStorage.getItem('initialRecs')
    if (stored) {
      try {
        applyResponse(JSON.parse(stored))
        // Restore sliders from quiz style
        const quizStyle = localStorage.getItem('quizStyle')
        if (quizStyle) setSliders(styleToSliders(quizStyle))
      } catch { /* ignore */ }
    } else {
      handleRerank(DEFAULT_SLIDERS, true)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

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

  // Click = mark as read + refresh feed (live update)
  async function handleRead(newsId) {
    if (!sessionId) return
    setClickedIds(prev => new Set([...prev, newsId]))
    // Track article object for history panel
    const article = recommendations.find(a => a.news_id === newsId)
    if (article) {
      setReadArticles(prev => [article, ...prev.filter(a => a.news_id !== newsId)])
    }
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

  function handleOpenArticle(article) {
    setModalArticle(article)
  }

  function handleCloseModal() {
    setModalArticle(null)
  }

  async function handleRefresh() {
    if (!sessionId) return
    setLoading(true)
    setError(null)
    setClickedIds(new Set())
    setReadArticles([])
    setEchoDismissed(false)
    try {
      const res = await fetch('/api/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      })
      if (!res.ok) throw new Error(`Reset error ${res.status}`)
      const data = await res.json()
      applyResponse(data)
      // Restore initial slider values from quiz style
      const quizStyle = localStorage.getItem('quizStyle')
      if (quizStyle) setSliders(styleToSliders(quizStyle))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function openHistory() {
    setHistoryOpen(true)
    setHistoryLoading(true)
    try {
      const res = await fetch(`/api/history/${sessionId}`)
      if (!res.ok) throw new Error('Could not load history')
      const data = await res.json()
      setHistoryArticles(data.articles || [])
    } catch {
      setHistoryArticles([])
    } finally {
      setHistoryLoading(false)
    }
  }

  function handleSignOut() {
    localStorage.removeItem('quizCompleted')
    localStorage.removeItem('initialRecs')
    localStorage.removeItem('sessionId')
    localStorage.removeItem('displayName')
    localStorage.removeItem('quizStyle')
    navigate('/')
  }

  const [article0, ...rest] = recommendations

  return (
    <div className="min-h-screen bg-paper">

      {/* ── Masthead ── */}
      <header className="bg-paper border-b-4 border-ink sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-4 pt-3 pb-1">
          {/* Top utility row */}
          <div className="flex items-center justify-between text-xs text-ink-light mb-1">
            <span>{today()}</span>
            <div className="flex items-center gap-3">
              <span>
                Reader: <span className="font-semibold text-ink">{displayName}</span>
              </span>
              <button
                onClick={openHistory}
                className="border-l border-rule pl-3 hover:text-ink transition-colors underline-offset-2 hover:underline"
                title="View reading history"
              >
                {historyCount} articles read
              </button>
              <button
                onClick={handleRefresh}
                disabled={loading || refreshing}
                className="border-l border-rule pl-3 hover:text-ink transition-colors"
                title="Refresh feed"
              >
                {(loading || refreshing) ? '⟳ Loading…' : '⟳ Refresh'}
              </button>
              <button
                onClick={() => navigate('/compare')}
                className="border-l border-rule pl-3 hover:text-ink transition-colors"
                title="Compare baseline vs diversity side by side"
              >
                Compare ⇄
              </button>
              <button
                onClick={handleSignOut}
                className="border-l border-rule pl-3 hover:text-ink transition-colors"
              >
                Sign Out
              </button>
            </div>
          </div>
          {/* Masthead name */}
          <div className="text-center py-1">
            <h1 className="font-headline text-3xl md:text-4xl font-bold text-ink tracking-tight leading-none">
              The NewsLens
            </h1>
          </div>
        </div>
        {/* Section bar */}
        <div className="border-t border-rule bg-ink text-paper">
          <div className="max-w-7xl mx-auto px-4 py-1 flex items-center justify-between">
            <span className="text-xs uppercase tracking-[0.25em] text-paper/80">
              Your Personalised Edition
            </span>
            <span className={`text-xs uppercase tracking-widest px-2 py-0.5 font-bold
              ${activeMethod === 'baseline' ? 'bg-white/10' : 'bg-white/20'}`}>
              {activeMethod === 'baseline'  ? 'Accuracy Mode'  :
               activeMethod === 'composite' ? 'Diversity Mode'  :
               activeMethod === 'mmr'       ? 'Balanced Mode'  :
               activeMethod === 'serendipity' ? 'Explore Mode' :
               activeMethod === 'calibrated' ? 'Calibrated Mode' :
               activeMethod === 'xquad'    ? 'Fair Coverage Mode' : activeMethod}
            </span>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-5">
        {error && (
          <div className="mb-4 border-l-4 border-red-600 bg-red-50 text-red-700 text-sm px-4 py-2">
            {error}
          </div>
        )}

        {/* ── Echo chamber warning ── */}
        {metrics?.gini > 0.7 && !echoDismissed && (
          <div className="mb-4 border-l-4 border-amber-500 bg-amber-50 px-4 py-3 flex items-start justify-between gap-4">
            <div>
              <p className="text-sm font-semibold text-amber-800">
                ⚠ Echo chamber detected (Gini {metrics.gini.toFixed(2)})
              </p>
              <p className="text-xs text-amber-700 mt-0.5">
                Your feed is heavily concentrated in a few categories. Try raising the Diversity slider to broaden your recommendations.
              </p>
            </div>
            <button
              onClick={() => setEchoDismissed(true)}
              className="text-amber-500 hover:text-amber-700 text-lg flex-shrink-0"
              title="Dismiss"
            >
              ×
            </button>
          </div>
        )}

        <div className="flex gap-6">
          {/* ── Main content ── */}
          <main className={`flex-1 min-w-0 transition-opacity duration-200 ${
            (loading || refreshing) && recommendations.length > 0 ? 'opacity-60' : 'opacity-100'
          }`}>

            {/* Skeletons while first load */}
            {loading && recommendations.length === 0 ? (
              <div className="space-y-3">
                {Array.from({ length: 5 }).map((_, i) => (
                  <div key={i} className="article-tile animate-pulse px-4 py-4">
                    <div className="h-2.5 bg-rule/40 rounded w-20 mb-2" />
                    <div className="h-5 bg-rule/40 rounded w-3/4 mb-2" />
                    <div className="h-3 bg-rule/30 rounded w-full mb-1" />
                    <div className="h-3 bg-rule/30 rounded w-5/6" />
                  </div>
                ))}
              </div>
            ) : recommendations.length === 0 ? (
              <div className="text-center py-16 border border-rule bg-white">
                <p className="font-headline text-xl text-ink-light mb-3">No articles yet</p>
                <button className="btn-primary" onClick={handleRefresh}>Refresh Edition</button>
              </div>
            ) : (
              <>
                {/* Feature story (first article) */}
                {article0 && (
                  <div className="mb-5 pb-5 border-b-2 border-ink">
                    <div className="section-rule mb-1" />
                    <p className="badge-section mb-2 text-xs">Lead Story</p>
                    <ArticleCard
                      article={article0}
                      rank={1}
                      featured={true}
                      alreadyRead={clickedIds.has(article0.news_id)}
                      onRead={handleRead}
                      onOpen={handleOpenArticle}
                    />
                  </div>
                )}

                {/* Remaining articles in 2-column grid */}
                {rest.length > 0 && (
                  <>
                    <div className="section-rule mb-1" />
                    <p className="badge-section mb-3 text-xs">More Stories</p>
                    <div className="article-grid">
                      {rest.map((article, idx) => (
                        <div key={article.news_id} className="h-full">
                          <ArticleCard
                            article={article}
                            rank={idx + 2}
                            featured={false}
                            alreadyRead={clickedIds.has(article.news_id)}
                            onRead={handleRead}
                            onOpen={handleOpenArticle}
                          />
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </>
            )}
          </main>

          {/* ── Sidebar ── */}
          <aside className="w-64 flex-shrink-0 hidden lg:block">
            <div className="sticky top-28 space-y-4">
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
                  diversityPreference={diversityPreference}
                />
              )}

              {/* ── Reading history ── */}
              {readArticles.length > 0 && (
                <div className="border border-rule bg-white p-4">
                  <div className="section-rule mb-2" />
                  <h2 className="text-xs font-bold uppercase tracking-widest text-ink mb-3">
                    Recently Read
                  </h2>
                  <div className="space-y-3">
                    {readArticles.slice(0, 6).map(a => (
                      <div key={a.news_id} className="flex items-start gap-2">
                        <span className={`mt-0.5 flex-shrink-0 w-2 h-2 rounded-full ${
                          { sports:'bg-blue-500', health:'bg-green-500', technology:'bg-indigo-500',
                            politics:'bg-red-500', finance:'bg-amber-500', entertainment:'bg-pink-500',
                            travel:'bg-cyan-500', science:'bg-orange-500', foodanddrink:'bg-lime-500',
                            lifestyle:'bg-rose-500', autos:'bg-slate-500', weather:'bg-sky-500'
                          }[a.category] ?? 'bg-gray-400'
                        }`} />
                        <div className="min-w-0">
                          <p className="text-xs font-bold uppercase tracking-wider text-ink-light leading-none mb-0.5">
                            {a.category}
                          </p>
                          <p className="text-xs text-ink leading-snug line-clamp-2">
                            {a.title}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </aside>
        </div>
      </div>

      {/* ── Article modal ── */}
      {modalArticle && (
        <ArticleModal
          article={modalArticle}
          onClose={handleCloseModal}
          onRead={handleRead}
          alreadyRead={clickedIds.has(modalArticle.news_id)}
        />
      )}

      {/* ── Reading history modal ── */}
      {historyOpen && (
        <div
          className="modal-overlay"
          onClick={() => setHistoryOpen(false)}
        >
          <div
            className="bg-paper border border-rule w-full max-w-lg max-h-[80vh] flex flex-col shadow-xl"
            onClick={e => e.stopPropagation()}
          >
            {/* Modal header */}
            <div className="border-b-2 border-ink px-6 py-4 flex items-center justify-between flex-shrink-0">
              <div>
                <p className="text-xs uppercase tracking-widest text-ink-light mb-0.5">Your session</p>
                <h2 className="font-headline text-xl font-bold text-ink">Reading History</h2>
              </div>
              <button
                onClick={() => setHistoryOpen(false)}
                className="text-ink-light hover:text-ink text-2xl leading-none"
              >
                ×
              </button>
            </div>

            {/* Modal body */}
            <div className="flex-1 overflow-y-auto px-6 py-4">
              {historyLoading ? (
                <div className="space-y-3">
                  {[...Array(5)].map((_, i) => (
                    <div key={i} className="animate-pulse flex gap-3">
                      <div className="w-2 h-2 rounded-full bg-rule/40 mt-1.5 flex-shrink-0" />
                      <div className="flex-1">
                        <div className="h-2.5 bg-rule/40 rounded w-16 mb-1.5" />
                        <div className="h-4 bg-rule/30 rounded w-3/4" />
                      </div>
                    </div>
                  ))}
                </div>
              ) : historyArticles.length === 0 ? (
                <p className="text-sm text-ink-light text-center py-8">No reading history yet.</p>
              ) : (
                <div className="space-y-3">
                  {historyArticles.map((a, i) => {
                    const dotColor = {
                      sports:'bg-blue-500', health:'bg-green-500', technology:'bg-indigo-500',
                      politics:'bg-red-500', finance:'bg-amber-500', entertainment:'bg-pink-500',
                      travel:'bg-cyan-500', science:'bg-orange-500', foodanddrink:'bg-lime-500',
                      lifestyle:'bg-rose-500', autos:'bg-slate-500', weather:'bg-sky-500',
                    }[a.category] ?? 'bg-gray-400'
                    return (
                      <div key={a.news_id} className="flex items-start gap-3 pb-3 border-b border-rule/40 last:border-0">
                        <span className={`mt-1.5 w-2 h-2 rounded-full flex-shrink-0 ${dotColor}`} />
                        <div className="min-w-0">
                          <p className="text-xs font-bold uppercase tracking-wider text-ink-light mb-0.5">
                            {a.category}{a.subcategory ? ` · ${a.subcategory}` : ''}
                          </p>
                          <p className="text-sm text-ink leading-snug">{a.title}</p>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>

            {/* Modal footer */}
            <div className="border-t border-rule px-6 py-3 flex-shrink-0 flex items-center justify-between">
              <span className="text-xs text-ink-light">{historyCount} total articles in history</span>
              <button
                onClick={() => setHistoryOpen(false)}
                className="btn-ghost text-xs py-1.5 px-3"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
