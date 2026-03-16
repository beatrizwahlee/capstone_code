import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import ArticleCard from '../components/ArticleCard.jsx'

const CAT_DISPLAY = {
  sports: 'Sports', news: 'News', health: 'Health',
  finance: 'Finance', entertainment: 'Entertainment', travel: 'Travel',
  foodanddrink: 'Food & Drink', lifestyle: 'Lifestyle', autos: 'Autos',
  weather: 'Weather', music: 'Music', movies: 'Movies',
  tv: 'TV', video: 'Video', kids: 'Kids',
}

async function apiCompare(sessionId, k) {
  const res = await fetch(`/api/compare/${sessionId}?k=${k}`)
  if (res.status === 404) throw new Error('SESSION_EXPIRED')
  if (!res.ok) throw new Error(`Compare error ${res.status}`)
  return res.json()
}

function MetricRow({ label, baseVal, divVal, lowerBetter, pct = false }) {
  const improved =
    baseVal != null && divVal != null
      ? lowerBetter ? divVal < baseVal : divVal > baseVal
      : null

  const fmt = v => {
    if (v == null) return '—'
    return pct ? (v * 100).toFixed(1) + '%' : v.toFixed(3)
  }

  const delta = baseVal != null && divVal != null ? divVal - baseVal : null
  const fmtDelta = () => {
    if (delta == null) return '—'
    const abs = pct ? (Math.abs(delta) * 100).toFixed(1) + '%' : Math.abs(delta).toFixed(3)
    return (delta >= 0 ? '+' : '−') + abs
  }

  return (
    <div className="grid grid-cols-[1fr_auto] items-center py-2 border-b border-rule/40 last:border-0 text-xs gap-4">
      <span className="text-ink-light uppercase tracking-wide">{label}</span>
      <div className="flex items-center gap-6">
        <span className="font-mono text-ink w-14 text-right">{fmt(baseVal)}</span>
        <span className={`font-mono w-14 text-right font-semibold ${
          improved === true  ? 'text-green-700' :
          improved === false ? 'text-red-600'   : 'text-masthead'
        }`}>
          {fmt(divVal)}
        </span>
        <span className={`font-mono w-16 text-right font-bold ${
          improved === true  ? 'text-green-700' :
          improved === false ? 'text-red-600'   : 'text-ink-light'
        }`}>
          {fmtDelta()}
        </span>
      </div>
    </div>
  )
}

function CategoryBadge({ category }) {
  const colors = {
    sports:        'text-blue-800 border-blue-200 bg-blue-50',
    news:          'text-red-800 border-red-200 bg-red-50',
    health:        'text-green-800 border-green-200 bg-green-50',
    finance:       'text-amber-800 border-amber-200 bg-amber-50',
    entertainment: 'text-pink-800 border-pink-200 bg-pink-50',
    travel:        'text-cyan-800 border-cyan-200 bg-cyan-50',
    foodanddrink:  'text-lime-800 border-lime-200 bg-lime-50',
    lifestyle:     'text-rose-800 border-rose-200 bg-rose-50',
    autos:         'text-slate-700 border-slate-200 bg-slate-50',
    weather:       'text-sky-800 border-sky-200 bg-sky-50',
    music:         'text-purple-800 border-purple-200 bg-purple-50',
    movies:        'text-indigo-800 border-indigo-200 bg-indigo-50',
    tv:            'text-orange-800 border-orange-200 bg-orange-50',
    video:         'text-teal-800 border-teal-200 bg-teal-50',
    kids:          'text-yellow-800 border-yellow-200 bg-yellow-50',
  }
  const cls = colors[category] ?? 'text-gray-700 border-gray-200 bg-gray-50'
  return (
    <span className={`inline-block text-[10px] font-bold uppercase tracking-widest border px-1.5 py-0.5 ${cls}`}>
      {CAT_DISPLAY[category] ?? category}
    </span>
  )
}

function ArticleRow({ rank, article, highlight }) {
  return (
    <div className={`flex items-start gap-3 py-2.5 border-b border-rule/40 last:border-0 ${
      highlight ? 'bg-green-50/60' : ''
    }`}>
      <span className="text-xs font-bold text-ink-light w-5 flex-shrink-0 mt-0.5">
        {rank}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1 flex-wrap">
          <CategoryBadge category={article.category} />
          {highlight && (
            <span className="text-[10px] text-green-700 font-semibold uppercase tracking-wide">
              ✦ New category
            </span>
          )}
        </div>
        <p className="text-sm font-semibold text-ink leading-snug line-clamp-2">
          {article.title}
        </p>
        <p className="text-xs text-ink-light mt-0.5">{Math.round(article.score * 100)}% match</p>
      </div>
    </div>
  )
}

export default function ComparePage() {
  const navigate = useNavigate()
  const sessionId = localStorage.getItem('sessionId')
  const displayName = localStorage.getItem('displayName') || 'Reader'

  const [k, setK] = useState(10)
  const [baselineRecs, setBaselineRecs]         = useState([])
  const [diversityRecs, setDiversityRecs]       = useState([])
  const [baselineMetrics, setBaselineMetrics]   = useState(null)
  const [diversityMetrics, setDiversityMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [errorB, setErrorB]   = useState(null)
  const [errorD, setErrorD]   = useState(null)

  useEffect(() => {
    if (!sessionId) { navigate('/'); return }

    setLoading(true)
    setErrorB(null)
    setErrorD(null)

    // Single read-only call — does not mutate session slider state
    apiCompare(sessionId, k)
      .then(d => {
        setBaselineRecs(d.baseline?.recommendations || [])
        setBaselineMetrics(d.baseline?.metrics ?? null)
        setDiversityRecs(d.diversity?.recommendations || [])
        setDiversityMetrics(d.diversity?.metrics ?? null)
      })
      .catch(e => {
        if (e.message === 'SESSION_EXPIRED') {
          localStorage.removeItem('sessionId')
          localStorage.removeItem('quizCompleted')
          localStorage.removeItem('initialRecs')
          navigate('/')
        } else {
          setErrorB(e.message)
          setErrorD(e.message)
        }
      })
      .finally(() => setLoading(false))
  }, [k]) // eslint-disable-line

  // Find categories in diversity feed that are NOT in baseline feed
  const baselineCats = new Set(baselineRecs.map(a => a.category))
  const newCats      = new Set(diversityRecs.map(a => a.category).filter(c => !baselineCats.has(c)))

  const bm = baselineMetrics
  const dm = diversityMetrics

  return (
    <div className="min-h-screen bg-paper">

      {/* Header */}
      <header className="border-b-4 border-ink bg-paper sticky top-0 z-20">
        <div className="max-w-6xl mx-auto px-4 pt-3 pb-2 flex items-center justify-between">
          <button
            onClick={() => navigate('/feed')}
            className="btn-ghost text-xs py-1.5 px-3 flex items-center gap-1"
          >
            ← Back to Feed
          </button>
          <div className="text-center">
            <h1 className="font-headline text-2xl md:text-3xl font-bold text-ink tracking-tight">
              The NewsLens
            </h1>
            <p className="text-xs text-ink-light">
              Reader: <span className="font-semibold text-ink">{displayName}</span>
            </p>
          </div>
          <div className="w-28 text-right">
            <span className="text-xs text-ink-light uppercase tracking-widest">Comparison View</span>
          </div>
        </div>
        <div className="border-t border-rule bg-ink text-paper">
          <div className="max-w-6xl mx-auto px-4 py-1 text-xs uppercase tracking-widest text-paper/70 text-center">
            Baseline (pure relevance) vs Explore Mode — same reader, same history
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-4 py-5">

        {/* k selector */}
        <div className="flex items-center gap-3 mb-5">
          <span className="text-xs font-bold uppercase tracking-widest text-ink-light">
            Recommendations per model:
          </span>
          {[10, 20, 30].map(n => (
            <button
              key={n}
              onClick={() => setK(n)}
              className={`text-sm font-bold px-4 py-1.5 border transition-colors ${
                k === n
                  ? 'border-ink bg-ink text-paper'
                  : 'border-rule text-ink hover:border-ink'
              }`}
            >
              {n}
            </button>
          ))}
          <span className="text-xs text-ink-light ml-2">
            — increase to test whether diversity gains are real or just slot inflation
          </span>
        </div>

        {/* Metric comparison bar */}
        {(bm || dm) && (
          <div className="border border-rule bg-white p-4 mb-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-bold uppercase tracking-widest text-ink">
                Metrics Comparison
              </h3>
              <div className="flex items-center gap-6 text-xs">
                <span className="text-ink-light font-medium w-14 text-right">Baseline</span>
                <span className="font-bold text-masthead w-14 text-right">Explore</span>
                <span className="font-bold text-ink w-16 text-right">Δ Change</span>
              </div>
            </div>
            {/* Accuracy */}
            <div className="pb-2 mb-1">
              <p className="text-[10px] font-bold uppercase tracking-widest text-ink-light/60 mb-1">Accuracy</p>
              <MetricRow label="Avg. Relevance Score ↑" baseVal={bm?.avg_relevance} divVal={dm?.avg_relevance} lowerBetter={false} pct={true} />
            </div>
            {/* Diversity */}
            <div className="pt-1">
              <p className="text-[10px] font-bold uppercase tracking-widest text-ink-light/60 mb-1">Diversity</p>
              <MetricRow label="Gini ↓ (lower = more diverse)" baseVal={bm?.gini}     divVal={dm?.gini}     lowerBetter={true}  pct={false} />
              <MetricRow label="ILD ↑ (higher = more varied)"  baseVal={bm?.ild}      divVal={dm?.ild}      lowerBetter={false} pct={true}  />
              <MetricRow label="Coverage ↑"                    baseVal={bm?.coverage} divVal={dm?.coverage} lowerBetter={false} pct={true}  />
              <MetricRow label="Entropy ↑"                     baseVal={bm?.entropy}  divVal={dm?.entropy}  lowerBetter={false} pct={false} />
            </div>
            {newCats.size > 0 && (
              <div className="mt-3 pt-3 border-t border-rule">
                <p className="text-xs text-ink-light mb-1.5">
                  <span className="text-green-700 font-semibold">{newCats.size} new categor{newCats.size > 1 ? 'ies' : 'y'}</span> introduced by explore mode:
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {[...newCats].map(c => <CategoryBadge key={c} category={c} />)}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Two-column feed */}
        <div className="grid grid-cols-[1fr_1px_1fr] gap-0 items-start">

          {/* ── Baseline column ── */}
          <div>
            <div className="px-4 py-3 border border-rule bg-white mb-3">
              <p className="text-xs text-ink-light uppercase tracking-widest mb-0.5">No diversity adjustment</p>
              <h2 className="font-headline text-xl font-bold text-ink">Baseline</h2>
            </div>

            {errorB && <div className="p-3 text-xs text-red-600 border-l-4 border-red-400 mb-3">{errorB}</div>}

            {loading ? (
              <div className="space-y-2">
                {[...Array(8)].map((_, i) => (
                  <div key={i} className="border border-rule bg-white px-4 py-3 animate-pulse">
                    <div className="h-2.5 bg-rule/40 rounded w-16 mb-2" />
                    <div className="h-4 bg-rule/40 rounded w-3/4 mb-1" />
                    <div className="h-3 bg-rule/30 rounded w-1/3" />
                  </div>
                ))}
              </div>
            ) : (
              <div className="border border-rule bg-white divide-y divide-rule/40 px-4">
                {baselineRecs.map((a, i) => (
                  <ArticleRow key={a.news_id} rank={i + 1} article={a} highlight={false} />
                ))}
              </div>
            )}
          </div>

          {/* Vertical divider */}
          <div className="bg-rule self-stretch mx-4" />

          {/* ── Diversity column ── */}
          <div>
            <div className="px-4 py-3 border border-rule bg-masthead mb-3">
              <p className="text-xs text-paper/60 uppercase tracking-widest mb-0.5">With exploration</p>
              <h2 className="font-headline text-xl font-bold text-paper">Explore Mode</h2>
            </div>

            {errorD && <div className="p-3 text-xs text-red-600 border-l-4 border-red-400 mb-3">{errorD}</div>}

            {loading ? (
              <div className="space-y-2">
                {[...Array(8)].map((_, i) => (
                  <div key={i} className="border border-rule bg-white px-4 py-3 animate-pulse">
                    <div className="h-2.5 bg-rule/40 rounded w-16 mb-2" />
                    <div className="h-4 bg-rule/40 rounded w-3/4 mb-1" />
                    <div className="h-3 bg-rule/30 rounded w-1/3" />
                  </div>
                ))}
              </div>
            ) : (
              <div className="border border-rule bg-white divide-y divide-rule/40 px-4">
                {diversityRecs.map((a, i) => (
                  <ArticleRow
                    key={a.news_id}
                    rank={i + 1}
                    article={a}
                    highlight={newCats.has(a.category)}
                  />
                ))}
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  )
}
