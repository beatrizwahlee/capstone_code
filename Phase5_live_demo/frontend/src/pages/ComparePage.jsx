import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import ArticleCard from '../components/ArticleCard.jsx'

const BASELINE_SLIDERS = { main_diversity: 0.0, calibration: 0.0, serendipity: 0.0, fairness: 0.0 }
const DIVERSITY_SLIDERS = { main_diversity: 0.5, calibration: 0.3, serendipity: 0.2, fairness: 0.2 }

const CAT_DISPLAY = {
  sports: 'Sports', health: 'Health', technology: 'Technology',
  politics: 'Politics', finance: 'Finance', entertainment: 'Entertainment',
  travel: 'Travel', science: 'Science', foodanddrink: 'Food & Drink',
  lifestyle: 'Lifestyle', autos: 'Autos', weather: 'Weather',
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

function MetricRow({ label, baseVal, divVal, lowerBetter }) {
  const improved =
    baseVal != null && divVal != null
      ? lowerBetter ? divVal < baseVal : divVal > baseVal
      : null
  const fmt = v => (v != null ? (v < 1 && v > 0 ? (v * 100).toFixed(1) + '%' : v.toFixed(3)) : '—')

  return (
    <div className="flex items-center justify-between py-2 border-b border-rule/40 last:border-0 text-xs">
      <span className="text-ink-light uppercase tracking-wide">{label}</span>
      <div className="flex items-center gap-8">
        <span className="font-mono text-ink w-14 text-right">{fmt(baseVal)}</span>
        <span className={`font-mono w-14 text-right font-semibold ${
          improved === true  ? 'text-green-700' :
          improved === false ? 'text-red-600'   : 'text-masthead'
        }`}>
          {fmt(divVal)}
          {improved === true ? ' ↑' : improved === false ? ' ↓' : ''}
        </span>
      </div>
    </div>
  )
}

function CategoryBadge({ category }) {
  const colors = {
    sports: 'text-blue-800 border-blue-200 bg-blue-50',
    health: 'text-green-800 border-green-200 bg-green-50',
    technology: 'text-indigo-800 border-indigo-200 bg-indigo-50',
    politics: 'text-red-800 border-red-200 bg-red-50',
    finance: 'text-amber-800 border-amber-200 bg-amber-50',
    entertainment: 'text-pink-800 border-pink-200 bg-pink-50',
    travel: 'text-cyan-800 border-cyan-200 bg-cyan-50',
    science: 'text-orange-800 border-orange-200 bg-orange-50',
    foodanddrink: 'text-lime-800 border-lime-200 bg-lime-50',
    lifestyle: 'text-rose-800 border-rose-200 bg-rose-50',
    autos: 'text-slate-700 border-slate-200 bg-slate-50',
    weather: 'text-sky-800 border-sky-200 bg-sky-50',
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

  const [baselineRecs, setBaselineRecs]       = useState([])
  const [diversityRecs, setDiversityRecs]     = useState([])
  const [baselineMetrics, setBaselineMetrics] = useState(null)
  const [diversityMetrics, setDiversityMetrics] = useState(null)
  const [loadingB, setLoadingB] = useState(true)
  const [loadingD, setLoadingD] = useState(true)
  const [errorB, setErrorB]     = useState(null)
  const [errorD, setErrorD]     = useState(null)

  useEffect(() => {
    if (!sessionId) { navigate('/'); return }

    // Fetch both simultaneously
    apiRerank(sessionId, BASELINE_SLIDERS)
      .then(d => { setBaselineRecs(d.recommendations || []); setBaselineMetrics(d.metrics) })
      .catch(e => setErrorB(e.message))
      .finally(() => setLoadingB(false))

    apiRerank(sessionId, DIVERSITY_SLIDERS)
      .then(d => { setDiversityRecs(d.recommendations || []); setDiversityMetrics(d.metrics) })
      .catch(e => setErrorD(e.message))
      .finally(() => setLoadingD(false))
  }, []) // eslint-disable-line

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
            Baseline (pure relevance) vs Diversity-Enhanced — same reader, same history
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-4 py-5">

        {/* Metric comparison bar */}
        {(bm || dm) && (
          <div className="border border-rule bg-white p-4 mb-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-bold uppercase tracking-widest text-ink">
                Diversity Metrics Comparison
              </h3>
              <div className="flex items-center gap-8 text-xs">
                <span className="text-ink-light font-medium">Baseline</span>
                <span className="font-bold text-masthead">Diversity</span>
              </div>
            </div>
            <MetricRow label="Gini ↓ (lower = more diverse)" baseVal={bm?.gini}     divVal={dm?.gini}     lowerBetter={true}  />
            <MetricRow label="ILD ↑ (higher = more varied)"  baseVal={bm?.ild}      divVal={dm?.ild}      lowerBetter={false} />
            <MetricRow label="Coverage ↑"                    baseVal={bm?.coverage} divVal={dm?.coverage} lowerBetter={false} />
            <MetricRow label="Entropy ↑"                     baseVal={bm?.entropy}  divVal={dm?.entropy}  lowerBetter={false} />
            {newCats.size > 0 && (
              <div className="mt-3 pt-3 border-t border-rule">
                <p className="text-xs text-ink-light mb-1.5">
                  <span className="text-green-700 font-semibold">{newCats.size} new categor{newCats.size > 1 ? 'ies' : 'y'}</span> introduced by diversity mode:
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
            <div className="px-4 py-3 border border-rule bg-white mb-3 flex items-center justify-between">
              <div>
                <p className="text-xs text-ink-light uppercase tracking-widest mb-0.5">No diversity adjustment</p>
                <h2 className="font-headline text-xl font-bold text-ink">Baseline</h2>
              </div>
              {bm && (
                <div className="text-right">
                  <div className={`text-xl font-bold font-headline ${bm.gini > 0.7 ? 'text-red-600' : 'text-ink'}`}>
                    {bm.gini?.toFixed(2)}
                  </div>
                  <div className="text-xs text-ink-light">Gini</div>
                </div>
              )}
            </div>

            {errorB && <div className="p-3 text-xs text-red-600 border-l-4 border-red-400 mb-3">{errorB}</div>}

            {loadingB ? (
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
            <div className="px-4 py-3 border border-rule bg-masthead mb-3 flex items-center justify-between">
              <div>
                <p className="text-xs text-paper/60 uppercase tracking-widest mb-0.5">With diversity</p>
                <h2 className="font-headline text-xl font-bold text-paper">Diversity Mode</h2>
              </div>
              {dm && (
                <div className="text-right">
                  <div className={`text-xl font-bold font-headline ${dm.gini < 0.5 ? 'text-green-300' : 'text-yellow-300'}`}>
                    {dm.gini?.toFixed(2)}
                  </div>
                  <div className="text-xs text-paper/60">Gini</div>
                </div>
              )}
            </div>

            {errorD && <div className="p-3 text-xs text-red-600 border-l-4 border-red-400 mb-3">{errorD}</div>}

            {loadingD ? (
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
