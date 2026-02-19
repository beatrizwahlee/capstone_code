import { useEffect } from 'react'

const CAT_DISPLAY = {
  sports: 'Sports', health: 'Health', technology: 'Technology',
  politics: 'Politics', finance: 'Finance', entertainment: 'Entertainment',
  travel: 'Travel', science: 'Science', foodanddrink: 'Food & Drink',
  lifestyle: 'Lifestyle', autos: 'Autos', weather: 'Weather',
}

const CAT_COLORS = {
  sports:        'bg-blue-100 text-blue-900',
  health:        'bg-green-100 text-green-900',
  technology:    'bg-indigo-100 text-indigo-900',
  politics:      'bg-red-100 text-red-900',
  finance:       'bg-amber-100 text-amber-900',
  entertainment: 'bg-pink-100 text-pink-900',
  travel:        'bg-cyan-100 text-cyan-900',
  science:       'bg-orange-100 text-orange-900',
  foodanddrink:  'bg-lime-100 text-lime-900',
  lifestyle:     'bg-rose-100 text-rose-900',
  autos:         'bg-slate-100 text-slate-900',
  weather:       'bg-sky-100 text-sky-900',
}

function fakeDate() {
  const d = new Date()
  d.setDate(d.getDate() - Math.floor(Math.random() * 5))
  return d.toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })
}

// Expand abstract into plausible article body paragraphs
function buildBody(abstract) {
  if (!abstract) return ['No summary available for this article.']
  const sentences = abstract.match(/[^.!?]+[.!?]+/g) || [abstract]
  const paras = []
  // Para 1: first half of sentences
  const mid = Math.ceil(sentences.length / 2)
  paras.push(sentences.slice(0, mid).join(' ').trim())
  if (sentences.length > 1) {
    paras.push(sentences.slice(mid).join(' ').trim())
  }
  paras.push(
    'This story is part of your AI-curated edition. Our diversity-aware recommender selected this article based on your reading history and preference settings.'
  )
  return paras.filter(Boolean)
}

export default function ArticleModal({ article, onClose, onRead, alreadyRead }) {
  const { news_id, title, category, subcategory, abstract, score } = article
  const catDisplay = CAT_DISPLAY[category] ?? category
  const catColor = CAT_COLORS[category] ?? 'bg-gray-100 text-gray-800'
  const body = buildBody(abstract)
  const pubDate = fakeDate()

  // Close on Escape key
  useEffect(() => {
    function handleKey(e) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onClose])

  // Prevent body scroll while modal is open
  useEffect(() => {
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = '' }
  }, [])

  return (
    <div className="modal-overlay fade-in" onClick={onClose}>
      <div
        className="bg-paper border border-rule w-full max-w-2xl max-h-[90vh] overflow-y-auto
                   shadow-2xl flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        {/* Modal header bar */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-rule bg-white">
          <span className={`text-xs font-bold uppercase tracking-widest px-2 py-0.5 ${catColor}`}>
            {catDisplay}
            {subcategory && ` · ${subcategory}`}
          </span>
          <button
            onClick={onClose}
            className="text-ink-light hover:text-ink text-xl leading-none px-1"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        {/* Article content */}
        <div className="px-6 pt-5 pb-6">
          {/* Headline */}
          <h1 className="font-headline text-2xl md:text-3xl font-bold text-ink leading-tight mb-3">
            {title}
          </h1>

          {/* Byline */}
          <div className="thin-rule my-3" />
          <div className="flex items-center justify-between text-xs text-ink-light mb-4">
            <span>
              <span className="font-semibold text-ink">NewsLens AI</span>
              {' '}· AI-curated recommendation
            </span>
            <span>{pubDate}</span>
          </div>
          <div className="thin-rule my-3" />

          {/* Body */}
          <div className="space-y-4">
            {body.map((para, i) => (
              <p key={i} className={`text-sm leading-relaxed ${
                i === body.length - 1 ? 'text-ink-light italic text-xs border-t border-rule pt-3' : 'text-ink'
              }`}>
                {para}
              </p>
            ))}
          </div>

          {/* Relevance score */}
          <div className="thin-rule mt-5 mb-4" />
          <div className="flex items-center justify-between text-xs text-ink-light">
            <span>Relevance score: <span className="font-semibold text-ink">{(score * 100).toFixed(0)}%</span></span>
            <span className="uppercase tracking-widest">AI Recommendation</span>
          </div>
        </div>

        {/* Footer actions */}
        <div className="border-t border-rule px-5 py-3 flex items-center justify-between bg-white">
          <button onClick={onClose} className="btn-ghost text-xs py-1.5 px-3">
            ← Back to Feed
          </button>
          <button
            onClick={() => { if (!alreadyRead) onRead(news_id); onClose() }}
            disabled={alreadyRead}
            className={`btn-read ${alreadyRead ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {alreadyRead ? '✓ Marked as Read' : 'Mark as Read'}
          </button>
        </div>
      </div>
    </div>
  )
}
