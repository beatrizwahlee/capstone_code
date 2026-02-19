const CAT_DISPLAY = {
  sports: 'Sports', health: 'Health', technology: 'Technology',
  politics: 'Politics', finance: 'Finance', entertainment: 'Entertainment',
  travel: 'Travel', science: 'Science', foodanddrink: 'Food & Drink',
  lifestyle: 'Lifestyle', autos: 'Autos', weather: 'Weather',
}

const CAT_COLORS = {
  sports:        'text-blue-800 border-blue-300',
  health:        'text-green-800 border-green-300',
  technology:    'text-indigo-800 border-indigo-300',
  politics:      'text-red-800 border-red-300',
  finance:       'text-amber-800 border-amber-300',
  entertainment: 'text-pink-800 border-pink-300',
  travel:        'text-cyan-800 border-cyan-300',
  science:       'text-orange-800 border-orange-300',
  foodanddrink:  'text-lime-800 border-lime-300',
  lifestyle:     'text-rose-800 border-rose-300',
  autos:         'text-slate-700 border-slate-300',
  weather:       'text-sky-800 border-sky-300',
}

const DEFAULT_CAT_COLOR = 'text-gray-700 border-gray-300'

export default function ArticleCard({ article, rank, alreadyRead, onRead, onOpen, featured = false }) {
  const { news_id, title, category, subcategory, abstract, score } = article
  const catLabel = CAT_DISPLAY[category] ?? category.toUpperCase()
  const catColor = CAT_COLORS[category] ?? DEFAULT_CAT_COLOR
  const pct = Math.round(score * 100)
  const excerpt = abstract
    ? abstract.slice(0, 180) + (abstract.length > 180 ? '…' : '')
    : ''

  return (
    <article
      className={`article-tile group cursor-pointer transition-all hover:shadow-md
                  ${alreadyRead ? 'opacity-60' : ''}
                  px-4 py-4`}
      onClick={() => onOpen && onOpen(article)}
    >
      {/* Section tag */}
      <div className={`inline-flex items-center gap-1.5 mb-2 border-b ${catColor} pb-0.5`}>
        <span className={`text-xs font-bold uppercase tracking-widest ${catColor.split(' ')[0]}`}>
          {catLabel}
        </span>
        {subcategory && (
          <span className="text-xs text-ink-light capitalize">· {subcategory}</span>
        )}
        {alreadyRead && (
          <span className="text-xs text-ink-light">· Read</span>
        )}
      </div>

      {/* Headline */}
      <h2 className="font-headline font-bold text-ink leading-snug mb-2 text-base md:text-lg
                     group-hover:underline decoration-1 underline-offset-2">
        {title}
      </h2>

      {/* Excerpt */}
      {excerpt && (
        <p className="text-xs text-ink-light leading-relaxed mb-3 flex-1">
          {excerpt}
        </p>
      )}

      {/* Footer: relevance + actions */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {/* Thin relevance bar */}
          <div className="w-16 h-0.5 bg-rule overflow-hidden">
            <div
              className="h-full bg-masthead transition-all duration-500"
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className="text-xs text-ink-light">{pct}% match</span>
        </div>

        <div className="flex items-center gap-2" onClick={e => e.stopPropagation()}>
          <button
            onClick={() => onOpen && onOpen(article)}
            className="text-xs text-masthead hover:underline underline-offset-2 font-medium"
          >
            Read →
          </button>
          {!alreadyRead && (
            <button
              onClick={() => onRead(news_id)}
              className="text-xs text-ink-light hover:text-ink border-l border-rule pl-2"
              title="Mark as read without opening"
            >
              ✓
            </button>
          )}
        </div>
      </div>
    </article>
  )
}
