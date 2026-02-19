const CATEGORY_STYLES = {
  sports:        { bg: 'bg-blue-100',   text: 'text-blue-800'   },
  health:        { bg: 'bg-green-100',  text: 'text-green-800'  },
  politics:      { bg: 'bg-red-100',    text: 'text-red-800'    },
  technology:    { bg: 'bg-indigo-100', text: 'text-indigo-800' },
  finance:       { bg: 'bg-amber-100',  text: 'text-amber-800'  },
  entertainment: { bg: 'bg-pink-100',   text: 'text-pink-800'   },
  travel:        { bg: 'bg-cyan-100',   text: 'text-cyan-800'   },
  science:       { bg: 'bg-orange-100', text: 'text-orange-800' },
  foodanddrink:  { bg: 'bg-lime-100',   text: 'text-lime-800'   },
  lifestyle:     { bg: 'bg-rose-100',   text: 'text-rose-800'   },
  autos:         { bg: 'bg-slate-100',  text: 'text-slate-800'  },
  weather:       { bg: 'bg-sky-100',    text: 'text-sky-800'    },
}

const DEFAULT_STYLE = { bg: 'bg-gray-100', text: 'text-gray-700' }

const SCORE_COLORS = [
  { min: 0.8, bar: 'from-emerald-400 to-emerald-600' },
  { min: 0.6, bar: 'from-blue-400 to-blue-600'       },
  { min: 0.4, bar: 'from-amber-400 to-amber-600'     },
  { min: 0.0, bar: 'from-gray-300 to-gray-400'       },
]

function scoreMeta(score) {
  const entry = SCORE_COLORS.find(e => score >= e.min) ?? SCORE_COLORS[SCORE_COLORS.length - 1]
  return entry.bar
}

export default function ArticleCard({ article, rank, alreadyRead, onRead }) {
  const { news_id, title, category, subcategory, abstract, score } = article
  const catStyle = CATEGORY_STYLES[category] ?? DEFAULT_STYLE
  const excerpt = abstract ? abstract.slice(0, 180) + (abstract.length > 180 ? '…' : '') : ''
  const barColor = scoreMeta(score)
  const pct = Math.round(score * 100)

  return (
    <article className={`card p-5 transition-all duration-200 hover:shadow-md ${alreadyRead ? 'opacity-75' : ''}`}>
      {/* Top row: badge + score */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className={`badge ${catStyle.bg} ${catStyle.text}`}>
            {category}
          </span>
          {subcategory && (
            <span className="text-xs text-gray-400 capitalize">{subcategory}</span>
          )}
        </div>
        <div className="flex items-center gap-1.5 text-xs text-gray-500">
          <span className="text-amber-500">★</span>
          <span className="font-medium">{score.toFixed(2)}</span>
        </div>
      </div>

      {/* Score bar */}
      <div className="h-1 w-full bg-gray-100 rounded-full mb-3 overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${barColor} transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Title */}
      <h2 className="text-base font-semibold text-gray-900 leading-snug mb-2 line-clamp-2">
        {title}
      </h2>

      {/* Excerpt */}
      {excerpt && (
        <p className="text-sm text-gray-500 leading-relaxed mb-4 line-clamp-3">
          {excerpt}
        </p>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-300">#{rank}</span>
        <button
          onClick={() => onRead(news_id)}
          disabled={alreadyRead}
          className={`text-sm font-medium px-4 py-1.5 rounded-lg transition-colors ${
            alreadyRead
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800'
          }`}
        >
          {alreadyRead ? '✓ Read' : 'Read →'}
        </button>
      </div>
    </article>
  )
}
