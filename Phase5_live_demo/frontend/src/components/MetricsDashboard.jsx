const CAT_DISPLAY = {
  sports: 'Sports', health: 'Health', technology: 'Tech',
  politics: 'Politics', finance: 'Finance', entertainment: 'Entertain.',
  travel: 'Travel', science: 'Science', foodanddrink: 'Food',
  lifestyle: 'Lifestyle', autos: 'Autos', weather: 'Weather',
}

const METRIC_TOOLTIPS = {
  Gini: 'Measures how concentrated your feed is across categories (0–1). 0 = perfectly balanced, 1 = all articles from one category. Lower is better — high Gini signals an echo chamber risk.',
  ILD: 'Intra-List Diversity: average semantic distance between articles in your feed. Higher means articles are more distinct in content and style — your feed covers a wider range of ideas.',
  Coverage: 'Share of all available news categories that appear in your feed. Higher = more breadth. Low coverage means just one or two topics are dominating your recommendations.',
  Entropy: 'Shannon entropy of the category distribution. Higher = categories are spread more evenly. Lower = a few categories take up most of your feed.',
}

function MetricTile({ label, value, sub, good }) {
  const tooltip = METRIC_TOOLTIPS[label]
  return (
    <div className="text-center relative group cursor-help">
      {/* Tooltip */}
      {tooltip && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-52 bg-ink text-paper
                        text-xs px-3 py-2 leading-relaxed opacity-0 group-hover:opacity-100
                        transition-opacity duration-150 pointer-events-none z-20 text-left shadow-lg">
          {tooltip}
          {/* Arrow */}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-ink" />
        </div>
      )}
      <div className={`text-lg font-bold font-headline ${
        good === true ? 'text-green-700' : good === false ? 'text-red-600' : 'text-masthead'
      }`}>
        {value}
      </div>
      <div className="text-xs font-semibold uppercase tracking-wide text-ink mt-0.5 flex items-center justify-center gap-1">
        {label}
        <span className="text-ink-light/60 text-[10px]">ⓘ</span>
      </div>
      <div className="text-xs text-ink-light">{sub}</div>
    </div>
  )
}

export default function MetricsDashboard({ metrics, categoryDist, historyCount }) {
  if (!metrics) return null

  const { gini, ild, coverage, coverage_str, entropy } = metrics
  const entries = Object.entries(categoryDist).sort((a, b) => b[1] - a[1])
  const maxCount = entries[0]?.[1] ?? 1

  return (
    <div className="border border-rule bg-white p-4 space-y-4">
      <div>
        <div className="section-rule mb-2" />
        <h2 className="text-xs font-bold uppercase tracking-widest text-ink">Diversity Metrics</h2>
      </div>

      {/* 2×2 metrics grid */}
      <div className="grid grid-cols-2 gap-3 border border-rule p-3">
        <MetricTile
          label="Gini"
          value={gini?.toFixed(2) ?? '—'}
          sub={gini < 0.35 ? 'Balanced ✓' : gini < 0.6 ? 'Moderate' : 'Echo risk ⚠'}
          good={gini != null ? gini < 0.5 : null}
        />
        <MetricTile
          label="ILD"
          value={ild?.toFixed(2) ?? '—'}
          sub={ild > 0.5 ? 'Diverse ✓' : 'Low'}
          good={ild != null ? ild > 0.4 : null}
        />
        <MetricTile
          label="Coverage"
          value={coverage_str ?? `${Math.round((coverage ?? 0) * 100)}%`}
          sub={`${Math.round((coverage ?? 0) * 100)}% cats`}
          good={coverage != null ? coverage > 0.25 : null}
        />
        <MetricTile
          label="Entropy"
          value={entropy?.toFixed(2) ?? '—'}
          sub={entropy > 2 ? 'Diverse' : entropy > 1 ? 'Balanced' : 'Concentrated'}
          good={entropy != null ? entropy > 1.5 : null}
        />
      </div>

      {/* Category distribution */}
      {entries.length > 0 && (
        <div className="border-t border-rule pt-3">
          <p className="text-xs font-bold uppercase tracking-widest text-ink mb-2">
            Category Mix
          </p>
          <div className="space-y-1.5">
            {entries.map(([cat, count]) => (
              <div key={cat} className="flex items-center gap-2">
                <span className="text-xs text-ink-light w-16 truncate text-right shrink-0">
                  {CAT_DISPLAY[cat] ?? cat}
                </span>
                <div className="flex-1 bg-rule/30 h-1.5 overflow-hidden">
                  <div
                    className="h-full bg-masthead transition-all duration-500"
                    style={{ width: `${(count / maxCount) * 100}%` }}
                  />
                </div>
                <span className="text-xs text-ink-light w-3 text-right">{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* History counter */}
      <div className="border-t border-rule pt-3">
        <div className="flex items-center justify-between text-xs">
          <span className="text-ink-light uppercase tracking-wide">Articles read</span>
          <span className="font-bold text-ink">{historyCount}</span>
        </div>
        {historyCount > 0 && (
          <div className="mt-1 h-0.5 bg-rule overflow-hidden">
            <div
              className="h-full bg-masthead transition-all duration-500"
              style={{ width: `${Math.min(100, historyCount * 10)}%` }}
            />
          </div>
        )}
      </div>
    </div>
  )
}
