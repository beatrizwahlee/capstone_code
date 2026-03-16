import { useState } from 'react'

const CAT_DISPLAY = {
  sports: 'Sports', news: 'News', health: 'Health',
  finance: 'Finance', entertainment: 'Entertain.', travel: 'Travel',
  foodanddrink: 'Food', lifestyle: 'Lifestyle', autos: 'Autos',
  weather: 'Weather', music: 'Music', movies: 'Movies',
  tv: 'TV', video: 'Video', kids: 'Kids',
}

const METRIC_TOOLTIPS = {
  Gini: 'Measures how concentrated your feed is across categories (0–1). 0 = perfectly balanced, 1 = all articles from one category. Lower is better — high Gini signals an echo chamber risk.',
  ILD: 'Intra-List Diversity: average pairwise difference between articles in your feed. Higher means articles are more distinct — your feed covers a wider range of ideas.',
  Coverage: 'Share of all available news categories that appear in your feed. Higher = more breadth. Low coverage means just one or two topics are dominating your recommendations.',
  Entropy: 'Shannon entropy of the category distribution. Higher = categories are spread more evenly. Lower = a few categories take up most of your feed.',
  Minority: 'Fraction of corpus-rare categories (Travel, Food, Music, Autos, Movies — each under 5% of MIND) present in your feed. Responds directly to the Fairness slider. Higher = better supply-side fairness.',
}

function MetricTile({ label, value, sub, good, onTooltip }) {
  const tooltip = METRIC_TOOLTIPS[label]
  return (
    <div
      className="text-center cursor-help"
      onMouseEnter={tooltip ? e => {
        const rect = e.currentTarget.getBoundingClientRect()
        onTooltip({ text: tooltip, x: rect.left + rect.width / 2, y: rect.top })
      } : undefined}
      onMouseLeave={tooltip ? () => onTooltip(null) : undefined}
    >
      <div className={`text-lg font-bold font-headline ${
        good === true ? 'text-green-700' : good === false ? 'text-red-600' : 'text-masthead'
      }`}>
        {value}
      </div>
      <div className="text-xs font-semibold uppercase tracking-wide text-ink mt-0.5 flex items-center justify-center gap-1">
        {label}
        {tooltip && <span className="text-ink-light/60 text-[10px]">ⓘ</span>}
      </div>
      <div className="text-xs text-ink-light">{sub}</div>
    </div>
  )
}

export default function MetricsDashboard({ metrics, categoryDist, historyCount, diversityPreference }) {
  const [activeTooltip, setActiveTooltip] = useState(null)

  if (!metrics) return null

  const { gini, ild, coverage, coverage_str, entropy, minority_coverage } = metrics
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
          onTooltip={setActiveTooltip}
        />
        <MetricTile
          label="ILD"
          value={ild?.toFixed(2) ?? '—'}
          sub={ild > 0.5 ? 'Diverse ✓' : 'Low'}
          good={ild != null ? ild > 0.4 : null}
          onTooltip={setActiveTooltip}
        />
        <MetricTile
          label="Coverage"
          value={coverage_str ?? `${Math.round((coverage ?? 0) * 100)}%`}
          sub={`${Math.round((coverage ?? 0) * 100)}% cats`}
          good={coverage != null ? coverage > 0.25 : null}
          onTooltip={setActiveTooltip}
        />
        <MetricTile
          label="Entropy"
          value={entropy?.toFixed(2) ?? '—'}
          sub={entropy > 2 ? 'Diverse' : entropy > 1 ? 'Balanced' : 'Concentrated'}
          good={entropy != null ? entropy > 1.5 : null}
          onTooltip={setActiveTooltip}
        />
        <MetricTile
          label="Minority"
          value={minority_coverage != null ? `${Math.round(minority_coverage * 100)}%` : '—'}
          sub={minority_coverage >= 0.6 ? 'Fair ✓' : minority_coverage >= 0.4 ? 'Partial' : 'Low'}
          good={minority_coverage != null ? minority_coverage >= 0.4 : null}
          onTooltip={setActiveTooltip}
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

      {/* Learned diversity preference */}
      {diversityPreference != null && (
        <div className="border-t border-rule pt-3">
          <p className="text-xs font-bold uppercase tracking-widest text-ink mb-1.5">
            Learned Preference
          </p>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs text-ink-light w-16 text-right">Focused</span>
            <div className="flex-1 bg-rule/30 h-1.5 overflow-hidden">
              <div
                className="h-full bg-masthead transition-all duration-700"
                style={{ width: `${diversityPreference * 100}%` }}
              />
            </div>
            <span className="text-xs text-ink-light w-16">Explorer</span>
          </div>
          <p className="text-xs text-ink-light text-center">
            {diversityPreference > 0.65 ? 'Explorer — you enjoy variety'
             : diversityPreference < 0.35 ? 'Specialist — you prefer depth'
             : 'Balanced reader'}
          </p>
        </div>
      )}

      {/* Fixed-position tooltip — escapes overflow clipping */}
      {activeTooltip && (
        <div
          style={{
            position: 'fixed',
            left: activeTooltip.x,
            top: activeTooltip.y,
            transform: 'translate(-50%, calc(-100% - 8px))',
            zIndex: 9999,
          }}
          className="w-56 bg-ink text-paper text-xs px-3 py-2 leading-relaxed pointer-events-none shadow-lg text-left"
        >
          {activeTooltip.text}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-ink" />
        </div>
      )}
    </div>
  )
}
