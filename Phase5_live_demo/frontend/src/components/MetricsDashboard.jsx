function MetricTile({ label, value, sub, color = 'text-blue-600' }) {
  return (
    <div className="bg-gray-50 rounded-lg p-3 text-center">
      <div className={`text-lg font-bold ${color}`}>{value}</div>
      <div className="text-xs font-medium text-gray-600 mt-0.5">{label}</div>
      <div className="text-xs text-gray-400 mt-0.5">{sub}</div>
    </div>
  )
}

function giniColor(g) {
  if (g < 0.3) return 'text-emerald-600'
  if (g < 0.6) return 'text-amber-500'
  return 'text-red-500'
}

function ildColor(v) {
  if (v > 0.6) return 'text-emerald-600'
  if (v > 0.3) return 'text-blue-600'
  return 'text-amber-500'
}

function entropyLabel(e) {
  if (e > 2.5) return 'Very diverse'
  if (e > 1.5) return 'Balanced'
  if (e > 0.8) return 'Moderate'
  return 'Concentrated'
}

export default function MetricsDashboard({ metrics, categoryDist, historyCount }) {
  if (!metrics) return null

  const { gini, ild, coverage, coverage_str, entropy } = metrics
  const entries = Object.entries(categoryDist).sort((a, b) => b[1] - a[1])
  const maxCount = entries.length > 0 ? entries[0][1] : 1

  return (
    <div className="card p-4 space-y-4">
      <h2 className="text-sm font-semibold text-gray-900">📊 Diversity Metrics</h2>

      {/* 4 tiles */}
      <div className="grid grid-cols-2 gap-2">
        <MetricTile
          label="Gini"
          value={gini?.toFixed(2) ?? '—'}
          sub={gini < 0.35 ? 'Balanced ✓' : gini < 0.6 ? 'Moderate' : 'Echo risk ⚠️'}
          color={giniColor(gini ?? 0)}
        />
        <MetricTile
          label="ILD"
          value={ild?.toFixed(2) ?? '—'}
          sub={ild > 0.5 ? 'Diverse ✓' : 'Low diversity'}
          color={ildColor(ild ?? 0)}
        />
        <MetricTile
          label="Coverage"
          value={coverage_str ?? `${Math.round((coverage ?? 0) * 100)}%`}
          sub={`${Math.round((coverage ?? 0) * 100)}% categories`}
          color="text-indigo-600"
        />
        <MetricTile
          label="Entropy"
          value={entropy?.toFixed(2) ?? '—'}
          sub={entropyLabel(entropy ?? 0)}
          color="text-purple-600"
        />
      </div>

      {/* Category distribution bar chart */}
      {entries.length > 0 && (
        <div className="pt-2 border-t border-gray-100">
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3">
            Category Distribution
          </p>
          <div className="space-y-2">
            {entries.map(([cat, count]) => (
              <div key={cat} className="flex items-center gap-2">
                <span className="text-xs text-gray-500 capitalize w-20 truncate text-right">
                  {cat}
                </span>
                <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-blue-400 to-blue-600 transition-all duration-500"
                    style={{ width: `${(count / maxCount) * 100}%` }}
                  />
                </div>
                <span className="text-xs text-gray-400 w-4 text-right">{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* History count */}
      <div className="pt-2 border-t border-gray-100">
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">📚 Articles read</span>
          <span className="font-semibold text-gray-800">{historyCount}</span>
        </div>
        {historyCount > 0 && (
          <div className="mt-1 h-1 bg-gray-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-green-400 to-green-600 rounded-full transition-all duration-500"
              style={{ width: `${Math.min(100, historyCount * 10)}%` }}
            />
          </div>
        )}
      </div>
    </div>
  )
}
