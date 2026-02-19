const METHOD_META = {
  baseline:  { label: 'Baseline',  desc: 'Pure relevance — no diversity adjustment' },
  composite: { label: 'Composite', desc: 'All four diversity dimensions active simultaneously: embedding variety, calibration toward your interests, serendipitous exploration, and popularity fairness — each weighted by the sliders.' },
  // legacy method names kept for fallback
  mmr:         { label: 'MMR',         desc: 'Balanced: relevance + category diversity'        },
  calibrated:  { label: 'Calibrated',  desc: 'Matches your reading history distribution'       },
  serendipity: { label: 'Serendipity', desc: 'Explores new topics beyond your usual interests' },
  xquad:       { label: 'xQuAD',       desc: 'Fair proportional category coverage'             },
}

const SUB_SLIDERS = [
  { key: 'calibration', label: 'Calibration', tooltip: 'Match my reading history'         },
  { key: 'serendipity', label: 'Serendipity', tooltip: 'Surprise me with new topics'      },
  { key: 'fairness',    label: 'Fairness',    tooltip: 'Proportional category coverage'   },
]

function Slider({ value, onChange }) {
  return (
    <input
      type="range"
      min={0} max={1} step={0.05}
      value={value}
      onChange={e => onChange(parseFloat(e.target.value))}
      className="slider w-full"
      style={{
        background: `linear-gradient(to right, #1a3a5c ${value * 100}%, #c9b99a ${value * 100}%)`,
      }}
    />
  )
}

export default function DiversitySidebar({ sliders, activeMethod, onChange }) {
  const method = METHOD_META[activeMethod] ?? METHOD_META.mmr
  const showSub = sliders.main_diversity >= 0.1

  function update(key, val) {
    onChange({ ...sliders, [key]: val })
  }

  return (
    <div className="border border-rule bg-white p-4 space-y-4">
      {/* Header */}
      <div>
        <div className="section-rule mb-2" />
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-bold uppercase tracking-widest text-ink">Controls</h2>
          <span className="text-xs font-bold uppercase tracking-wider text-masthead border border-masthead px-1.5 py-0.5">
            {method.label}
          </span>
        </div>
      </div>

      {/* Main accuracy ↔ diversity slider */}
      <div>
        <div className="flex justify-between text-xs text-ink-light mb-1.5">
          <span className="font-medium">Accuracy</span>
          <span className="font-medium">Diversity</span>
        </div>
        <Slider value={sliders.main_diversity} onChange={val => update('main_diversity', val)} />
        <p className="text-xs text-ink-light mt-1 text-center">
          {sliders.main_diversity < 0.05
            ? 'Baseline — pure relevance ranking'
            : sliders.main_diversity > 0.75
            ? 'Maximum diversity'
            : 'Balanced'}
        </p>
      </div>

      {/* Sub-sliders */}
      {showSub && (
        <div className="space-y-3 pt-2 border-t border-rule">
          <p className="text-xs uppercase tracking-widest text-ink-light font-medium">Diversity Mode</p>
          {SUB_SLIDERS.map(meta => (
            <div key={meta.key}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium text-ink">{meta.label}</span>
                <span className="text-xs text-ink-light">
                  {(sliders[meta.key] * 100).toFixed(0)}%
                </span>
              </div>
              <Slider value={sliders[meta.key]} onChange={val => update(meta.key, val)} />
              <p className="text-xs text-ink-light mt-0.5">{meta.tooltip}</p>
            </div>
          ))}
        </div>
      )}

      {/* Algorithm description */}
      <div className="border-t border-rule pt-3">
        <p className="text-xs text-ink-light leading-relaxed">
          <span className="font-semibold text-ink">{method.label}:</span> {method.desc}
        </p>
      </div>
    </div>
  )
}
