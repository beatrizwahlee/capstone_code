
const SUB_SLIDERS = [
  { key: 'diversity',   label: 'Diversity',   tooltip: 'Reduce embedding-similar articles clustering together' },
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

export default function DiversitySidebar({ sliders, onChange }) {
  const showSub = sliders.main_diversity > 0

  function update(key, val) {
    onChange({ ...sliders, [key]: val })
  }

  return (
    <div className="border border-rule bg-white p-4 space-y-4">
      {/* Header */}
      <div>
        <div className="section-rule mb-2" />
        <h2 className="text-xs font-bold uppercase tracking-widest text-ink">Controls</h2>
      </div>

      {/* Main accuracy ↔ explore slider */}
      <div>
        <div className="flex justify-between text-xs text-ink-light mb-1.5">
          <span className="font-medium">Accuracy</span>
          <span className="font-medium">Explore</span>
        </div>
        <Slider value={sliders.main_diversity} onChange={val => update('main_diversity', val)} />
        <p className="text-xs text-ink-light mt-1 text-center">
          {sliders.main_diversity < 0.05
            ? 'Baseline — pure relevance ranking'
            : sliders.main_diversity > 0.75
            ? 'Maximum exploration'
            : 'Balanced'}
        </p>
      </div>

      {/* Four pillar sub-sliders — visible only when exploring */}
      {showSub && (
        <div className="space-y-3 pt-2 border-t border-rule">
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
    </div>
  )
}
