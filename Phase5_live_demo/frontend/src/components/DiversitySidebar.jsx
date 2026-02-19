const METHOD_META = {
  baseline:    { label: 'Baseline',    color: 'bg-gray-100 text-gray-700',    desc: 'Accuracy only'            },
  mmr:         { label: 'MMR',         color: 'bg-blue-100 text-blue-800',    desc: 'Balanced diversity'        },
  calibrated:  { label: 'Calibrated',  color: 'bg-green-100 text-green-800',  desc: 'Matches your history'      },
  serendipity: { label: 'Serendipity', color: 'bg-purple-100 text-purple-800',desc: 'Explores new topics'       },
  xquad:       { label: 'xQuAD',       color: 'bg-orange-100 text-orange-800',desc: 'Fair category coverage'    },
}

const SUB_SLIDER_META = [
  {
    key: 'calibration',
    label: 'Calibration',
    tooltip: 'Match my reading history',
    icon: '⚖️',
  },
  {
    key: 'serendipity',
    label: 'Serendipity',
    tooltip: 'Surprise me with new topics',
    icon: '✨',
  },
  {
    key: 'fairness',
    label: 'Fairness',
    tooltip: 'Proportional category coverage',
    icon: '⚡',
  },
]

function Slider({ value, onChange, min = 0, max = 1, step = 0.05, label, left, right, color = 'bg-blue-600' }) {
  return (
    <div>
      {label && <div className="text-xs font-medium text-gray-700 mb-1">{label}</div>}
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          className="slider w-full"
          style={{
            background: `linear-gradient(to right, #2563eb ${value * 100}%, #e5e7eb ${value * 100}%)`,
          }}
        />
      </div>
      {(left || right) && (
        <div className="flex justify-between text-xs text-gray-400 mt-1">
          <span>{left}</span>
          <span>{right}</span>
        </div>
      )}
    </div>
  )
}

export default function DiversitySidebar({ sliders, activeMethod, onChange }) {
  const method = METHOD_META[activeMethod] ?? METHOD_META.mmr
  const showSubSliders = sliders.main_diversity >= 0.1

  function update(key, val) {
    onChange({ ...sliders, [key]: val })
  }

  return (
    <div className="card p-4 space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-900">🎛️ Controls</h2>
        <span className={`badge text-xs ${method.color}`} title={method.desc}>
          {method.label}
        </span>
      </div>

      {/* Main accuracy ↔ diversity slider */}
      <div>
        <Slider
          value={sliders.main_diversity}
          onChange={val => update('main_diversity', val)}
          left="Accuracy"
          right="Diversity"
        />
        <div className="text-xs text-center text-gray-400 mt-1">
          {sliders.main_diversity < 0.15
            ? 'Accuracy-focused'
            : sliders.main_diversity > 0.75
            ? 'Maximum diversity'
            : 'Balanced'}
        </div>
      </div>

      {/* Sub-sliders */}
      {showSubSliders && (
        <div className="space-y-4 pt-2 border-t border-gray-100">
          <p className="text-xs text-gray-400 font-medium uppercase tracking-wide">Diversity Mode</p>
          {SUB_SLIDER_META.map(meta => (
            <div key={meta.key}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium text-gray-700">
                  {meta.icon} {meta.label}
                </span>
                <span className="text-xs text-gray-400" title={meta.tooltip}>
                  {(sliders[meta.key] * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={sliders[meta.key]}
                onChange={e => update(meta.key, parseFloat(e.target.value))}
                className="slider w-full"
                title={meta.tooltip}
                style={{
                  background: `linear-gradient(to right, #2563eb ${sliders[meta.key] * 100}%, #e5e7eb ${sliders[meta.key] * 100}%)`,
                }}
              />
              <p className="text-xs text-gray-400 mt-0.5">{meta.tooltip}</p>
            </div>
          ))}
        </div>
      )}

      {/* Active algorithm explanation */}
      <div className="bg-gray-50 rounded-lg p-3 text-xs text-gray-500 leading-relaxed">
        <span className="font-medium text-gray-700">{method.label}:</span> {method.desc}
      </div>
    </div>
  )
}
