import { useState } from 'react'

const SUB_SLIDERS = [
  { key: 'diversity',   label: 'Diversity',   tooltip: 'Reduce embedding-similar articles clustering together' },
  { key: 'calibration', label: 'Calibration', tooltip: 'Match my reading history'         },
  { key: 'serendipity', label: 'Serendipity', tooltip: 'Surprise me with new topics'      },
  { key: 'fairness',    label: 'Fairness',    tooltip: 'Balance coverage across all topics in the corpus, not just your history'   },
]

function Slider({ value, onChange, disabled }) {
  return (
    <input
      type="range"
      min={0} max={1} step={0.05}
      value={value}
      onChange={e => onChange(parseFloat(e.target.value))}
      disabled={disabled}
      className="slider w-full"
      style={{
        background: `linear-gradient(to right, #1a3a5c ${value * 100}%, #c9b99a ${value * 100}%)`,
        opacity: disabled ? 0.4 : 1,
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
    />
  )
}

function LockIcon({ locked }) {
  return locked ? (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="w-3 h-3">
      <path fillRule="evenodd" d="M8 1a3 3 0 0 0-3 3v1H4a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V6a1 1 0 0 0-1-1h-1V4a3 3 0 0 0-3-3Zm0 9a1 1 0 1 0 0-2 1 1 0 0 0 0 2ZM7 4a1 1 0 1 1 2 0v1H7V4Z" clipRule="evenodd" />
    </svg>
  ) : (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="w-3 h-3">
      <path fillRule="evenodd" d="M5 4a3 3 0 0 1 6 0v1h1a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h1V4Zm5 1V4a2 2 0 1 0-4 0v1h4Zm-2 5a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z" clipRule="evenodd" />
    </svg>
  )
}

const SUB_KEYS = SUB_SLIDERS.map(s => s.key)
const BUDGET = 1.0

export default function DiversitySidebar({ sliders, onChange }) {
  const showSub = sliders.main_diversity > 0
  const [locked, setLocked] = useState({ diversity: false, calibration: false, serendipity: false, fairness: false })

  function toggleLock(key) {
    // Don't allow locking all four — at least one must be free
    const newLocked = { ...locked, [key]: !locked[key] }
    const freeCount = SUB_KEYS.filter(k => !newLocked[k]).length
    if (freeCount === 0) return
    setLocked(newLocked)
  }

  function update(key, val) {
    // Main slider — independent
    if (!SUB_KEYS.includes(key)) {
      onChange({ ...sliders, [key]: val })
      return
    }

    // Clamp so locked pillars always keep their share
    const lockedSum = SUB_KEYS.filter(k => k !== key && locked[k]).reduce((s, k) => s + sliders[k], 0)
    const newVal = Math.min(Math.max(val, 0), BUDGET - lockedSum)

    // Redistribute remaining budget among unlocked keys (excluding the one being moved)
    const freeKeys = SUB_KEYS.filter(k => k !== key && !locked[k])
    const freeSum  = freeKeys.reduce((s, k) => s + sliders[k], 0)
    const remaining = BUDGET - newVal - lockedSum

    const updated = { ...sliders, [key]: newVal }
    freeKeys.forEach(k => {
      updated[k] = freeSum > 0
        ? remaining * (sliders[k] / freeSum)
        : remaining / freeKeys.length
    })
    onChange(updated)
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
          <p className="text-xs text-ink-light italic">Lock a pillar to pin its value; others absorb the change.</p>
          {SUB_SLIDERS.map(meta => (
            <div key={meta.key}>
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => toggleLock(meta.key)}
                    title={locked[meta.key] ? 'Unlock' : 'Lock this value'}
                    className={`flex items-center justify-center w-5 h-5 rounded border transition-colors ${
                      locked[meta.key]
                        ? 'bg-ink text-paper border-ink'
                        : 'bg-white text-ink-light border-rule hover:border-ink'
                    }`}
                  >
                    <LockIcon locked={locked[meta.key]} />
                  </button>
                  <span className="text-xs font-medium text-ink">{meta.label}</span>
                </div>
                <span className="text-xs text-ink-light">
                  {(sliders[meta.key] * 100).toFixed(0)}%
                </span>
              </div>
              <Slider
                value={sliders[meta.key]}
                onChange={val => update(meta.key, val)}
                disabled={locked[meta.key]}
              />
              <p className="text-xs text-ink-light mt-0.5">{meta.tooltip}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
