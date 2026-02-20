'use client'

import { cn } from '@/lib/utils'
import type { ModelType, ModelTypeInfo } from '@/types/api'

// ── Static model catalogue ────────────────────────────────────────────────────

export const MODEL_CATALOGUE: ModelTypeInfo[] = [
  {
    id: 'nilut',
    label: 'NILUT',
    tagline: 'Neural Implicit LUT — compact, blendable',
    description:
      'A tiny MLP (<0.25 MB) that learns a continuous color mapping. Supports smooth multi-style blending at inference time without retraining.',
    features: [
      'Up to 5 styles blended with a single model',
      'Model size under 0.25 MB — edge-device ready',
      'Smooth interpolation between learned styles',
      'Standard .cube export via LUT baking',
    ],
    speed: 'fast',
    modelSize: '< 0.25 MB',
    isAdaptive: false,
    supportsBlending: true,
    useCases: [
      'Multi-mood filter presets',
      'Mobile and edge deployments',
      'Creative style mixing workflows',
    ],
    badge: 'Recommended',
    badgeVariant: 'recommended',
  },
  {
    id: 'lut3d',
    label: 'Image-Adaptive 3D LUT',
    tagline: 'Content-aware LUT — real-time 4K capable',
    description:
      'A lightweight CNN predicts per-image blending weights across a set of basis LUTs. The result adapts to each photo\'s content, producing more accurate color grading than a fixed LUT.',
    features: [
      'Content-aware: adapts to each image automatically',
      'Real-time 4K inference (< 2 ms on GPU)',
      'Standard 3D LUT output compatible with Photoshop & Premiere',
      'Paired and unpaired training support',
    ],
    speed: 'fast',
    modelSize: '< 5 MB',
    isAdaptive: true,
    supportsBlending: false,
    useCases: [
      'Professional photo editing workflows',
      'Batch processing of large image sets',
      'When per-image adaptation matters most',
    ],
    badge: 'Adaptive',
    badgeVariant: 'fast',
  },
]

const BADGE_STYLES: Record<ModelTypeInfo['badgeVariant'], string> = {
  recommended:
    'bg-brand-100 dark:bg-brand-900/40 text-brand-700 dark:text-brand-300',
  fast: 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300',
  quality:
    'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300',
}

const SPEED_LABEL: Record<ModelTypeInfo['speed'], string> = {
  fast: 'Fast',
  medium: 'Medium',
  slow: 'Slow',
}

const SPEED_COLOR: Record<ModelTypeInfo['speed'], string> = {
  fast: 'text-green-600 dark:text-green-400',
  medium: 'text-yellow-600 dark:text-yellow-400',
  slow: 'text-red-500 dark:text-red-400',
}

// ── Comparison table ──────────────────────────────────────────────────────────

const COMPARISON_ROWS: Array<{
  label: string
  nilut: string
  lut3d: string
}> = [
  { label: 'Model size', nilut: '< 0.25 MB', lut3d: '< 5 MB' },
  { label: 'Inference speed', nilut: 'Fast', lut3d: 'Very fast (< 2 ms)' },
  { label: 'Multi-style blending', nilut: 'Yes (up to 5)', lut3d: 'No' },
  { label: 'Content-adaptive', nilut: 'No', lut3d: 'Yes' },
  { label: '.cube export', nilut: 'Yes', lut3d: 'Yes' },
  { label: 'Edge deployable', nilut: 'Yes', lut3d: 'Yes' },
]

// ── Main component ────────────────────────────────────────────────────────────

interface ModelSelectorProps {
  value: ModelType
  onChange: (type: ModelType) => void
  className?: string
}

export function ModelSelector({ value, onChange, className }: ModelSelectorProps) {
  const selected = MODEL_CATALOGUE.find((m) => m.id === value) ?? MODEL_CATALOGUE[0]

  return (
    <div className={cn('space-y-4', className)}>
      {/* Card grid */}
      <div
        className="grid grid-cols-1 sm:grid-cols-2 gap-3"
        role="radiogroup"
        aria-label="AI model selection"
      >
        {MODEL_CATALOGUE.map((model) => {
          const isSelected = value === model.id
          return (
            <button
              key={model.id}
              type="button"
              role="radio"
              aria-checked={isSelected}
              onClick={() => onChange(model.id)}
              className={cn(
                'relative text-left rounded-2xl border-2 p-4 transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500',
                isSelected
                  ? 'border-brand-500 bg-brand-50 dark:bg-brand-950/20 shadow-sm'
                  : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 hover:border-gray-300 dark:hover:border-gray-600'
              )}
            >
              {/* Selected indicator */}
              <span
                className={cn(
                  'absolute top-3 right-3 h-5 w-5 rounded-full border-2 flex items-center justify-center transition-colors',
                  isSelected
                    ? 'border-brand-500 bg-brand-500'
                    : 'border-gray-300 dark:border-gray-600 bg-transparent'
                )}
                aria-hidden="true"
              >
                {isSelected && (
                  <svg className="h-3 w-3 text-white" viewBox="0 0 20 20" fill="currentColor">
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                )}
              </span>

              {/* Header */}
              <div className="pr-8 space-y-1 mb-3">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-semibold text-gray-900 dark:text-white">{model.label}</span>
                  <span
                    className={cn(
                      'rounded-full px-2 py-0.5 text-xs font-medium',
                      BADGE_STYLES[model.badgeVariant]
                    )}
                  >
                    {model.badge}
                  </span>
                  {model.isAdaptive && (
                    <span className="rounded-full bg-emerald-100 dark:bg-emerald-900/40 px-2 py-0.5 text-xs font-medium text-emerald-700 dark:text-emerald-300">
                      Adaptive
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400">{model.tagline}</p>
              </div>

              {/* Key stats */}
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
                <div>
                  <dt className="text-gray-400 dark:text-gray-500">Speed</dt>
                  <dd className={cn('font-medium', SPEED_COLOR[model.speed])}>
                    {SPEED_LABEL[model.speed]}
                  </dd>
                </div>
                <div>
                  <dt className="text-gray-400 dark:text-gray-500">Model size</dt>
                  <dd className="font-medium text-gray-700 dark:text-gray-300">{model.modelSize}</dd>
                </div>
                <div>
                  <dt className="text-gray-400 dark:text-gray-500">Style blending</dt>
                  <dd
                    className={cn(
                      'font-medium',
                      model.supportsBlending
                        ? 'text-green-600 dark:text-green-400'
                        : 'text-gray-400 dark:text-gray-500'
                    )}
                  >
                    {model.supportsBlending ? 'Supported' : 'Not supported'}
                  </dd>
                </div>
                <div>
                  <dt className="text-gray-400 dark:text-gray-500">Content-adaptive</dt>
                  <dd
                    className={cn(
                      'font-medium',
                      model.isAdaptive
                        ? 'text-green-600 dark:text-green-400'
                        : 'text-gray-400 dark:text-gray-500'
                    )}
                  >
                    {model.isAdaptive ? 'Yes' : 'No'}
                  </dd>
                </div>
              </dl>
            </button>
          )
        })}
      </div>

      {/* Expanded detail panel for the selected model */}
      <div
        className="rounded-2xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-5 space-y-4 animate-fade-in"
        aria-live="polite"
        aria-label={`Details for ${selected.label}`}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white">{selected.label}</h3>
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">{selected.description}</p>
          </div>
        </div>

        {/* Feature list */}
        <ul className="space-y-1.5">
          {selected.features.map((f) => (
            <li key={f} className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300">
              <svg
                className="mt-0.5 h-4 w-4 shrink-0 text-brand-500"
                viewBox="0 0 20 20"
                fill="currentColor"
                aria-hidden="true"
              >
                <path
                  fillRule="evenodd"
                  d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                  clipRule="evenodd"
                />
              </svg>
              {f}
            </li>
          ))}
        </ul>

        {/* Use cases */}
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-1.5">
            Best for
          </p>
          <div className="flex flex-wrap gap-2">
            {selected.useCases.map((uc) => (
              <span
                key={uc}
                className="rounded-full bg-gray-100 dark:bg-gray-800 px-2.5 py-1 text-xs text-gray-600 dark:text-gray-400"
              >
                {uc}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Comparison table */}
      <details className="group rounded-2xl border border-gray-200 dark:border-gray-700 overflow-hidden">
        <summary className="flex cursor-pointer items-center justify-between px-5 py-3 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors list-none">
          Compare models side by side
          <svg
            className="h-4 w-4 text-gray-400 transition-transform group-open:rotate-180"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            aria-hidden="true"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </summary>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
                <th className="px-5 py-2.5 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400 w-1/3">
                  Feature
                </th>
                <th className="px-4 py-2.5 text-center text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
                  NILUT
                </th>
                <th className="px-4 py-2.5 text-center text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
                  3D LUT Adaptive
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
              {COMPARISON_ROWS.map((row) => (
                <tr key={row.label} className="bg-white dark:bg-gray-900">
                  <td className="px-5 py-2.5 text-gray-600 dark:text-gray-400">{row.label}</td>
                  <td
                    className={cn(
                      'px-4 py-2.5 text-center font-medium',
                      value === 'nilut'
                        ? 'text-brand-600 dark:text-brand-400'
                        : 'text-gray-700 dark:text-gray-300'
                    )}
                  >
                    {row.nilut}
                  </td>
                  <td
                    className={cn(
                      'px-4 py-2.5 text-center font-medium',
                      value === 'lut3d'
                        ? 'text-brand-600 dark:text-brand-400'
                        : 'text-gray-700 dark:text-gray-300'
                    )}
                  >
                    {row.lut3d}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  )
}
