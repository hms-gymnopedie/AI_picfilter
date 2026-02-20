'use client'

import { useEffect, useRef, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { connectProgressSSE } from '@/lib/api'
import { formatDuration } from '@/lib/utils'
import type { ProgressEvent, ColorDistribution, StyleFeatures } from '@/types/filter'

interface LearningProgressProps {
  styleId: string
  styleName: string
  onComplete: (deltaE: number, trainingTimeSeconds: number) => void
  onError: (code: string, message: string) => void
}

const FEATURE_LABELS: Record<string, string> = {
  warmTones: 'Warm tones',
  contrast: 'Contrast',
  saturation: 'Saturation',
  shadowDetail: 'Shadow detail',
}

const LEVEL_COLOR: Record<string, string> = {
  low: 'text-blue-500',
  medium: 'text-yellow-500',
  high: 'text-orange-500',
}

export function LearningProgress({ styleId, styleName, onComplete, onError }: LearningProgressProps) {
  const [progress, setProgress] = useState(0)
  const [phase, setPhase] = useState<string>('Initializing...')
  const [message, setMessage] = useState<string>('')
  const [elapsed, setElapsed] = useState(0)
  const [deltaE, setDeltaE] = useState<number | null>(null)
  const [epoch, setEpoch] = useState<{ current: number; total: number } | null>(null)
  const [distribution, setDistribution] = useState<ColorDistribution | null>(null)
  const [features, setFeatures] = useState<StyleFeatures | null>(null)
  const [log, setLog] = useState<string[]>([])
  const [connectionState, setConnectionState] = useState<'connecting' | 'connected' | 'reconnecting' | 'done' | 'error'>('connecting')

  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const disconnect = connectProgressSSE(styleId, {
      onOpen: () => setConnectionState('connected'),
      onReconnecting: () => setConnectionState('reconnecting'),

      onProgress: (data: ProgressEvent) => {
        setProgress(Math.round(data.progress * 100))
        setPhase(data.phase)
        setMessage(data.message)
        setElapsed(data.elapsed)
        if (data.deltaE != null) setDeltaE(data.deltaE)
        if (data.epoch != null && data.totalEpochs != null) {
          setEpoch({ current: data.epoch, total: data.totalEpochs })
        }
        setLog((prev) => [...prev.slice(-99), `> ${data.message}`])
      },

      onMetrics: (data: ColorDistribution) => {
        setDistribution(data)
      },

      onFeatures: (data: StyleFeatures) => {
        setFeatures(data)
      },

      onComplete: (data) => {
        setProgress(100)
        setPhase('complete')
        setConnectionState('done')
        setLog((prev) => [...prev, '> Training complete!'])
        onComplete(data.deltaE, data.trainingTimeSeconds)
      },

      onError: (data) => {
        setConnectionState('error')
        setLog((prev) => [...prev, `> Error: ${data.message}`])
        onError(data.code, data.message)
      },
    })

    return disconnect
  }, [styleId, onComplete, onError])

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [log])

  const histogramData = distribution
    ? distribution.r.map((_, i) => ({
        bin: i,
        R: distribution.r[i],
        G: distribution.g[i],
        B: distribution.b[i],
      }))
    : []

  const estimatedRemaining =
    progress > 0 && progress < 100
      ? Math.round((elapsed / progress) * (100 - progress))
      : null

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-50">
            Learning &ldquo;{styleName}&rdquo;
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 capitalize">{phase.replace('_', ' ')}</p>
        </div>
        {connectionState === 'reconnecting' && (
          <span
            className="inline-flex items-center gap-1.5 rounded-full bg-yellow-100 dark:bg-yellow-900/30 px-3 py-1 text-xs font-medium text-yellow-700 dark:text-yellow-400"
            role="status"
            aria-live="polite"
          >
            <span className="h-2 w-2 rounded-full bg-yellow-400 animate-pulse" aria-hidden="true" />
            Reconnecting...
          </span>
        )}
      </div>

      {/* Progress bar */}
      <div>
        <div className="flex justify-between text-sm mb-1">
          <span className="text-gray-600 dark:text-gray-400">{message || 'Waiting...'}</span>
          <span className="font-medium tabular-nums">{progress}%</span>
        </div>
        <div
          className="h-3 w-full rounded-full bg-gray-200 dark:bg-gray-700 overflow-hidden"
          role="progressbar"
          aria-valuenow={progress}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Training progress"
        >
          <div
            className="h-full rounded-full bg-gradient-to-r from-brand-500 to-brand-400 transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-400 dark:text-gray-500 mt-1">
          <span>Elapsed: {formatDuration(elapsed)}</span>
          {estimatedRemaining != null && (
            <span>~{formatDuration(estimatedRemaining)} remaining</span>
          )}
        </div>
      </div>

      {/* Epoch + DeltaE badges */}
      <div className="flex flex-wrap gap-3">
        {epoch && (
          <div className="rounded-lg bg-gray-100 dark:bg-gray-800 px-3 py-1.5 text-sm">
            <span className="text-gray-500 dark:text-gray-400">Epoch </span>
            <span className="font-medium tabular-nums">{epoch.current}</span>
            <span className="text-gray-400 dark:text-gray-500"> / {epoch.total}</span>
          </div>
        )}
        {deltaE != null && (
          <div className="rounded-lg bg-gray-100 dark:bg-gray-800 px-3 py-1.5 text-sm">
            <span className="text-gray-500 dark:text-gray-400">CIE76 &Delta;E </span>
            <span
              className={`font-medium tabular-nums ${deltaE < 1 ? 'text-green-600 dark:text-green-400' : 'text-orange-500'}`}
            >
              {deltaE.toFixed(2)}
            </span>
          </div>
        )}
      </div>

      {/* Analytics panels */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Color histogram */}
        <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-4">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Color Distribution
          </h3>
          {histogramData.length > 0 ? (
            <ResponsiveContainer width="100%" height={140}>
              <BarChart data={histogramData} barCategoryGap={0} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
                <XAxis dataKey="bin" hide />
                <YAxis hide />
                <Tooltip
                  contentStyle={{ fontSize: 11, padding: '4px 8px' }}
                  labelFormatter={(v) => `Bin ${v}`}
                />
                <Bar dataKey="R" fill="#ef4444" opacity={0.8} />
                <Bar dataKey="G" fill="#22c55e" opacity={0.8} />
                <Bar dataKey="B" fill="#3b82f6" opacity={0.8} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[140px] flex items-center justify-center text-xs text-gray-400">
              Waiting for analysis...
            </div>
          )}
        </div>

        {/* Style features */}
        <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-4">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Style Features Detected
          </h3>
          {features ? (
            <dl className="space-y-2">
              {(Object.entries(features) as [string, string][]).map(([key, level]) => (
                <div key={key} className="flex items-center justify-between">
                  <dt className="text-sm text-gray-600 dark:text-gray-400">
                    {FEATURE_LABELS[key] ?? key}
                  </dt>
                  <dd className={`text-sm font-medium capitalize ${LEVEL_COLOR[level] ?? ''}`}>
                    {level}
                  </dd>
                </div>
              ))}
            </dl>
          ) : (
            <div className="h-full flex items-center justify-center text-xs text-gray-400 py-8">
              Waiting for analysis...
            </div>
          )}
        </div>
      </div>

      {/* Training log */}
      <div className="rounded-xl border border-gray-200 dark:border-gray-700">
        <div className="px-4 py-2 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
            Log
          </h3>
        </div>
        <div
          ref={logRef}
          className="h-32 overflow-y-auto scrollbar-thin px-4 py-2 space-y-0.5 font-mono text-xs text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-900/50 rounded-b-xl"
          aria-live="polite"
          aria-label="Training log"
        >
          {log.length === 0 ? (
            <span className="text-gray-400">Waiting for training to start...</span>
          ) : (
            log.map((line, i) => <div key={i}>{line}</div>)
          )}
        </div>
      </div>
    </div>
  )
}
