'use client'

import { useState, Suspense } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { stylesApi, exportsApi } from '@/lib/api'
import { useFilterStore } from '@/store/filterStore'
import { normalizeWeights } from '@/lib/utils'
import type { ExportFormat, LUTSize } from '@/types/filter'

const FORMATS: Array<{ id: ExportFormat; label: string; description: string; compatWith?: string }> = [
  {
    id: 'cube',
    label: '.cube (3D LUT)',
    description: 'Compatible with Adobe Photoshop, Premiere Pro, DaVinci Resolve, and more.',
    compatWith: 'Photoshop, Premiere, DaVinci',
  },
  {
    id: 'hald_png',
    label: '.png (Hald image)',
    description: 'Hald CLUT format, usable in GIMP, darktable, and RawTherapee.',
    compatWith: 'GIMP, darktable',
  },
  {
    id: 'nilut_json',
    label: '.json (NILUT weights)',
    description: 'Raw model weights for programmatic use. Only available for NILUT models.',
    compatWith: 'Python, custom pipelines',
  },
]

const LUT_SIZES: LUTSize[] = [17, 33, 65]

function ExportContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const styleIdParam = searchParams.get('styleId') ?? searchParams.get('filterId')

  const { styles, filters, styleBlends, exportFormat, exportLUTSize, setExportFormat, setExportLUTSize } = useFilterStore()

  const style = styles.find((s) => s.id === styleIdParam) ??
    (() => {
      const f = filters.find((f) => f.id === styleIdParam)
      return f ? styles.find((s) => s.id === f.styleId) : undefined
    })()

  const [isExporting, setIsExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const blendWeights = styleBlends.length > 0
    ? normalizeWeights(styleBlends.map((b) => b.weight))
    : undefined

  const handleExport = async () => {
    if (!style) return
    setIsExporting(true)
    setError(null)
    try {
      const job = await stylesApi.export(style.id, {
        format: exportFormat,
        lutSize: exportLUTSize,
        styleWeights: blendWeights,
      })

      const filename = `${style.name.replace(/\s+/g, '_')}_${exportLUTSize}x${exportLUTSize}x${exportLUTSize}.${exportFormat === 'cube' ? 'cube' : exportFormat === 'hald_png' ? 'png' : 'json'}`
      await exportsApi.pollAndDownload(job.exportId, filename)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Export failed. Please try again.')
    } finally {
      setIsExporting(false)
    }
  }

  const isNILUT = style?.method === 'nilut'

  return (
    <div className="max-w-lg mx-auto space-y-6 animate-fade-in">
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => router.back()}
          className="flex items-center gap-1.5 text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back
        </button>
        <h1 className="text-xl font-bold text-gray-900 dark:text-white">Export Filter</h1>
      </div>

      {style && (
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Filter: <span className="font-medium text-gray-900 dark:text-white">{style.name}</span>
          {' '}({style.method.toUpperCase()}{styleBlends.length > 0 ? `, ${styleBlends.length} blended styles` : ''})
        </p>
      )}

      {/* Format selector */}
      <fieldset>
        <legend className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
          Export Format
        </legend>
        <div className="space-y-2">
          {FORMATS.map((fmt) => {
            const disabled = fmt.id === 'nilut_json' && !isNILUT
            return (
              <label
                key={fmt.id}
                className={`flex items-start gap-3 p-3 rounded-xl border ${
                  disabled
                    ? 'opacity-40 cursor-not-allowed border-gray-200 dark:border-gray-700'
                    : exportFormat === fmt.id
                    ? 'border-brand-500 bg-brand-50 dark:bg-brand-950/20 cursor-pointer'
                    : 'border-gray-200 dark:border-gray-700 cursor-pointer hover:border-gray-300 dark:hover:border-gray-600'
                }`}
              >
                <input
                  type="radio"
                  name="exportFormat"
                  value={fmt.id}
                  checked={exportFormat === fmt.id}
                  onChange={() => !disabled && setExportFormat(fmt.id)}
                  disabled={disabled}
                  className="mt-0.5 accent-brand-500"
                />
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-gray-900 dark:text-gray-100">{fmt.label}</span>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{fmt.description}</p>
                  {fmt.compatWith && (
                    <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
                      Compatible: {fmt.compatWith}
                    </p>
                  )}
                </div>
              </label>
            )
          })}
        </div>
      </fieldset>

      {/* LUT size selector (not for nilut_json) */}
      {exportFormat !== 'nilut_json' && (
        <div>
          <label htmlFor="lut-size" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            LUT Size
          </label>
          <select
            id="lut-size"
            value={exportLUTSize}
            onChange={(e) => setExportLUTSize(parseInt(e.target.value) as LUTSize)}
            className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 px-3 py-2 text-sm text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-brand-500"
          >
            {LUT_SIZES.map((s) => (
              <option key={s} value={s}>
                {s}x{s}x{s} — {s === 17 ? '~150 KB, fastest' : s === 33 ? '~1.1 MB, balanced' : '~8.3 MB, highest quality'}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Blend snapshot */}
      {styleBlends.length > 0 && (
        <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-4">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Current Blend Snapshot</h3>
          <div className="flex flex-wrap gap-2">
            {styleBlends.map((b, i) => (
              <span key={b.styleId} className="text-sm text-gray-600 dark:text-gray-400">
                Style {String.fromCharCode(65 + i)}: {Math.round(b.weight * 100)}%
                {i < styleBlends.length - 1 && ' |'}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="rounded-lg bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 px-4 py-3" role="alert">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center justify-between pt-2">
        <button
          type="button"
          onClick={() => router.back()}
          className="rounded-lg border border-gray-200 dark:border-gray-700 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={handleExport}
          disabled={!style || isExporting}
          className="flex items-center gap-2 rounded-lg bg-brand-500 px-5 py-2 text-sm font-medium text-white hover:bg-brand-600 disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 transition-colors"
          aria-busy={isExporting}
        >
          {isExporting && (
            <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          )}
          {isExporting ? 'Generating...' : 'Download'}
        </button>
      </div>
    </div>
  )
}

export default function ExportPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center min-h-64">
        <div className="animate-spin h-8 w-8 rounded-full border-4 border-brand-500 border-t-transparent" role="status" aria-label="Loading" />
      </div>
    }>
      <ExportContent />
    </Suspense>
  )
}
