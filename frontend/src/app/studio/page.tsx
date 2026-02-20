'use client'

import { useState, useCallback, useEffect, useRef, Suspense } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { LUTPreview } from '@/components/preview/LUTPreview'
import { useFilterStore } from '@/store/filterStore'
import { stylesApi, filtersApi } from '@/lib/api'
import { cn, normalizeWeights } from '@/lib/utils'
import { getCachedLUT, cacheLUT } from '@/lib/webgl/lut-texture'
import type { ViewMode, Style, LearningMethod } from '@/types/filter'

const METHOD_BADGE: Record<LearningMethod, { label: string; title?: string }> = {
  nilut: { label: 'NILUT' },
  adaptive_3dlut: {
    label: '3DLUT Adaptive',
    title: 'Image-Adaptive 3D LUT — automatically adjusts to each image\'s content',
  },
  dlut: { label: 'D-LUT' },
}

const VIEW_MODES: Array<{ id: ViewMode; label: string }> = [
  { id: 'side-by-side', label: 'Side by Side' },
  { id: 'slider', label: 'Slider Overlay' },
  { id: 'toggle', label: 'Toggle A/B' },
]

function StudioContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const styleIdParam = searchParams.get('styleId')
  const filterIdParam = searchParams.get('filterId')

  const {
    styles,
    filters,
    intensity,
    styleBlends,
    viewMode,
    originalImageUrl,
    setIntensity,
    setStyleBlends,
    updateBlendWeight,
    addStyleBlend,
    removeStyleBlend,
    setViewMode,
    setOriginalImage,
    setActiveStyle,
    addFilter,
  } = useFilterStore()

  const [activeStyle, setLocalActiveStyle] = useState<Style | null>(null)
  const [lutData, setLutData] = useState<Float32Array | null>(null)
  const [lutSize, setLutSize] = useState(33)
  const [isLoadingLUT, setIsLoadingLUT] = useState(false)
  const [lutError, setLutError] = useState<string | null>(null)
  const [useServerPreview, setUseServerPreview] = useState(false)
  const [serverPreviewUrl, setServerPreviewUrl] = useState<string | null>(null)
  const [isSaving, setIsSaving] = useState(false)
  const [saveSuccess, setSaveSuccess] = useState(false)

  // Resolve initial style from URL params
  useEffect(() => {
    const targetId = styleIdParam ?? filterIdParam
    if (!targetId) return

    let style: Style | undefined
    if (styleIdParam) {
      style = styles.find((s) => s.id === styleIdParam)
    } else if (filterIdParam) {
      const filter = filters.find((f) => f.id === filterIdParam)
      style = filter ? styles.find((s) => s.id === filter.styleId) : undefined
    }

    if (style) {
      setLocalActiveStyle(style)
      setActiveStyle(style.id)
    }
  }, [styleIdParam, filterIdParam, styles, filters, setActiveStyle])

  // Load LUT when activeStyle changes
  useEffect(() => {
    if (!activeStyle) return

    const weightsKey = styleBlends.map((b) => `${b.styleId}:${b.weight}`).join(',')
    const cacheKey = `${activeStyle.id}|${lutSize}|${weightsKey}`

    async function loadLUT() {
      setIsLoadingLUT(true)
      setLutError(null)
      try {
        // Check IndexedDB cache first
        const cached = await getCachedLUT(cacheKey)
        if (cached) {
          setLutData(new Float32Array(cached))
          setIsLoadingLUT(false)
          return
        }

        const weights = styleBlends.length > 0
          ? normalizeWeights(styleBlends.map((b) => b.weight)).join(',')
          : undefined

        const buffer = await stylesApi.getLUT(activeStyle!.id, {
          format: 'texture',
          size: lutSize as 17 | 33 | 65,
          styleWeights: weights,
        }) as ArrayBuffer

        await cacheLUT(cacheKey, buffer)
        setLutData(new Float32Array(buffer))
      } catch (e) {
        setLutError('Failed to load LUT data. Preview unavailable.')
      } finally {
        setIsLoadingLUT(false)
      }
    }

    loadLUT()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeStyle?.id, lutSize])

  const handleImageUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const url = URL.createObjectURL(file)
    setOriginalImage(url, file)
    setServerPreviewUrl(null)
  }, [setOriginalImage])

  const handleSaveFilter = useCallback(async () => {
    if (!activeStyle) return
    setIsSaving(true)
    try {
      const filter = await filtersApi.save({
        name: `${activeStyle.name} preset`,
        styleId: activeStyle.id,
        intensity,
        styleWeights: styleBlends.length > 0
          ? normalizeWeights(styleBlends.map((b) => b.weight))
          : [1],
      })
      addFilter(filter)
      setSaveSuccess(true)
      setTimeout(() => setSaveSuccess(false), 3000)
    } catch {}
    setIsSaving(false)
  }, [activeStyle, intensity, styleBlends, addFilter])

  const supportsBlending = activeStyle?.method === 'nilut'

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Toolbar */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => router.push('/')}
            className="flex items-center gap-1.5 text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back
          </button>
          <h1 className="text-lg font-bold text-gray-900 dark:text-white">Filter Studio</h1>
          {activeStyle && (
            <>
              <span className="text-sm text-gray-500 dark:text-gray-400">
                — {activeStyle.name}
              </span>
              <ModelBadge method={activeStyle.method} />
            </>
          )}
        </div>
        <div className="flex items-center gap-2">
          {saveSuccess && (
            <span className="text-sm text-green-600 dark:text-green-400" role="status" aria-live="polite">
              Saved!
            </span>
          )}
          <button
            type="button"
            onClick={handleSaveFilter}
            disabled={!activeStyle || isSaving}
            className="rounded-lg border border-gray-200 dark:border-gray-700 px-4 py-1.5 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 disabled:opacity-50 transition-colors"
          >
            {isSaving ? 'Saving...' : 'Save'}
          </button>
          <button
            type="button"
            onClick={() => router.push(`/studio/export?styleId=${activeStyle?.id}`)}
            disabled={!activeStyle}
            className="rounded-lg bg-brand-500 px-4 py-1.5 text-sm font-medium text-white hover:bg-brand-600 disabled:opacity-50 transition-colors"
          >
            Export .cube
          </button>
        </div>
      </div>

      {/* View mode tabs */}
      <div className="flex gap-1 rounded-lg bg-gray-100 dark:bg-gray-800 p-1 w-fit" role="tablist" aria-label="Comparison view mode">
        {VIEW_MODES.map((vm) => (
          <button
            key={vm.id}
            type="button"
            role="tab"
            aria-selected={viewMode === vm.id}
            onClick={() => setViewMode(vm.id)}
            className={cn(
              'rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
              viewMode === vm.id
                ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow-sm'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
            )}
          >
            {vm.label}
          </button>
        ))}
      </div>

      {/* Main canvas area */}
      <div className={cn(
        'gap-4',
        viewMode === 'side-by-side' ? 'grid grid-cols-1 lg:grid-cols-2' : 'flex flex-col'
      )}>
        {/* Original image */}
        <div className="space-y-2">
          <p className="text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Original</p>
          <div className="relative min-h-64 rounded-xl border-2 border-dashed border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 overflow-hidden">
            {originalImageUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={originalImageUrl}
                alt="Original image"
                className="w-full h-full object-contain max-h-[480px]"
              />
            ) : (
              <label className="absolute inset-0 flex flex-col items-center justify-center cursor-pointer gap-3">
                <div className="rounded-full bg-gray-200 dark:bg-gray-700 p-4">
                  <svg className="h-8 w-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                  </svg>
                </div>
                <span className="text-sm text-gray-500 dark:text-gray-400">Click to upload target image</span>
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/tiff"
                  onChange={handleImageUpload}
                  className="sr-only"
                  aria-label="Upload target image"
                />
              </label>
            )}
          </div>
        </div>

        {/* Filtered preview */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <p className="text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Filtered</p>
            {isLoadingLUT && (
              <span className="text-xs text-gray-400 animate-pulse">Loading LUT...</span>
            )}
            {lutError && (
              <span className="text-xs text-red-500" role="alert">{lutError}</span>
            )}
          </div>
          {useServerPreview ? (
            serverPreviewUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={serverPreviewUrl}
                alt="Server-rendered filtered preview"
                className="w-full rounded-xl max-h-[480px] object-contain"
              />
            ) : (
              <div className="min-h-64 rounded-xl bg-gray-100 dark:bg-gray-800 flex items-center justify-center text-sm text-gray-400">
                Server preview unavailable
              </div>
            )
          ) : (
            <LUTPreview
              imageUrl={originalImageUrl}
              lutData={isLoadingLUT ? null : lutData}
              lutSize={lutSize}
              intensity={intensity}
              className="min-h-64 max-h-[480px]"
              onWebGLUnsupported={() => setUseServerPreview(true)}
            />
          )}
        </div>
      </div>

      {/* Filter controls */}
      <div className="rounded-2xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-6 space-y-6">
        <div>
          <h2 className="text-sm font-semibold text-gray-800 dark:text-gray-200 mb-4">
            Active Filter: {activeStyle?.name ?? 'None'}
          </h2>

          {/* Intensity slider */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <label htmlFor="intensity-slider" className="text-gray-600 dark:text-gray-400">
                Intensity
              </label>
              <span className="font-medium tabular-nums text-gray-900 dark:text-white">
                {Math.round(intensity * 100)}%
              </span>
            </div>
            <input
              id="intensity-slider"
              type="range"
              min={0}
              max={100}
              value={Math.round(intensity * 100)}
              onChange={(e) => setIntensity(parseInt(e.target.value) / 100)}
              className="w-full accent-brand-500"
              aria-label={`Filter intensity: ${Math.round(intensity * 100)}%`}
            />
          </div>
        </div>

        {/* Style blending (NILUT only) */}
        {supportsBlending && (
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">Style Blend</h3>
            {styleBlends.length === 0 && (
              <p className="text-xs text-gray-400 dark:text-gray-500">
                Add styles to blend up to 5 styles together.
              </p>
            )}
            {styleBlends.map((blend, i) => (
              <div key={blend.styleId} className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600 dark:text-gray-400 truncate max-w-[200px]">
                    Style {String.fromCharCode(65 + i)}: {blend.styleName}
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="font-medium tabular-nums text-gray-900 dark:text-white">
                      {Math.round(blend.weight * 100)}%
                    </span>
                    <button
                      type="button"
                      onClick={() => removeStyleBlend(blend.styleId)}
                      className="text-gray-400 hover:text-red-500 transition-colors"
                      aria-label={`Remove ${blend.styleName} from blend`}
                    >
                      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={Math.round(blend.weight * 100)}
                  onChange={(e) => updateBlendWeight(blend.styleId, parseInt(e.target.value) / 100)}
                  className="w-full accent-brand-500"
                  aria-label={`Blend weight for ${blend.styleName}: ${Math.round(blend.weight * 100)}%`}
                />
              </div>
            ))}
            {styleBlends.length < 5 && (
              <button
                type="button"
                onClick={() => {
                  const availableStyle = styles.find(
                    (s) => s.id !== activeStyle?.id && !styleBlends.find((b) => b.styleId === s.id)
                  )
                  if (availableStyle) {
                    addStyleBlend({ styleId: availableStyle.id, styleName: availableStyle.name, weight: 0.5 })
                  }
                }}
                className="flex items-center gap-1.5 text-sm text-brand-600 dark:text-brand-400 hover:text-brand-700 dark:hover:text-brand-300 transition-colors"
              >
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                Add Style to Blend
              </button>
            )}
          </div>
        )}

        {!supportsBlending && activeStyle && (
          <p className="text-xs text-gray-400 dark:text-gray-500">
            Style blending is only available with NILUT models.
          </p>
        )}

        {/* Quick filter bar */}
        {styles.length > 0 && (
          <div className="space-y-2">
            <h3 className="text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">
              Quick Filters
            </h3>
            <div className="flex flex-wrap gap-2">
              {styles.slice(0, 6).map((style) => (
                <button
                  key={style.id}
                  type="button"
                  onClick={() => {
                    setLocalActiveStyle(style)
                    setActiveStyle(style.id)
                    setStyleBlends([])
                  }}
                  className={cn(
                    'rounded-full px-3 py-1 text-xs font-medium transition-colors',
                    activeStyle?.id === style.id
                      ? 'bg-brand-500 text-white'
                      : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
                  )}
                >
                  {style.name}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Model badge with adaptive tooltip ────────────────────────────────────────

function ModelBadge({ method }: { method: LearningMethod }) {
  const info = METHOD_BADGE[method]
  const isAdaptive = method === 'adaptive_3dlut'

  return (
    <span
      className={cn(
        'relative group inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium cursor-default select-none',
        isAdaptive
          ? 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300'
          : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
      )}
      aria-label={info.title ?? info.label}
    >
      {info.label}
      {isAdaptive && (
        <>
          {/* Info icon */}
          <svg
            className="h-3 w-3 opacity-60"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          {/* Tooltip */}
          <span
            role="tooltip"
            className={cn(
              'pointer-events-none absolute left-1/2 -translate-x-1/2 top-full mt-2 z-10',
              'w-64 rounded-lg bg-gray-900 dark:bg-gray-700 px-3 py-2 text-xs text-white shadow-lg',
              'opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity duration-150'
            )}
          >
            <strong className="block mb-0.5">Image-Adaptive 3D LUT</strong>
            Automatically adjusts color grading to each image&apos;s unique content. A lightweight
            CNN predicts per-image blend weights across basis LUTs, giving more accurate results
            than a fixed LUT.
            {/* Tooltip arrow */}
            <span
              className="absolute bottom-full left-1/2 -translate-x-1/2 border-4 border-transparent border-b-gray-900 dark:border-b-gray-700"
              aria-hidden="true"
            />
          </span>
        </>
      )}
    </span>
  )
}

export default function StudioPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center min-h-64">
        <div className="animate-spin h-8 w-8 rounded-full border-4 border-brand-500 border-t-transparent" role="status" aria-label="Loading studio" />
      </div>
    }>
      <StudioContent />
    </Suspense>
  )
}
