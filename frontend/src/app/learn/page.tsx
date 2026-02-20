'use client'

import { useState, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { ImageDropzone, type UploadFile } from '@/components/upload/ImageDropzone'
import { ModelSelector } from '@/components/model/ModelSelector'
import { uploadsApi, stylesApi } from '@/lib/api'
import { useFilterStore } from '@/store/filterStore'
import { cn } from '@/lib/utils'
import type { LearningMethod } from '@/types/filter'
import type { ModelType } from '@/types/api'

// Map the two-option ModelType to the full LearningMethod the API accepts.
// D-LUT remains available only via the advanced selector in Step 2.
function modelTypeToMethod(type: ModelType): LearningMethod {
  return type === 'lut3d' ? 'adaptive_3dlut' : 'nilut'
}

type Step = 1 | 2

export default function LearnPage() {
  const router = useRouter()
  const { addStyle, selectedModelType, setSelectedModelType } = useFilterStore()

  const [step, setStep] = useState<Step>(1)
  const [files, setFiles] = useState<UploadFile[]>([])
  const [styleName, setStyleName] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  // Derived: the LearningMethod sent to the API
  const method: LearningMethod = modelTypeToMethod(selectedModelType)

  const handleSubmit = useCallback(async () => {
    if (files.length === 0 || !styleName.trim()) return
    setIsSubmitting(true)
    setSubmitError(null)

    try {
      // 1. Get presigned URLs
      const presignRes = await uploadsApi.presign({
        files: files.map((f) => ({
          filename: f.file.name,
          contentType: f.file.type || 'image/jpeg',
          sizeBytes: f.file.size,
        })),
      })

      // 2. Upload files directly to storage with per-file progress
      const updatedFiles = [...files]
      await Promise.all(
        presignRes.uploads.map(async (upload, i) => {
          updatedFiles[i] = { ...updatedFiles[i], status: 'uploading', uploadProgress: 0 }
          setFiles([...updatedFiles])

          await uploadsApi.uploadToStorage(upload.uploadUrl, files[i].file, (pct) => {
            updatedFiles[i] = { ...updatedFiles[i], uploadProgress: pct }
            setFiles([...updatedFiles])
          })

          updatedFiles[i] = {
            ...updatedFiles[i],
            status: 'done',
            uploadProgress: 100,
            storageKey: upload.key,
          }
          setFiles([...updatedFiles])
        })
      )

      // 3. Confirm uploads
      await uploadsApi.confirm({ keys: presignRes.uploads.map((u) => u.key) })

      // 4. Create style learning job — include selected model type
      const style = await stylesApi.create({
        name: styleName.trim(),
        method,
        imageKeys: presignRes.uploads.map((u) => u.key),
      })

      addStyle(style)

      // 5. Navigate to progress page
      router.push(`/learn/${style.id}/progress`)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Upload failed. Please try again.'
      setSubmitError(msg)
      setIsSubmitting(false)
    }
  }, [files, styleName, method, addStyle, router])

  const canSubmit = files.length > 0 && styleName.trim().length > 0 && !isSubmitting

  return (
    <div className="max-w-2xl mx-auto space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-900 dark:text-white">Style Learning</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
            Upload reference images to teach the AI a visual style.
          </p>
        </div>
        <span className="text-sm text-gray-400 dark:text-gray-500 tabular-nums">
          Step {step} of 2
        </span>
      </div>

      {/* Step indicator */}
      <div className="flex items-center gap-3" aria-label="Progress steps">
        {([1, 2] as Step[]).map((s) => (
          <div key={s} className="flex items-center gap-3 flex-1">
            <div
              className={cn(
                'h-7 w-7 rounded-full flex items-center justify-center text-xs font-bold transition-colors',
                s <= step
                  ? 'bg-brand-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400'
              )}
              aria-current={s === step ? 'step' : undefined}
            >
              {s}
            </div>
            {s < 2 && (
              <div
                className={cn(
                  'h-0.5 flex-1 transition-colors',
                  step > s ? 'bg-brand-500' : 'bg-gray-200 dark:bg-gray-700'
                )}
              />
            )}
          </div>
        ))}
      </div>

      {/* ── Step 1: Select model + upload images ─────────────────────────── */}
      {step === 1 && (
        <section className="space-y-6">
          {/* Model selection — placed before upload to guide intent */}
          <div className="space-y-3">
            <h2 className="font-semibold text-gray-800 dark:text-gray-200">
              Step 1: Choose AI Model
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Select the model that best fits your use case. Your choice affects style blending
              capability, output quality, and inference speed.
            </p>
            <ModelSelector
              value={selectedModelType}
              onChange={(type) => setSelectedModelType(type)}
            />
          </div>

          {/* Image upload */}
          <div className="space-y-3">
            <h2 className="font-semibold text-gray-800 dark:text-gray-200">
              Upload Reference Images
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Upload 1–20 images that represent the visual style you want to learn.
            </p>
            <ImageDropzone files={files} onChange={setFiles} disabled={isSubmitting} />
          </div>

          <div className="flex justify-end">
            <button
              type="button"
              onClick={() => setStep(2)}
              disabled={files.length === 0}
              className="rounded-lg bg-brand-500 px-5 py-2 text-sm font-medium text-white hover:bg-brand-600 disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 transition-colors"
            >
              Next: Configure
            </button>
          </div>
        </section>
      )}

      {/* ── Step 2: Name + confirm ────────────────────────────────────────── */}
      {step === 2 && (
        <section className="space-y-6">
          <h2 className="font-semibold text-gray-800 dark:text-gray-200">
            Step 2: Configure Learning
          </h2>

          {/* Selected model summary */}
          <div className="flex items-center gap-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 px-4 py-3">
            <div className="flex-1 min-w-0">
              <p className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-0.5">
                Selected model
              </p>
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                {selectedModelType === 'nilut'
                  ? 'NILUT — Neural Implicit LUT'
                  : 'Image-Adaptive 3D LUT'}
              </p>
            </div>
            <button
              type="button"
              onClick={() => setStep(1)}
              className="text-xs text-brand-600 dark:text-brand-400 hover:underline shrink-0"
            >
              Change
            </button>
          </div>

          {/* Style name */}
          <div className="space-y-1.5">
            <label
              htmlFor="styleName"
              className="block text-sm font-medium text-gray-700 dark:text-gray-300"
            >
              Style Name
            </label>
            <input
              id="styleName"
              type="text"
              value={styleName}
              onChange={(e) => setStyleName(e.target.value)}
              placeholder="e.g. Warm Sunset, Cool Film..."
              maxLength={64}
              className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 px-3 py-2 text-sm text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-brand-500 transition"
              aria-required="true"
            />
          </div>

          {/* Upload summary */}
          <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 px-4 py-3">
            <p className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-0.5">
              Reference images
            </p>
            <p className="text-sm font-medium text-gray-900 dark:text-white">
              {files.length} image{files.length !== 1 ? 's' : ''} selected
            </p>
          </div>

          {/* Error */}
          {submitError && (
            <div
              className="rounded-lg bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 px-4 py-3"
              role="alert"
            >
              <p className="text-sm text-red-600 dark:text-red-400">{submitError}</p>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center justify-between pt-2">
            <button
              type="button"
              onClick={() => setStep(1)}
              disabled={isSubmitting}
              className="text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors"
            >
              Back
            </button>
            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => router.push('/')}
                disabled={isSubmitting}
                className="rounded-lg border border-gray-200 dark:border-gray-700 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSubmit}
                disabled={!canSubmit}
                className="flex items-center gap-2 rounded-lg bg-brand-500 px-5 py-2 text-sm font-medium text-white hover:bg-brand-600 disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 transition-colors"
                aria-busy={isSubmitting}
              >
                {isSubmitting && (
                  <svg
                    className="h-4 w-4 animate-spin"
                    viewBox="0 0 24 24"
                    fill="none"
                    aria-hidden="true"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                    />
                  </svg>
                )}
                {isSubmitting ? 'Starting...' : 'Start Learning'}
              </button>
            </div>
          </div>
        </section>
      )}
    </div>
  )
}
