'use client'

import { useCallback, useState } from 'react'
import { useDropzone, type FileRejection } from 'react-dropzone'
import Image from 'next/image'
import { cn } from '@/lib/utils'

const ACCEPTED_TYPES = {
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
  'image/tiff': ['.tif', '.tiff'],
}

const MAX_FILES = 20
const MAX_SIZE_BYTES = 50 * 1024 * 1024 // 50 MB

export interface UploadFile {
  id: string
  file: File
  previewUrl: string
  uploadProgress: number
  status: 'pending' | 'uploading' | 'done' | 'error'
  errorMessage?: string
  storageKey?: string
}

interface ImageDropzoneProps {
  files: UploadFile[]
  onChange: (files: UploadFile[]) => void
  disabled?: boolean
  className?: string
}

export function ImageDropzone({ files, onChange, disabled, className }: ImageDropzoneProps) {
  const [rejections, setRejections] = useState<string[]>([])

  const onDrop = useCallback(
    (accepted: File[], rejected: FileRejection[]) => {
      setRejections([])

      const errorMessages: string[] = []
      rejected.forEach(({ file, errors }) => {
        errors.forEach((e) => {
          if (e.code === 'file-too-large') {
            errorMessages.push(`"${file.name}" exceeds 50 MB limit`)
          } else if (e.code === 'file-invalid-type') {
            errorMessages.push(`"${file.name}" is not a supported format`)
          } else if (e.code === 'too-many-files') {
            errorMessages.push(`Maximum ${MAX_FILES} images allowed`)
          } else {
            errorMessages.push(`"${file.name}": ${e.message}`)
          }
        })
      })
      if (errorMessages.length > 0) {
        setRejections([...new Set(errorMessages)])
      }

      const remaining = MAX_FILES - files.length
      const newFiles: UploadFile[] = accepted.slice(0, remaining).map((file) => ({
        id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
        file,
        previewUrl: URL.createObjectURL(file),
        uploadProgress: 0,
        status: 'pending',
      }))

      if (newFiles.length < accepted.length) {
        setRejections((prev) => [
          ...prev,
          `${accepted.length - newFiles.length} file(s) skipped — max ${MAX_FILES} images`,
        ])
      }

      onChange([...files, ...newFiles])
    },
    [files, onChange]
  )

  const removeFile = useCallback(
    (id: string) => {
      const file = files.find((f) => f.id === id)
      if (file) URL.revokeObjectURL(file.previewUrl)
      onChange(files.filter((f) => f.id !== id))
    },
    [files, onChange]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxSize: MAX_SIZE_BYTES,
    maxFiles: MAX_FILES,
    disabled,
  })

  return (
    <div className={cn('space-y-4', className)}>
      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={cn(
          'relative rounded-xl border-2 border-dashed p-8 text-center transition-colors cursor-pointer',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500',
          isDragActive
            ? 'border-brand-500 bg-brand-50 dark:bg-brand-950/20'
            : 'border-gray-300 dark:border-gray-700 hover:border-brand-400 dark:hover:border-brand-600',
          disabled && 'pointer-events-none opacity-50'
        )}
        aria-label="Image upload area. Drag and drop images or click to browse."
      >
        <input {...getInputProps()} aria-hidden="true" />
        <div className="flex flex-col items-center gap-3">
          <div className="rounded-full bg-gray-100 dark:bg-gray-800 p-4">
            <svg
              className="h-8 w-8 text-gray-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
              />
            </svg>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {isDragActive ? 'Drop images here' : 'Drag & drop images here, or click to browse'}
            </p>
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Supported: JPG, PNG, TIFF &nbsp;&bull;&nbsp; Max 50 MB per file &nbsp;&bull;&nbsp; Up to {MAX_FILES} images
            </p>
          </div>
        </div>
      </div>

      {/* Rejection errors */}
      {rejections.length > 0 && (
        <ul
          className="rounded-lg bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 px-4 py-3 space-y-1"
          role="alert"
          aria-live="polite"
        >
          {rejections.map((msg, i) => (
            <li key={i} className="text-sm text-red-600 dark:text-red-400 flex items-start gap-2">
              <span aria-hidden="true">&#x26A0;</span>
              {msg}
            </li>
          ))}
        </ul>
      )}

      {/* Preview grid */}
      {files.length > 0 && (
        <div
          className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-3"
          aria-label={`${files.length} image${files.length !== 1 ? 's' : ''} selected`}
        >
          {files.map((f) => (
            <div
              key={f.id}
              className="relative aspect-square rounded-lg overflow-hidden bg-gray-100 dark:bg-gray-800 group"
            >
              <Image
                src={f.previewUrl}
                alt={f.file.name}
                fill
                className="object-cover"
                sizes="(max-width: 640px) 33vw, (max-width: 768px) 25vw, 20vw"
              />

              {/* Upload progress overlay */}
              {f.status === 'uploading' && (
                <div className="absolute inset-0 bg-black/50 flex flex-col items-center justify-center">
                  <span className="text-white text-sm font-medium">{f.uploadProgress}%</span>
                  <div className="mt-1 w-3/4 h-1 bg-white/30 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-white transition-all duration-300"
                      style={{ width: `${f.uploadProgress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Done indicator */}
              {f.status === 'done' && (
                <div className="absolute top-1 right-1 rounded-full bg-green-500 p-0.5">
                  <svg className="h-3 w-3 text-white" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                </div>
              )}

              {/* Error indicator */}
              {f.status === 'error' && (
                <div
                  className="absolute inset-0 bg-red-900/60 flex items-center justify-center"
                  title={f.errorMessage}
                >
                  <span className="text-red-200 text-xs text-center px-1">Error</span>
                </div>
              )}

              {/* Remove button */}
              {f.status !== 'uploading' && (
                <button
                  type="button"
                  onClick={() => removeFile(f.id)}
                  className="absolute top-1 left-1 rounded-full bg-gray-900/70 p-1 text-white opacity-0 group-hover:opacity-100 focus-visible:opacity-100 transition-opacity"
                  aria-label={`Remove ${f.file.name}`}
                >
                  <svg className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              )}
            </div>
          ))}

          {/* Add more slot */}
          {files.length < MAX_FILES && (
            <div
              {...getRootProps()}
              className="relative aspect-square rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-700 flex items-center justify-center cursor-pointer hover:border-brand-400 transition-colors"
              aria-label="Add more images"
            >
              <input {...getInputProps()} aria-hidden="true" />
              <svg className="h-6 w-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </div>
          )}
        </div>
      )}

      {/* File count */}
      <p className="text-xs text-gray-500 dark:text-gray-400" aria-live="polite">
        {files.length} / {MAX_FILES} images selected
      </p>
    </div>
  )
}
