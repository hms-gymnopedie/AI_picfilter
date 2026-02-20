'use client'

import { useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { LearningProgress } from '@/components/progress/LearningProgress'
import { useFilterStore } from '@/store/filterStore'

interface ProgressPageProps {
  params: { id: string }
}

export default function ProgressPage({ params }: ProgressPageProps) {
  const router = useRouter()
  const styles = useFilterStore((s) => s.styles)
  const style = styles.find((s) => s.id === params.id)
  const styleName = style?.name ?? 'Style'

  const handleComplete = useCallback(
    (_deltaE: number, _trainingTimeSeconds: number) => {
      // Brief delay so user can see 100% then navigate to studio
      setTimeout(() => {
        router.push(`/studio?styleId=${params.id}`)
      }, 1500)
    },
    [params.id, router]
  )

  const handleError = useCallback(
    (_code: string, _message: string) => {
      // Stay on page — LearningProgress renders the error in the log
    },
    []
  )

  return (
    <div className="max-w-2xl mx-auto">
      {/* Back nav */}
      <div className="mb-6 flex items-center justify-between">
        <button
          type="button"
          onClick={() => router.push('/')}
          className="flex items-center gap-1.5 text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back to Dashboard
        </button>
        <span className="text-sm text-gray-400 dark:text-gray-500 tabular-nums">Step 2 of 3</span>
      </div>

      <LearningProgress
        styleId={params.id}
        styleName={styleName}
        onComplete={handleComplete}
        onError={handleError}
      />
    </div>
  )
}
