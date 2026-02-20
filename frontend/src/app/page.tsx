import type { Metadata } from 'next'
import Link from 'next/link'
import Image from 'next/image'

export const metadata: Metadata = {
  title: 'Dashboard',
}

// Mock data for initial render — will be replaced with API calls
const RECENT_FILTERS = [
  { id: 'filter_1', name: 'Warm Sunset', method: 'nilut', styleCount: 3, thumbnailUrl: null, createdAt: '2026-02-15' },
  { id: 'filter_2', name: 'Cool Film', method: 'adaptive_3dlut', styleCount: 1, thumbnailUrl: null, createdAt: '2026-02-10' },
  { id: 'filter_3', name: 'Moody B&W', method: 'dlut', styleCount: 2, thumbnailUrl: null, createdAt: '2026-02-08' },
  { id: 'filter_4', name: 'Pastel Dream', method: 'nilut', styleCount: 1, thumbnailUrl: null, createdAt: '2026-02-05' },
]

const METHOD_LABELS: Record<string, string> = {
  nilut: 'NILUT',
  adaptive_3dlut: '3D LUT',
  dlut: 'D-LUT',
}

export default function DashboardPage() {
  return (
    <div className="space-y-8 animate-fade-in">
      {/* Hero */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Your Filters</h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Learn AI styles from reference images and apply them as reusable filters.
        </p>
      </div>

      {/* Filter grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {/* New style card */}
        <Link
          href="/learn"
          className="group relative flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-gray-300 dark:border-gray-700 p-8 text-center transition-colors hover:border-brand-400 dark:hover:border-brand-500 hover:bg-brand-50 dark:hover:bg-brand-950/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
          aria-label="Create a new style"
        >
          <div className="rounded-full bg-gray-100 dark:bg-gray-800 group-hover:bg-brand-100 dark:group-hover:bg-brand-900/30 p-4 transition-colors">
            <svg className="h-8 w-8 text-gray-400 group-hover:text-brand-500 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4v16m8-8H4" />
            </svg>
          </div>
          <div>
            <p className="font-medium text-gray-700 dark:text-gray-300 group-hover:text-brand-600 dark:group-hover:text-brand-400 transition-colors">
              New Style
            </p>
            <p className="mt-0.5 text-xs text-gray-400 dark:text-gray-500">
              Upload reference images to get started
            </p>
          </div>
        </Link>

        {/* Existing filter cards */}
        {RECENT_FILTERS.map((filter) => (
          <FilterCard key={filter.id} filter={filter} />
        ))}
      </div>

      {/* Empty state (shown when no filters) */}
      {RECENT_FILTERS.length === 0 && (
        <div className="text-center py-16">
          <div className="mx-auto h-16 w-16 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center mb-4">
            <svg className="h-8 w-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
            </svg>
          </div>
          <h2 className="text-lg font-medium text-gray-900 dark:text-white">No filters yet</h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Upload some reference images to train your first AI style.
          </p>
          <Link
            href="/learn"
            className="mt-4 inline-flex items-center gap-2 rounded-lg bg-brand-500 px-4 py-2 text-sm font-medium text-white hover:bg-brand-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 transition-colors"
          >
            Get started
          </Link>
        </div>
      )}
    </div>
  )
}

function FilterCard({ filter }: { filter: typeof RECENT_FILTERS[number] }) {
  return (
    <article className="group rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 overflow-hidden shadow-sm hover:shadow-md transition-shadow">
      {/* Thumbnail */}
      <div className="relative aspect-video bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-700">
        {filter.thumbnailUrl ? (
          <Image
            src={filter.thumbnailUrl}
            alt={`${filter.name} thumbnail`}
            fill
            className="object-cover"
            sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 25vw"
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <svg className="h-10 w-10 text-gray-300 dark:text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
        )}
        {/* Method badge */}
        <span className="absolute top-2 right-2 rounded-md bg-black/60 px-2 py-0.5 text-xs font-medium text-white">
          {METHOD_LABELS[filter.method] ?? filter.method}
        </span>
      </div>

      {/* Meta */}
      <div className="p-4 space-y-3">
        <div>
          <h3 className="font-medium text-gray-900 dark:text-white truncate">{filter.name}</h3>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
            {filter.styleCount} {filter.styleCount === 1 ? 'style' : 'styles'} &bull; {filter.createdAt}
          </p>
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <Link
            href={`/studio?filterId=${filter.id}`}
            className="flex-1 text-center rounded-lg bg-brand-500 px-3 py-1.5 text-xs font-medium text-white hover:bg-brand-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 transition-colors"
          >
            Apply
          </Link>
          <Link
            href={`/studio/export?filterId=${filter.id}`}
            className="flex-1 text-center rounded-lg border border-gray-200 dark:border-gray-700 px-3 py-1.5 text-xs font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-400 transition-colors"
          >
            Export
          </Link>
        </div>
      </div>
    </article>
  )
}
