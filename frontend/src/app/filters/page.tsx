'use client'

import { useState, useEffect, useCallback, Suspense } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Link from 'next/link'
import Image from 'next/image'
import { filtersApi, stylesApi } from '@/lib/api'
import { useFilterStore } from '@/store/filterStore'
import { cn } from '@/lib/utils'
import type { Filter } from '@/types/filter'

const SORT_OPTIONS = [
  { value: 'createdAt:desc', label: 'Recent' },
  { value: 'createdAt:asc', label: 'Oldest' },
  { value: 'name:asc', label: 'Name A-Z' },
  { value: 'name:desc', label: 'Name Z-A' },
]

const METHOD_LABELS: Record<string, string> = {
  nilut: 'NILUT',
  adaptive_3dlut: '3D LUT',
  dlut: 'D-LUT',
}

function FiltersContent() {
  const router = useRouter()
  const { filters, setFilters, removeFilter } = useFilterStore()
  const [search, setSearch] = useState('')
  const [sort, setSort] = useState('createdAt:desc')
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const loadFilters = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const res = await filtersApi.list({ search: search || undefined, sort, limit: 50 })
      setFilters(res.items)
    } catch {
      setError('Failed to load filters.')
    } finally {
      setIsLoading(false)
    }
  }, [search, sort, setFilters])

  useEffect(() => {
    const timer = setTimeout(loadFilters, search ? 300 : 0)
    return () => clearTimeout(timer)
  }, [loadFilters, search])

  const handleDelete = useCallback(async (filterId: string) => {
    if (!confirm('Delete this filter? This cannot be undone.')) return
    setDeletingId(filterId)
    try {
      await filtersApi.delete(filterId)
      removeFilter(filterId)
    } catch {
      alert('Failed to delete filter.')
    } finally {
      setDeletingId(null)
    }
  }, [removeFilter])

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">My Filters</h1>
        <Link
          href="/learn"
          className="rounded-lg bg-brand-500 px-4 py-2 text-sm font-medium text-white hover:bg-brand-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 transition-colors"
        >
          + New Style
        </Link>
      </div>

      {/* Search and sort */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="relative flex-1 min-w-48">
          <svg
            className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            aria-hidden="true"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            type="search"
            placeholder="Search filters..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 pl-9 pr-3 py-2 text-sm text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-brand-500"
            aria-label="Search filters"
          />
        </div>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Sort:
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value)}
            className="rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 px-3 py-2 text-sm text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-brand-500"
          >
            {SORT_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </label>
      </div>

      {/* Error */}
      {error && (
        <div className="rounded-lg bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 px-4 py-3" role="alert">
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
          <button
            type="button"
            onClick={loadFilters}
            className="mt-1 text-sm font-medium text-red-600 dark:text-red-400 underline hover:no-underline"
          >
            Retry
          </button>
        </div>
      )}

      {/* Loading */}
      {isLoading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="rounded-2xl border border-gray-200 dark:border-gray-800 overflow-hidden animate-pulse">
              <div className="aspect-video bg-gray-200 dark:bg-gray-700" />
              <div className="p-4 space-y-2">
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4" />
                <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-1/2" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Filter grid */}
      {!isLoading && filters.length === 0 && (
        <div className="text-center py-16">
          <div className="mx-auto h-16 w-16 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center mb-4">
            <svg className="h-8 w-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <h2 className="text-lg font-medium text-gray-900 dark:text-white">
            {search ? 'No filters found' : 'No saved filters yet'}
          </h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            {search ? `No results for "${search}"` : 'Create your first AI style to get started.'}
          </p>
        </div>
      )}

      {!isLoading && filters.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {filters.map((filter) => (
            <FilterLibraryCard
              key={filter.id}
              filter={filter}
              isDeleting={deletingId === filter.id}
              onDelete={() => handleDelete(filter.id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function FilterLibraryCard({
  filter,
  isDeleting,
  onDelete,
}: {
  filter: Filter
  isDeleting: boolean
  onDelete: () => void
}) {
  return (
    <article className={cn(
      'group rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 overflow-hidden shadow-sm hover:shadow-md transition-all',
      isDeleting && 'opacity-50 pointer-events-none'
    )}>
      {/* Thumbnail */}
      <div className="relative aspect-video bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-700">
        {filter.thumbnailUrl ? (
          <Image
            src={filter.thumbnailUrl}
            alt={`${filter.name} preview`}
            fill
            className="object-cover"
            sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <svg className="h-10 w-10 text-gray-300 dark:text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
        )}
      </div>

      {/* Meta */}
      <div className="p-4 space-y-3">
        <div>
          <h3 className="font-medium text-gray-900 dark:text-white truncate">{filter.name}</h3>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
            {new Date(filter.createdAt).toLocaleDateString()}
          </p>
        </div>

        {/* Actions */}
        <div className="grid grid-cols-2 gap-2">
          <Link
            href={`/studio?filterId=${filter.id}`}
            className="text-center rounded-lg bg-brand-500 px-2 py-1.5 text-xs font-medium text-white hover:bg-brand-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 transition-colors"
          >
            Apply
          </Link>
          <Link
            href={`/studio?filterId=${filter.id}`}
            className="text-center rounded-lg border border-gray-200 dark:border-gray-700 px-2 py-1.5 text-xs font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
          >
            Edit
          </Link>
          <Link
            href={`/studio/export?filterId=${filter.id}`}
            className="text-center rounded-lg border border-gray-200 dark:border-gray-700 px-2 py-1.5 text-xs font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
          >
            Export
          </Link>
          <button
            type="button"
            onClick={onDelete}
            disabled={isDeleting}
            className="rounded-lg border border-red-200 dark:border-red-900 px-2 py-1.5 text-xs font-medium text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/20 disabled:opacity-50 transition-colors"
          >
            {isDeleting ? 'Deleting...' : 'Delete'}
          </button>
        </div>
      </div>
    </article>
  )
}

export default function FiltersPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center min-h-64">
        <div className="animate-spin h-8 w-8 rounded-full border-4 border-brand-500 border-t-transparent" role="status" aria-label="Loading filters" />
      </div>
    }>
      <FiltersContent />
    </Suspense>
  )
}
