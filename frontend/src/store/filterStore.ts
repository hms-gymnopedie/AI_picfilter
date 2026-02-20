import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'
import type { Style, Filter, ViewMode, LearningMethod, ExportFormat, LUTSize } from '@/types/filter'
import type { ModelType } from '@/types/api'

interface StyleBlend {
  styleId: string
  styleName: string
  weight: number
}

interface FilterStoreState {
  // Active studio state
  activeStyleId: string | null
  activeFilterId: string | null
  intensity: number
  styleBlends: StyleBlend[]
  viewMode: ViewMode
  originalImageUrl: string | null
  originalImageFile: File | null

  // Library
  styles: Style[]
  filters: Filter[]

  // Model selection
  selectedModelType: ModelType

  // Export config
  exportFormat: ExportFormat
  exportLUTSize: LUTSize

  // UI
  isLearning: boolean
  learningProgress: number
  learningPhase: string
  learningMessage: string
  learningElapsed: number
  learningDeltaE: number | null

  // Actions
  setActiveStyle: (styleId: string | null) => void
  setActiveFilter: (filterId: string | null) => void
  setIntensity: (value: number) => void
  setStyleBlends: (blends: StyleBlend[]) => void
  updateBlendWeight: (styleId: string, weight: number) => void
  addStyleBlend: (blend: StyleBlend) => void
  removeStyleBlend: (styleId: string) => void
  setViewMode: (mode: ViewMode) => void
  setOriginalImage: (url: string | null, file: File | null) => void

  setStyles: (styles: Style[]) => void
  addStyle: (style: Style) => void
  removeStyle: (styleId: string) => void

  setFilters: (filters: Filter[]) => void
  addFilter: (filter: Filter) => void
  removeFilter: (filterId: string) => void

  setSelectedModelType: (type: ModelType) => void
  setExportFormat: (format: ExportFormat) => void
  setExportLUTSize: (size: LUTSize) => void

  setLearningProgress: (
    progress: number,
    phase: string,
    message: string,
    elapsed: number,
    deltaE?: number
  ) => void
  startLearning: () => void
  stopLearning: () => void

  resetStudio: () => void
}

export const useFilterStore = create<FilterStoreState>()(
  devtools(
    persist(
      (set) => ({
        activeStyleId: null,
        activeFilterId: null,
        intensity: 1.0,
        styleBlends: [],
        viewMode: 'side-by-side',
        originalImageUrl: null,
        originalImageFile: null,

        styles: [],
        filters: [],

        selectedModelType: 'nilut',

        exportFormat: 'cube',
        exportLUTSize: 33,

        isLearning: false,
        learningProgress: 0,
        learningPhase: '',
        learningMessage: '',
        learningElapsed: 0,
        learningDeltaE: null,

        setActiveStyle: (styleId) => set({ activeStyleId: styleId }),
        setActiveFilter: (filterId) => set({ activeFilterId: filterId }),
        setIntensity: (value) => set({ intensity: Math.max(0, Math.min(1, value)) }),

        setStyleBlends: (blends) => set({ styleBlends: blends }),
        updateBlendWeight: (styleId, weight) =>
          set((state) => ({
            styleBlends: state.styleBlends.map((b) =>
              b.styleId === styleId ? { ...b, weight } : b
            ),
          })),
        addStyleBlend: (blend) =>
          set((state) => ({
            styleBlends: [...state.styleBlends.filter((b) => b.styleId !== blend.styleId), blend],
          })),
        removeStyleBlend: (styleId) =>
          set((state) => ({
            styleBlends: state.styleBlends.filter((b) => b.styleId !== styleId),
          })),

        setViewMode: (mode) => set({ viewMode: mode }),
        setOriginalImage: (url, file) => set({ originalImageUrl: url, originalImageFile: file }),

        setStyles: (styles) => set({ styles }),
        addStyle: (style) =>
          set((state) => ({
            styles: [style, ...state.styles.filter((s) => s.id !== style.id)],
          })),
        removeStyle: (styleId) =>
          set((state) => ({ styles: state.styles.filter((s) => s.id !== styleId) })),

        setFilters: (filters) => set({ filters }),
        addFilter: (filter) =>
          set((state) => ({
            filters: [filter, ...state.filters.filter((f) => f.id !== filter.id)],
          })),
        removeFilter: (filterId) =>
          set((state) => ({ filters: state.filters.filter((f) => f.id !== filterId) })),

        setSelectedModelType: (type) => set({ selectedModelType: type }),
        setExportFormat: (format) => set({ exportFormat: format }),
        setExportLUTSize: (size) => set({ exportLUTSize: size }),

        setLearningProgress: (progress, phase, message, elapsed, deltaE) =>
          set({ learningProgress: progress, learningPhase: phase, learningMessage: message, learningElapsed: elapsed, learningDeltaE: deltaE ?? null }),
        startLearning: () =>
          set({ isLearning: true, learningProgress: 0, learningPhase: '', learningMessage: '', learningElapsed: 0, learningDeltaE: null }),
        stopLearning: () => set({ isLearning: false }),

        resetStudio: () =>
          set({
            activeStyleId: null,
            activeFilterId: null,
            intensity: 1.0,
            styleBlends: [],
            viewMode: 'side-by-side',
            originalImageUrl: null,
            originalImageFile: null,
          }),
      }),
      {
        name: 'picfilter-store',
        partialize: (state) => ({
          // Only persist library data and export prefs, not transient studio state
          styles: state.styles,
          filters: state.filters,
          selectedModelType: state.selectedModelType,
          exportFormat: state.exportFormat,
          exportLUTSize: state.exportLUTSize,
        }),
      }
    ),
    { name: 'FilterStore' }
  )
)
