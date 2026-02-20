import type { Style, Filter, ExportJob, LUTData } from './filter'

// Model type definitions
export type ModelType = 'nilut' | 'lut3d'

export interface ModelTypeInfo {
  id: ModelType
  /** Display name */
  label: string
  /** One-line tagline */
  tagline: string
  /** Detailed description */
  description: string
  /** Highlighted capability bullets */
  features: string[]
  /** Approximate inference speed */
  speed: 'fast' | 'medium' | 'slow'
  /** Approximate model file size */
  modelSize: string
  /** Whether the model adapts per-image content */
  isAdaptive: boolean
  /** Whether multi-style blending is supported */
  supportsBlending: boolean
  /** Recommended use cases */
  useCases: string[]
  /** Badge text shown on card */
  badge: string
  /** Badge colour variant */
  badgeVariant: 'recommended' | 'fast' | 'quality'
}

export interface ApiError {
  error: {
    code: string
    message: string
    details?: Array<{ field: string; issue: string }>
    retryAfter?: number
  }
}

export interface Pagination {
  page: number
  limit: number
  total: number
  totalPages: number
}

export interface PaginatedResponse<T> {
  items: T[]
  pagination: Pagination
}

// Style endpoints
export interface CreateStyleRequest {
  name: string
  method: 'adaptive_3dlut' | 'nilut' | 'dlut'
  imageKeys: string[]
}

export type CreateStyleResponse = Style

export type ListStylesResponse = PaginatedResponse<Style>

export type GetStyleResponse = Style

// Upload endpoints
export interface PresignFileMetadata {
  filename: string
  contentType: string
  sizeBytes: number
}

export interface PresignRequest {
  files: PresignFileMetadata[]
}

export interface PresignedUpload {
  key: string
  uploadUrl: string
  expiresAt: string
}

export interface PresignResponse {
  uploads: PresignedUpload[]
}

export interface ConfirmRequest {
  keys: string[]
}

export interface ConfirmedImage {
  key: string
  valid: boolean
  width: number
  height: number
  thumbnailUrl: string
  error?: string
}

export interface ConfirmResponse {
  confirmed: ConfirmedImage[]
}

// LUT endpoint
export type GetLUTResponse = LUTData

// Apply endpoint
export interface ApplyFilterRequest {
  intensity: number
  styleWeights?: number[]
}

export interface ApplyFilterResponse {
  resultUrl: string
  expiresAt: string
}

// Export endpoints
export interface CreateExportRequest {
  format: 'cube' | 'hald_png' | 'nilut_json'
  lutSize: 17 | 33 | 65
  styleWeights?: number[]
}

export type CreateExportResponse = ExportJob

export type GetExportResponse = ExportJob

// Filter library endpoints
export interface SaveFilterRequest {
  name: string
  styleId: string
  intensity: number
  styleWeights: number[]
}

export type SaveFilterResponse = Filter

export type ListFiltersResponse = PaginatedResponse<Filter>

export type GetFilterResponse = Filter

export interface UpdateFilterRequest {
  name?: string
  intensity?: number
  styleWeights?: number[]
}

export type UpdateFilterResponse = Filter
