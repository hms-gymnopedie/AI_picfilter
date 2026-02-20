import ky, { type KyInstance, type Options } from 'ky'
import type {
  CreateStyleRequest,
  CreateStyleResponse,
  ListStylesResponse,
  GetStyleResponse,
  PresignRequest,
  PresignResponse,
  ConfirmRequest,
  ConfirmResponse,
  GetLUTResponse,
  ApplyFilterRequest,
  ApplyFilterResponse,
  CreateExportRequest,
  CreateExportResponse,
  GetExportResponse,
  SaveFilterRequest,
  SaveFilterResponse,
  ListFiltersResponse,
  GetFilterResponse,
  UpdateFilterRequest,
  UpdateFilterResponse,
} from '@/types/api'

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000/api/v1'

function getAuthToken(): string | null {
  if (typeof window === 'undefined') return null
  return localStorage.getItem('auth_token')
}

const apiClient: KyInstance = ky.create({
  prefixUrl: BASE_URL,
  timeout: 30_000,
  retry: {
    limit: 3,
    methods: ['get'],
    statusCodes: [408, 429, 503],
    backoffLimit: 10_000,
  },
  hooks: {
    beforeRequest: [
      (request) => {
        const token = getAuthToken()
        if (token) {
          request.headers.set('Authorization', `Bearer ${token}`)
        }
        const lang = typeof window !== 'undefined'
          ? (document.documentElement.lang ?? 'ko')
          : 'ko'
        request.headers.set('Accept-Language', lang)
      },
    ],
    afterResponse: [
      async (_request, _options, response) => {
        if (response.status === 429) {
          const data = await response.clone().json<{ error: { retryAfter?: number } }>()
          const retryAfter = data.error?.retryAfter ?? 30
          console.warn(`Rate limited. Retry after ${retryAfter}s`)
        }
        return response
      },
    ],
  },
})

// ── Style Learning ───────────────────────────────────────────────────────────

export const stylesApi = {
  create(body: CreateStyleRequest): Promise<CreateStyleResponse> {
    return apiClient.post('styles', { json: body }).json()
  },

  list(params?: { page?: number; limit?: number; sort?: string }): Promise<ListStylesResponse> {
    return apiClient.get('styles', { searchParams: params as Record<string, string | number> }).json()
  },

  get(styleId: string): Promise<GetStyleResponse> {
    return apiClient.get(`styles/${styleId}`).json()
  },

  delete(styleId: string): Promise<void> {
    return apiClient.delete(`styles/${styleId}`).json()
  },

  cancel(styleId: string): Promise<void> {
    return apiClient.post(`styles/${styleId}/cancel`).json()
  },

  getLUT(
    styleId: string,
    params?: { format?: 'texture' | 'json'; size?: 17 | 33 | 65; styleWeights?: string }
  ): Promise<GetLUTResponse | ArrayBuffer> {
    const format = params?.format ?? 'json'
    if (format === 'texture') {
      return apiClient
        .get(`styles/${styleId}/lut`, {
          searchParams: { format, ...(params?.size && { size: params.size }), ...(params?.styleWeights && { styleWeights: params.styleWeights }) },
        })
        .arrayBuffer()
    }
    return apiClient
      .get(`styles/${styleId}/lut`, {
        searchParams: { format, ...(params?.size && { size: params.size }) },
      })
      .json()
  },

  applyFilter(styleId: string, file: File, options: ApplyFilterRequest): Promise<ApplyFilterResponse> {
    const form = new FormData()
    form.append('image', file)
    form.append('intensity', String(options.intensity))
    if (options.styleWeights) {
      form.append('styleWeights', JSON.stringify(options.styleWeights))
    }
    return apiClient.post(`styles/${styleId}/apply`, { body: form }).json()
  },

  export(styleId: string, body: CreateExportRequest): Promise<CreateExportResponse> {
    return apiClient.post(`styles/${styleId}/export`, { json: body }).json()
  },
}

// ── Uploads ──────────────────────────────────────────────────────────────────

export const uploadsApi = {
  presign(body: PresignRequest): Promise<PresignResponse> {
    return apiClient.post('uploads/presign', { json: body }).json()
  },

  confirm(body: ConfirmRequest): Promise<ConfirmResponse> {
    return apiClient.post('uploads/confirm', { json: body }).json()
  },

  /**
   * Upload a single file directly to storage via a presigned PUT URL.
   * Uses XMLHttpRequest for upload progress events.
   */
  uploadToStorage(
    url: string,
    file: File,
    onProgress?: (percent: number) => void
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('PUT', url)
      xhr.setRequestHeader('Content-Type', file.type)

      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100))
        }
      })

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve()
        } else {
          reject(new Error(`Upload failed with status ${xhr.status}`))
        }
      })

      xhr.addEventListener('error', () => reject(new Error('Upload network error')))
      xhr.addEventListener('abort', () => reject(new Error('Upload aborted')))

      xhr.send(file)
    })
  },
}

// ── Exports ──────────────────────────────────────────────────────────────────

export const exportsApi = {
  getStatus(exportId: string): Promise<GetExportResponse> {
    return apiClient.get(`exports/${exportId}`).json()
  },

  /**
   * Poll export status until ready or failed, then trigger download.
   */
  async pollAndDownload(
    exportId: string,
    filename: string,
    intervalMs = 2000,
    maxAttempts = 60
  ): Promise<void> {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const status = await exportsApi.getStatus(exportId)
      if (status.status === 'ready' && status.downloadUrl) {
        const blob = await apiClient.get(status.downloadUrl.replace('/api/v1/', '')).blob()
        const objectUrl = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = objectUrl
        a.download = filename
        a.click()
        URL.revokeObjectURL(objectUrl)
        return
      }
      if (status.status === 'failed') {
        throw new Error('Export generation failed')
      }
      await new Promise((r) => setTimeout(r, intervalMs))
    }
    throw new Error('Export timed out')
  },
}

// ── Filter Library ───────────────────────────────────────────────────────────

export const filtersApi = {
  list(params?: { page?: number; limit?: number; search?: string; sort?: string }): Promise<ListFiltersResponse> {
    return apiClient.get('filters', { searchParams: params as Record<string, string | number> }).json()
  },

  save(body: SaveFilterRequest): Promise<SaveFilterResponse> {
    return apiClient.post('filters', { json: body }).json()
  },

  get(filterId: string): Promise<GetFilterResponse> {
    return apiClient.get(`filters/${filterId}`).json()
  },

  update(filterId: string, body: UpdateFilterRequest): Promise<UpdateFilterResponse> {
    return apiClient.patch(`filters/${filterId}`, { json: body }).json()
  },

  delete(filterId: string): Promise<void> {
    return apiClient.delete(`filters/${filterId}`).json()
  },
}

// ── SSE Progress ─────────────────────────────────────────────────────────────

export interface SSEHandlers {
  onProgress?: (data: import('@/types/filter').ProgressEvent) => void
  onMetrics?: (data: import('@/types/filter').ColorDistribution) => void
  onFeatures?: (data: import('@/types/filter').StyleFeatures) => void
  onComplete?: (data: { styleId: string; deltaE: number; trainingTimeSeconds: number }) => void
  onError?: (data: { code: string; message: string }) => void
  onReconnecting?: () => void
  onOpen?: () => void
}

export function connectProgressSSE(styleId: string, handlers: SSEHandlers): () => void {
  const url = `${BASE_URL}/styles/${styleId}/progress`
  const token = getAuthToken()
  const fullUrl = token ? `${url}?token=${encodeURIComponent(token)}` : url

  let reconnectAttempts = 0
  const MAX_RECONNECTS = 5
  let es: EventSource | null = null
  let closed = false

  function connect() {
    if (closed) return
    es = new EventSource(fullUrl)

    es.addEventListener('open', () => {
      reconnectAttempts = 0
      handlers.onOpen?.()
    })

    es.addEventListener('progress', (e) => {
      try {
        handlers.onProgress?.(JSON.parse(e.data))
      } catch {}
    })

    es.addEventListener('metrics', (e) => {
      try {
        handlers.onMetrics?.(JSON.parse(e.data))
      } catch {}
    })

    es.addEventListener('features', (e) => {
      try {
        handlers.onFeatures?.(JSON.parse(e.data))
      } catch {}
    })

    es.addEventListener('complete', (e) => {
      try {
        handlers.onComplete?.(JSON.parse(e.data))
      } catch {}
      es?.close()
    })

    es.addEventListener('error', (e) => {
      try {
        // Named 'error' event from server (learning failure)
        if ('data' in e && (e as MessageEvent).data) {
          handlers.onError?.(JSON.parse((e as MessageEvent).data))
          es?.close()
          return
        }
      } catch {}

      // Connection error - attempt reconnect
      es?.close()
      reconnectAttempts++
      if (reconnectAttempts <= MAX_RECONNECTS) {
        handlers.onReconnecting?.()
        setTimeout(connect, 3000)
      } else {
        // Fall back: caller should switch to polling GET /styles/{id}
        handlers.onError?.({ code: 'SSE_FAILED', message: 'Real-time connection lost. Switching to polling.' })
      }
    })
  }

  connect()

  // Return cleanup function
  return () => {
    closed = true
    es?.close()
  }
}
