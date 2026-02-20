export type LearningMethod = 'adaptive_3dlut' | 'nilut' | 'dlut'

export type StyleStatus = 'queued' | 'preprocessing' | 'training' | 'completed' | 'failed' | 'cancelled'

export type LearningPhase = 'preprocessing' | 'training' | 'finalizing'

export interface StyleFeatures {
  warmTones: 'low' | 'medium' | 'high'
  contrast: 'low' | 'medium' | 'high'
  saturation: 'low' | 'medium' | 'high'
  shadowDetail: 'low' | 'medium' | 'high'
}

export interface StyleMetrics {
  deltaE: number
  epochs: number
  trainingTimeSeconds: number
}

export interface ReferenceImage {
  key: string
  thumbnailUrl: string
  width: number
  height: number
}

export interface Style {
  id: string
  name: string
  method: LearningMethod
  status: StyleStatus
  imageCount: number
  thumbnailUrl?: string
  styleCount?: number
  referenceImages?: ReferenceImage[]
  metrics?: StyleMetrics
  styleFeatures?: StyleFeatures
  createdAt: string
  completedAt?: string
}

export interface StyleBlendWeight {
  styleId: string
  styleName: string
  weight: number
}

export interface Filter {
  id: string
  name: string
  styleId: string
  intensity: number
  styleWeights: number[]
  thumbnailUrl?: string
  createdAt: string
  style?: Style
}

export interface LUTData {
  size: number
  data: number[]
  domain: {
    min: [number, number, number]
    max: [number, number, number]
  }
}

export interface ColorDistribution {
  r: number[]
  g: number[]
  b: number[]
  dominantColors: string[]
}

export interface ProgressEvent {
  phase: LearningPhase
  progress: number
  epoch?: number
  totalEpochs?: number
  deltaE?: number
  message: string
  elapsed: number
}

export interface ExportJob {
  exportId: string
  status: 'generating' | 'ready' | 'failed'
  filename?: string
  sizeBytes?: number
  downloadUrl?: string
  expiresAt?: string
  estimatedSeconds?: number
}

export type ExportFormat = 'cube' | 'hald_png' | 'nilut_json'

export type LUTSize = 17 | 33 | 65

export type ViewMode = 'side-by-side' | 'slider' | 'toggle'
