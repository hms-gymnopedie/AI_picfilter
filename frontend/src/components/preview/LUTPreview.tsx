'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { createProgram, VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, QUAD_VERTICES } from '@/lib/webgl/lut-shader'
import { createLUT3DTexture, createImageTexture } from '@/lib/webgl/lut-texture'
import { cn } from '@/lib/utils'

interface LUTPreviewProps {
  imageUrl: string | null
  lutData: Float32Array | null
  lutSize: number
  intensity: number
  className?: string
  onWebGLUnsupported?: () => void
}

interface GLState {
  program: WebGLProgram
  vao: WebGLVertexArrayObject
  imageTexture: WebGLTexture | null
  lutTexture: WebGLTexture | null
  imageUrl: string | null
}

export function LUTPreview({
  imageUrl,
  lutData,
  lutSize,
  intensity,
  className,
  onWebGLUnsupported,
}: LUTPreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const glRef = useRef<WebGL2RenderingContext | null>(null)
  const stateRef = useRef<GLState | null>(null)
  const rafRef = useRef<number>(0)
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Initialize WebGL context and shader program once
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const gl = canvas.getContext('webgl2', { preserveDrawingBuffer: false })
    if (!gl) {
      setError('WebGL 2 is not supported in your browser.')
      onWebGLUnsupported?.()
      return
    }

    if (!gl.getExtension('EXT_color_buffer_float')) {
      setError('Float texture extension not available.')
      onWebGLUnsupported?.()
      return
    }

    glRef.current = gl

    try {
      const program = createProgram(gl, VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC)

      // Create VAO
      const vao = gl.createVertexArray()!
      gl.bindVertexArray(vao)

      const vbo = gl.createBuffer()!
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo)
      gl.bufferData(gl.ARRAY_BUFFER, QUAD_VERTICES, gl.STATIC_DRAW)

      const stride = 4 * Float32Array.BYTES_PER_ELEMENT
      const posLoc = gl.getAttribLocation(program, 'a_position')
      gl.enableVertexAttribArray(posLoc)
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, stride, 0)

      const uvLoc = gl.getAttribLocation(program, 'a_texCoord')
      gl.enableVertexAttribArray(uvLoc)
      gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, stride, 2 * Float32Array.BYTES_PER_ELEMENT)

      gl.bindVertexArray(null)

      stateRef.current = {
        program,
        vao,
        imageTexture: null,
        lutTexture: null,
        imageUrl: null,
      }
      setIsReady(true)
    } catch (e) {
      setError(String(e))
    }

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [onWebGLUnsupported])

  // Load image texture when imageUrl changes
  useEffect(() => {
    const gl = glRef.current
    const state = stateRef.current
    if (!gl || !state || !imageUrl) return
    if (state.imageUrl === imageUrl) return

    const img = new window.Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      if (state.imageTexture) gl.deleteTexture(state.imageTexture)
      const canvas = canvasRef.current
      if (canvas) {
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
      }
      state.imageTexture = createImageTexture(gl, img)
      state.imageUrl = imageUrl
      render()
    }
    img.src = imageUrl
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageUrl, isReady])

  // Update LUT texture when lutData changes
  useEffect(() => {
    const gl = glRef.current
    const state = stateRef.current
    if (!gl || !state || !lutData) return

    if (state.lutTexture) gl.deleteTexture(state.lutTexture)
    state.lutTexture = createLUT3DTexture(gl, lutData, lutSize)
    render()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lutData, lutSize, isReady])

  // Re-render when intensity changes
  useEffect(() => {
    if (isReady) render()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intensity, isReady])

  const render = useCallback(() => {
    const gl = glRef.current
    const state = stateRef.current
    if (!gl || !state || !state.imageTexture) return

    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(() => {
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight)
      gl.clearColor(0, 0, 0, 0)
      gl.clear(gl.COLOR_BUFFER_BIT)

      gl.useProgram(state.program)
      gl.bindVertexArray(state.vao)

      // Bind image texture to unit 0
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, state.imageTexture)
      gl.uniform1i(gl.getUniformLocation(state.program, 'u_image'), 0)

      // Bind LUT texture to unit 1 (or use identity if not loaded)
      gl.activeTexture(gl.TEXTURE1)
      if (state.lutTexture) {
        gl.bindTexture(gl.TEXTURE_3D, state.lutTexture)
      }
      gl.uniform1i(gl.getUniformLocation(state.program, 'u_lut'), 1)
      gl.uniform1f(gl.getUniformLocation(state.program, 'u_intensity'), state.lutTexture ? intensity : 0)
      gl.uniform1i(gl.getUniformLocation(state.program, 'u_lutSize'), lutSize)

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
      gl.bindVertexArray(null)
    })
  }, [intensity, lutSize])

  if (error) {
    return (
      <div
        className={cn(
          'flex items-center justify-center rounded-xl bg-gray-100 dark:bg-gray-800 text-center p-8',
          className
        )}
        role="alert"
      >
        <div className="space-y-2">
          <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Preview unavailable</p>
          <p className="text-xs text-gray-500 dark:text-gray-400">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className={cn('relative rounded-xl overflow-hidden bg-gray-900', className)}>
      <canvas
        ref={canvasRef}
        className="w-full h-full object-contain"
        aria-label="Filtered image preview"
      />
      {!imageUrl && (
        <div className="absolute inset-0 flex items-center justify-center">
          <p className="text-sm text-gray-400">Upload an image to see preview</p>
        </div>
      )}
    </div>
  )
}
