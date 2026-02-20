/**
 * Load a LUT as a WebGL 3D texture.
 * Input data: flattened Float32 RGB array, length = size^3 * 3
 */
export function createLUT3DTexture(
  gl: WebGL2RenderingContext,
  data: Float32Array,
  size: number
): WebGLTexture {
  const texture = gl.createTexture()
  if (!texture) throw new Error('Failed to create 3D texture')

  gl.bindTexture(gl.TEXTURE_3D, texture)
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE)

  gl.texImage3D(
    gl.TEXTURE_3D,
    0,
    gl.RGB32F,
    size,
    size,
    size,
    0,
    gl.RGB,
    gl.FLOAT,
    data
  )

  gl.bindTexture(gl.TEXTURE_3D, null)
  return texture
}

/**
 * Create a WebGL 2D texture from an HTMLImageElement or ImageBitmap.
 */
export function createImageTexture(
  gl: WebGL2RenderingContext,
  source: HTMLImageElement | ImageBitmap
): WebGLTexture {
  const texture = gl.createTexture()
  if (!texture) throw new Error('Failed to create image texture')

  gl.bindTexture(gl.TEXTURE_2D, texture)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source)
  gl.bindTexture(gl.TEXTURE_2D, null)

  return texture
}

/**
 * Cache LUT binary data in IndexedDB, keyed by styleId + size + weights.
 */
export async function cacheLUT(key: string, buffer: ArrayBuffer): Promise<void> {
  const db = await openLUTCache()
  return new Promise((resolve, reject) => {
    const tx = db.transaction('luts', 'readwrite')
    tx.objectStore('luts').put({ key, buffer, cachedAt: Date.now() })
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

export async function getCachedLUT(key: string): Promise<ArrayBuffer | null> {
  const db = await openLUTCache()
  return new Promise((resolve, reject) => {
    const tx = db.transaction('luts', 'readonly')
    const req = tx.objectStore('luts').get(key)
    req.onsuccess = () => resolve(req.result?.buffer ?? null)
    req.onerror = () => reject(req.error)
  })
}

function openLUTCache(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('picfilter_lut_cache', 1)
    req.onupgradeneeded = () => {
      req.result.createObjectStore('luts', { keyPath: 'key' })
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}
