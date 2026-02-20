export const VERTEX_SHADER_SRC = `#version 300 es
in vec2 a_position;
in vec2 a_texCoord;
out vec2 v_texCoord;

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  v_texCoord = a_texCoord;
}
`

export const FRAGMENT_SHADER_SRC = `#version 300 es
precision highp float;
precision highp sampler3D;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D u_image;
uniform sampler3D u_lut;
uniform float u_intensity;
uniform int u_lutSize;

vec3 applyLUT(vec3 color) {
  float scale = float(u_lutSize - 1) / float(u_lutSize);
  float offset = 0.5 / float(u_lutSize);
  vec3 lutCoord = color * scale + offset;
  return texture(u_lut, lutCoord).rgb;
}

void main() {
  vec4 original = texture(u_image, v_texCoord);
  vec3 filtered = applyLUT(original.rgb);
  vec3 blended = mix(original.rgb, filtered, u_intensity);
  outColor = vec4(blended, original.a);
}
`

export function compileShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string
): WebGLShader {
  const shader = gl.createShader(type)
  if (!shader) throw new Error('Failed to create shader')
  gl.shaderSource(shader, source)
  gl.compileShader(shader)
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader)
    gl.deleteShader(shader)
    throw new Error(`Shader compile error: ${log}`)
  }
  return shader
}

export function createProgram(
  gl: WebGL2RenderingContext,
  vertSrc: string,
  fragSrc: string
): WebGLProgram {
  const vert = compileShader(gl, gl.VERTEX_SHADER, vertSrc)
  const frag = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc)
  const program = gl.createProgram()
  if (!program) throw new Error('Failed to create program')
  gl.attachShader(program, vert)
  gl.attachShader(program, frag)
  gl.linkProgram(program)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program)
    gl.deleteProgram(program)
    throw new Error(`Program link error: ${log}`)
  }
  gl.deleteShader(vert)
  gl.deleteShader(frag)
  return program
}

// Full-screen quad geometry
export const QUAD_VERTICES = new Float32Array([
  // position (xy), texCoord (uv)
  -1, -1, 0, 1,
   1, -1, 1, 1,
  -1,  1, 0, 0,
   1,  1, 1, 0,
])
