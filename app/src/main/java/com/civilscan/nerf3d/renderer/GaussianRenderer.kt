package com.civilscan.nerf3d.renderer

import android.content.Context
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import com.civilscan.nerf3d.data.GaussianPoint
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.*

/**
 * OpenGL ES 3.0 point-cloud viewer for the accumulated Gaussian point cloud.
 *
 * Rendering: instanced point sprites with distance-attenuated size and circular
 *            alpha discard in the fragment shader.
 * Camera:    orbit (one-finger drag) + pinch-zoom + two-finger pan.
 * Colour modes: RGB | Depth | Density heat-map | Anomaly (opacity < 0.3 = red).
 */
class GaussianRenderer(context: Context) : GLSurfaceView.Renderer {

    enum class ColorMode { RGB, DEPTH, ANOMALY }

    // ── GL objects ────────────────────────────────────────────────────────────

    private var program     = 0
    private var vao         = 0
    private var vbo         = 0
    private var pointCount  = 0

    // ── Camera ────────────────────────────────────────────────────────────────

    private var orbitR     = 30f
    private var orbitTheta = Math.toRadians(25.0).toFloat()  // elevation
    private var orbitPhi   = 0f                               // azimuth
    private var panX       = 0f
    private var panY       = 0f

    private val proj  = FloatArray(16)
    private val view  = FloatArray(16)
    private val mvp   = FloatArray(16)
    private val tmp   = FloatArray(16)

    // Centre of the scene
    private var cx = 0f; private var cy = 0f; private var cz = 0f

    // ── State ─────────────────────────────────────────────────────────────────

    @Volatile var colorMode: ColorMode = ColorMode.RGB
    @Volatile private var pendingGaussians: List<GaussianPoint>? = null
    private var width = 1; private var height = 1

    // ── Shaders ───────────────────────────────────────────────────────────────

    private val vertSrc = """
        #version 300 es
        in vec3 aPos;
        in vec4 aColor;
        out vec4 vColor;
        uniform mat4 uMVP;
        uniform float uPointSize;
        void main() {
            gl_Position = uMVP * vec4(aPos, 1.0);
            gl_PointSize = uPointSize / max(gl_Position.w, 0.1);
            vColor = aColor;
        }
    """.trimIndent()

    private val fragSrc = """
        #version 300 es
        precision mediump float;
        in vec4 vColor;
        out vec4 fragColor;
        void main() {
            vec2 coord = gl_PointCoord - 0.5;
            float d = length(coord);
            if (d > 0.5) discard;
            float alpha = 1.0 - smoothstep(0.3, 0.5, d);
            fragColor = vec4(vColor.rgb, vColor.a * alpha);
        }
    """.trimIndent()

    // ── GLSurfaceView.Renderer ────────────────────────────────────────────────

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES30.glClearColor(0.051f, 0.067f, 0.090f, 1f)  // #0D1117
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)
        // GL_PROGRAM_POINT_SIZE does not exist in OpenGL ES — gl_PointSize in the
        // vertex shader is always respected without needing an explicit glEnable call.

        program = buildProgram(vertSrc, fragSrc)
        val ids = IntArray(2)
        GLES30.glGenVertexArrays(1, ids, 0); vao = ids[0]
        GLES30.glGenBuffers(1, ids, 1);      vbo = ids[1]
    }

    override fun onSurfaceChanged(gl: GL10?, w: Int, h: Int) {
        GLES30.glViewport(0, 0, w, h)
        width = w; height = h
        val aspect = w.toFloat() / h.coerceAtLeast(1)
        Matrix.perspectiveM(proj, 0, 60f, aspect, 0.05f, 500f)
    }

    override fun onDrawFrame(gl: GL10?) {
        // Upload new data if available
        pendingGaussians?.let { upload(it); pendingGaussians = null }

        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)
        if (pointCount == 0) return

        buildView()
        Matrix.multiplyMM(mvp, 0, proj, 0, view, 0)

        GLES30.glUseProgram(program)
        GLES30.glUniformMatrix4fv(GLES30.glGetUniformLocation(program, "uMVP"),  1, false, mvp,  0)
        GLES30.glUniform1f(        GLES30.glGetUniformLocation(program, "uPointSize"), 18f)

        GLES30.glBindVertexArray(vao)
        GLES30.glDrawArrays(GLES30.GL_POINTS, 0, pointCount)
        GLES30.glBindVertexArray(0)
    }

    // ── Public interface ──────────────────────────────────────────────────────

    fun setGaussians(gaussians: List<GaussianPoint>) { pendingGaussians = gaussians }

    // ── Camera gesture support (call from GLSurfaceView.onTouchEvent) ─────────

    private var lastX = 0f; private var lastY = 0f; private var pointerCount = 0

    fun onTouch(event: MotionEvent) {
        scaleDetector.onTouchEvent(event)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_POINTER_DOWN -> {
                lastX = event.x; lastY = event.y; pointerCount = event.pointerCount
            }
            MotionEvent.ACTION_MOVE -> {
                val dx = event.x - lastX; val dy = event.y - lastY
                if (event.pointerCount == 1) {
                    orbitPhi   -= dx * 0.005f
                    orbitTheta  = (orbitTheta + dy * 0.005f).coerceIn(-PI_HALF + 0.05f, PI_HALF - 0.05f)
                } else if (event.pointerCount == 2) {
                    panX -= dx * orbitR * 0.001f
                    panY += dy * orbitR * 0.001f
                }
                lastX = event.x; lastY = event.y
            }
        }
    }

    private val PI_HALF = (Math.PI / 2.0).toFloat()

    private val scaleDetector = ScaleGestureDetector(context,
        object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(d: ScaleGestureDetector): Boolean {
                orbitR = (orbitR / d.scaleFactor).coerceIn(1f, 200f)
                return true
            }
        })

    // ── Upload to GPU ─────────────────────────────────────────────────────────

    private fun upload(gaussians: List<GaussianPoint>) {
        // Compute scene centre
        cx = gaussians.sumOf { it.x.toDouble() }.toFloat() / gaussians.size.coerceAtLeast(1)
        cy = gaussians.sumOf { it.y.toDouble() }.toFloat() / gaussians.size.coerceAtLeast(1)
        cz = gaussians.sumOf { it.z.toDouble() }.toFloat() / gaussians.size.coerceAtLeast(1)

        // Depth range for colour mapping
        val minZ = gaussians.minOfOrNull { it.z } ?: 0f
        val maxZ = gaussians.maxOfOrNull { it.z } ?: 1f
        val dz   = (maxZ - minZ).coerceAtLeast(0.001f)

        // Interleaved: x y z r g b a  (7 floats per point)
        val stride = 7
        val buf: FloatBuffer = ByteBuffer
            .allocateDirect(gaussians.size * stride * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()

        for (g in gaussians) {
            buf.put(g.x); buf.put(g.y); buf.put(g.z)
            when (colorMode) {
                ColorMode.RGB -> { buf.put(g.r); buf.put(g.g); buf.put(g.b); buf.put(g.opacity) }
                ColorMode.DEPTH -> {
                    val t = (g.z - minZ) / dz
                    buf.put(t); buf.put(1f - t); buf.put(0.5f); buf.put(1f)
                }
                ColorMode.ANOMALY -> {
                    if (g.opacity < 0.3f) { buf.put(1f); buf.put(0.2f); buf.put(0.1f); buf.put(1f) }
                    else { buf.put(g.r); buf.put(g.g); buf.put(g.b); buf.put(g.opacity) }
                }
            }
        }
        buf.position(0)

        GLES30.glBindVertexArray(vao)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, gaussians.size * stride * 4, buf, GLES30.GL_DYNAMIC_DRAW)

        val posLoc   = GLES30.glGetAttribLocation(program, "aPos")
        val colorLoc = GLES30.glGetAttribLocation(program, "aColor")
        val byteStride = stride * 4

        GLES30.glEnableVertexAttribArray(posLoc)
        GLES30.glVertexAttribPointer(posLoc, 3, GLES30.GL_FLOAT, false, byteStride, 0)
        GLES30.glEnableVertexAttribArray(colorLoc)
        GLES30.glVertexAttribPointer(colorLoc, 4, GLES30.GL_FLOAT, false, byteStride, 12)

        GLES30.glBindVertexArray(0)
        pointCount = gaussians.size
    }

    // ── Build view matrix ─────────────────────────────────────────────────────

    private fun buildView() {
        val ex = cx + panX + orbitR * cos(orbitTheta) * sin(orbitPhi)
        val ey = cy + panY + orbitR * sin(orbitTheta)
        val ez = cz +        orbitR * cos(orbitTheta) * cos(orbitPhi)
        Matrix.setLookAtM(view, 0, ex, ey, ez, cx + panX, cy + panY, cz, 0f, 1f, 0f)
    }

    // ── Shader compilation ────────────────────────────────────────────────────

    private fun buildProgram(vert: String, frag: String): Int {
        val vs = compileShader(GLES30.GL_VERTEX_SHADER,   vert)
        val fs = compileShader(GLES30.GL_FRAGMENT_SHADER, frag)
        return GLES30.glCreateProgram().also { p ->
            GLES30.glAttachShader(p, vs)
            GLES30.glAttachShader(p, fs)
            GLES30.glLinkProgram(p)
        }
    }

    private fun compileShader(type: Int, src: String): Int =
        GLES30.glCreateShader(type).also { s ->
            GLES30.glShaderSource(s, src)
            GLES30.glCompileShader(s)
        }
}
