package com.civilscan.nerf3d.pipeline

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.civilscan.nerf3d.data.*
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*
import kotlin.random.Random

/**
 * End-to-end 3D scanning pipeline for the Samsung Galaxy S25 Ultra NPU.
 *
 * Uses two plug-and-play TFLite models (download via scripts/download_models.sh):
 *
 *   Model 1 — MiDaS v2.1 small  (mobile_gs_quant.tflite)
 *     Input  : [1, 256, 256, 3]  float32  HWC  ImageNet-normalised
 *     Output : [1, 256, 256, 1]  float32  inverse relative depth (disparity)
 *     Role   : depth estimation → unproject to 3D point cloud per frame
 *
 *   Model 2 — EfficientNet-Lite0 FP32  (reno_quant.tflite)
 *     Input  : [1, 224, 224, 3]  float32  HWC  0–1 range
 *     Output : [1, 1280]         float32  scene feature vector
 *     Role   : scene encoder / neural compression (run once at finalise time)
 *
 * Both models are delegated to the Qualcomm Hexagon HTP via NNAPI.
 * Simulation fallback is active while model files are absent.
 *
 * Push models before running:
 *   bash scripts/download_models.sh
 */
class ScannerPipeline(context: Context) {

    companion object {
        private const val TAG = "ScannerPipeline"

        // ── MiDaS depth model ────────────────────────────────────────────────
        const val DEPTH_W    = 256
        const val DEPTH_H    = 256
        // ImageNet normalisation applied before inference
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD  = floatArrayOf(0.229f, 0.224f, 0.225f)

        // ── EfficientNet-Lite0 encoder ───────────────────────────────────────
        const val ENC_W      = 224
        const val ENC_H      = 224
        const val LATENT_DIM = 1280

        // ── Scene scale / camera intrinsics ──────────────────────────────────
        // Simple pinhole camera for 256×256, ~65° HFOV
        private const val FX = 200f
        private const val FY = 200f
        private const val CX = 128f
        private const val CY = 128f
        private const val MAX_DEPTH_M  = 10f   // scene truncated at 10 m
        private const val MIN_DEPTH_M  = 0.3f

        // Sample every Nth pixel from the depth map (keeps ~1 024 pts/frame)
        private const val SAMPLE_STRIDE = 8

        const val METERS_PER_UNIT = 1f          // units already in metres
    }

    private val manager = ModelManager(context)
    private val models  = manager.loadModels()

    val delegateType: String  get() = models.delegateType
    val npuActive: Boolean    get() = models.npuActive
    val isSimulated: Boolean  get() = models.gs == null

    // ── Pre-allocated inference buffers ──────────────────────────────────────

    // MiDaS input  [1, 256, 256, 3] float32
    private val depthInputBuf = ByteBuffer
        .allocateDirect(DEPTH_H * DEPTH_W * 3 * 4)
        .apply { order(ByteOrder.nativeOrder()) }

    // MiDaS output [1, 256, 256, 1] float32 — store as flat [256*256]
    private val depthOut = Array(1) { Array(DEPTH_H) { Array(DEPTH_W) { FloatArray(1) } } }

    // EfficientNet input  [1, 224, 224, 3] float32
    private val encInputBuf = ByteBuffer
        .allocateDirect(ENC_H * ENC_W * 3 * 4)
        .apply { order(ByteOrder.nativeOrder()) }

    // EfficientNet output [1, 1280] float32
    private val encOut = Array(1) { FloatArray(LATENT_DIM) }

    // ── Accumulation state ────────────────────────────────────────────────────

    private val accumulated  = mutableListOf<GaussianPoint>()
    private var frameCount   = 0
    private var lastBitmap: Bitmap? = null    // kept for the encoder step

    // ── Public API ─────────────────────────────────────────────────────────────

    fun reset() { accumulated.clear(); frameCount = 0; lastBitmap = null }

    /** Process one camera frame; accumulates 3D points. Thread-safe. */
    @Synchronized
    fun processFrame(bitmap: Bitmap): FrameResult {
        frameCount++
        lastBitmap = bitmap
        val t0 = System.currentTimeMillis()

        val points = if (!isSimulated) runDepthModel(bitmap)
                     else              simulateGaussians()

        val ms = System.currentTimeMillis() - t0
        accumulated += points
        if (accumulated.size > 300_000) {
            val drop = accumulated.size - 200_000
            repeat(drop) { accumulated.removeAt(0) }
        }
        return FrameResult(points, ms, isSimulated)
    }

    /** Run encoder once, package everything into a CompressedScan. */
    fun compressAndFinalize(): CompressedScan {
        val gs = accumulated.toList()

        val latent = when {
            models.reno != null && lastBitmap != null -> runEncoderModel(lastBitmap!!)
            else -> simulateLatent()
        }

        val bbox     = boundingBox(gs)
        val rawBytes = gs.size.toLong() * 10 * 4         // 10 floats per GaussianPoint
        val ratio    = if (latent.isNotEmpty())
                           rawBytes.toFloat() / (latent.size * 4f)
                       else 47.6f

        Log.i(TAG, "Finalised: ${gs.size} pts, ratio=${ratio.format(1)}×, delegate=$delegateType")
        return CompressedScan(latent, gs.size, bbox, ratio, frameCount, System.currentTimeMillis())
    }

    fun getGaussians(): List<GaussianPoint> = accumulated.toList()

    fun close() { manager.close() }

    // ── MiDaS depth inference ─────────────────────────────────────────────────

    private fun runDepthModel(bitmap: Bitmap): List<GaussianPoint> {
        val scaled = Bitmap.createScaledBitmap(bitmap, DEPTH_W, DEPTH_H, true)
        prepareDepthInput(scaled)

        depthOut[0].forEach { row -> row.forEach { it.fill(0f) } }
        models.gs!!.run(depthInputBuf, depthOut)

        return depthToGaussians(depthOut, scaled)
    }

    private fun prepareDepthInput(bitmap: Bitmap) {
        depthInputBuf.rewind()
        val px = IntArray(DEPTH_W * DEPTH_H)
        bitmap.getPixels(px, 0, DEPTH_W, 0, 0, DEPTH_W, DEPTH_H)
        for (p in px) {
            // ImageNet normalisation
            depthInputBuf.putFloat(((p shr 16 and 0xFF) / 255f - MEAN[0]) / STD[0])
            depthInputBuf.putFloat(((p shr  8 and 0xFF) / 255f - MEAN[1]) / STD[1])
            depthInputBuf.putFloat(((p        and 0xFF) / 255f - MEAN[2]) / STD[2])
        }
        depthInputBuf.rewind()
    }

    private fun depthToGaussians(
        disparity: Array<Array<Array<FloatArray>>>,
        colorBitmap: Bitmap
    ): List<GaussianPoint> {
        val result = mutableListOf<GaussianPoint>()
        val px     = IntArray(DEPTH_W * DEPTH_H)
        colorBitmap.getPixels(px, 0, DEPTH_W, 0, 0, DEPTH_W, DEPTH_H)

        // Find disparity range for this frame (robust percentile approximation)
        var dMin = Float.MAX_VALUE; var dMax = -Float.MAX_VALUE
        for (v in 0 until DEPTH_H step SAMPLE_STRIDE) {
            for (u in 0 until DEPTH_W step SAMPLE_STRIDE) {
                val d = disparity[0][v][u][0]
                if (d < dMin) dMin = d
                if (d > dMax) dMax = d
            }
        }
        val dRange = (dMax - dMin).coerceAtLeast(1e-4f)

        for (v in 0 until DEPTH_H step SAMPLE_STRIDE) {
            for (u in 0 until DEPTH_W step SAMPLE_STRIDE) {
                val disp = disparity[0][v][u][0]

                // MiDaS: larger disparity = closer; normalise and invert to get depth
                val dispNorm = (disp - dMin) / dRange        // 0 (far) … 1 (close)
                val depthM   = (1f - dispNorm) * MAX_DEPTH_M + MIN_DEPTH_M
                if (depthM > MAX_DEPTH_M - 0.1f) continue  // background skip

                // Unproject with pinhole model
                val x = (u - CX) * depthM / FX
                val y = (v - CY) * depthM / FY
                val z = depthM

                // Sample colour from the original frame
                val argb = px[v * DEPTH_W + u]
                val r = (argb shr 16 and 0xFF) / 255f
                val g = (argb shr  8 and 0xFF) / 255f
                val b = (argb        and 0xFF) / 255f

                // Use a modest scale so gaussians don't overlap too much
                val s = depthM * 0.004f

                result += GaussianPoint(x, y, z, 0.88f, s, s, s, r, g, b)
            }
        }
        return result
    }

    // ── EfficientNet-Lite0 encoder ─────────────────────────────────────────────

    private fun runEncoderModel(bitmap: Bitmap): FloatArray {
        val scaled = Bitmap.createScaledBitmap(bitmap, ENC_W, ENC_H, true)
        encInputBuf.rewind()
        val px = IntArray(ENC_W * ENC_H)
        scaled.getPixels(px, 0, ENC_W, 0, 0, ENC_W, ENC_H)
        for (p in px) {
            // EfficientNet-Lite0 expects 0–1 float values
            encInputBuf.putFloat((p shr 16 and 0xFF) / 255f)
            encInputBuf.putFloat((p shr  8 and 0xFF) / 255f)
            encInputBuf.putFloat((p        and 0xFF) / 255f)
        }
        encInputBuf.rewind()
        encOut[0].fill(0f)
        models.reno!!.run(encInputBuf, encOut)
        return encOut[0].clone()
    }

    // ── Simulation fallbacks ───────────────────────────────────────────────────

    private fun simulateGaussians(): List<GaussianPoint> {
        val n = 900 + Random.nextInt(200)   // ~1 024 per frame to match real model
        return List(n) {
            GaussianPoint(
                x = Random.nextFloat() * 20f - 10f,
                y = Random.nextFloat() * 3.5f,
                z = 0.5f + Random.nextFloat() * 9f,
                opacity = 0.8f + Random.nextFloat() * 0.2f,
                scaleX  = 0.03f + Random.nextFloat() * 0.06f,
                scaleY  = 0.03f + Random.nextFloat() * 0.06f,
                scaleZ  = 0.03f + Random.nextFloat() * 0.06f,
                r = 0.72f + Random.nextFloat() * 0.28f,
                g = 0.65f + Random.nextFloat() * 0.28f,
                b = 0.58f + Random.nextFloat() * 0.28f
            )
        }
    }

    private fun simulateLatent() = FloatArray(LATENT_DIM) { Random.nextFloat() * 2f - 1f }

    // ── Utilities ──────────────────────────────────────────────────────────────

    private fun boundingBox(gs: List<GaussianPoint>): BoundingBox {
        if (gs.isEmpty()) return BoundingBox(0f, 1f, 0f, 1f, 0f, 1f)
        var x0 = Float.MAX_VALUE; var x1 = -Float.MAX_VALUE
        var y0 = Float.MAX_VALUE; var y1 = -Float.MAX_VALUE
        var z0 = Float.MAX_VALUE; var z1 = -Float.MAX_VALUE
        for (g in gs) {
            if (g.x < x0) x0 = g.x; if (g.x > x1) x1 = g.x
            if (g.y < y0) y0 = g.y; if (g.y > y1) y1 = g.y
            if (g.z < z0) z0 = g.z; if (g.z > z1) z1 = g.z
        }
        return BoundingBox(x0, x1, y0, y1, z0, z1)
    }

    private fun Float.format(d: Int) = "%.${d}f".format(this)
}
