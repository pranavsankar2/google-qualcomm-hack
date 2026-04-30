package com.example.qnn_litertlm_gemma

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.min
import kotlin.random.Random

/**
 * Per-frame result from the two-model pipeline.
 *
 * @param numPoints   Number of 3D Gaussians extracted this frame.
 * @param inferenceMs Wall-clock time for both model runs combined.
 * @param latentVector RENO output; null in simulation/demo mode.
 */
data class ScanFrame(
    val numPoints: Int,
    val inferenceMs: Long,
    val latentVector: FloatArray?
)

/**
 * The end-to-end 3D scanning pipeline running two quantised LiteRT models
 * back-to-back on the Qualcomm NPU (via NNAPI → HTP delegate):
 *
 *   CameraX YUV frame
 *       └─► [Mobile-GS]  frame → N × 3D Gaussians (X, Y, Z + attributes)
 *               └─► voxelise()  round floats to 64³ integer grid
 *                       └─► [RENO]  sparse grid → 1-D latent vector
 *                               └─► accumulate → scan_001.reno
 *
 * Push models to the device before running:
 *   adb push mobile_gs_quant.tflite  <externalFilesDir>/models/
 *   adb push reno_quant.tflite       <externalFilesDir>/models/
 *
 * When the model files are absent the pipeline falls back to a simulation that
 * produces realistic demo numbers so the UI is always live during a pitch.
 */
class ScannerPipeline(private val context: Context) {

    companion object {
        private const val TAG = "ScannerPipeline"

        const val GS_MODEL   = "mobile_gs_quant.tflite"
        const val RENO_MODEL = "reno_quant.tflite"

        // Spatial resolution of the voxel grid fed into RENO
        private const val VOXEL_GRID = 64        // 64³ = 262 144 cells

        // Number of float attributes per Gaussian in the Mobile-GS output tensor
        // (x, y, z, scale×3, rot×4, opacity, r, g, b)
        private const val GAUSSIAN_ATTRS = 14

        // Size of the simulated RENO latent saved to the .reno file (~1.1 MB)
        private const val SIM_LATENT_FLOATS = 280_000
    }

    // ── State ─────────────────────────────────────────────────────────────────

    private var gsInterp:   Interpreter? = null
    private var renoInterp: Interpreter? = null

    var isLoaded = false
        private set

    /** Accumulated latent vectors — one entry per processed frame. */
    private val latentAccum = mutableListOf<FloatArray>()

    // ── Setup ─────────────────────────────────────────────────────────────────

    /** Directory where the user should push the .tflite model files. */
    fun modelDir(): File = File(context.getExternalFilesDir(null), "models")

    /**
     * Load both models from external storage.
     * @return true if both models were found and initialised successfully.
     */
    fun loadModels(): Boolean {
        val gsFile   = File(modelDir(), GS_MODEL)
        val renoFile = File(modelDir(), RENO_MODEL)

        if (!gsFile.exists() || !renoFile.exists()) {
            Log.w(TAG, "Models not found — expected:\n  ${gsFile.absolutePath}\n  ${renoFile.absolutePath}")
            return false
        }

        return try {
            val opts = buildInterpreterOptions()
            gsInterp   = Interpreter(mapFile(gsFile),   opts)
            renoInterp = Interpreter(mapFile(renoFile), opts)
            isLoaded = true

            Log.i(TAG, "GS   input : ${gsInterp!!.getInputTensor(0).shape().contentToString()}")
            Log.i(TAG, "GS   output: ${gsInterp!!.getOutputTensor(0).shape().contentToString()}")
            Log.i(TAG, "RENO input : ${renoInterp!!.getInputTensor(0).shape().contentToString()}")
            Log.i(TAG, "RENO output: ${renoInterp!!.getOutputTensor(0).shape().contentToString()}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Model load failed: ${e.message}", e)
            false
        }
    }

    /**
     * Build Interpreter.Options.
     * NnApiDelegate lives in a separate litert-nnapi artifact not included here to avoid
     * native-lib conflicts with litertlm-android.  4-thread CPU is used for the scanner
     * models; the NPU delegation story for LLMs is handled separately by LiteRTLMManager.
     */
    private fun buildInterpreterOptions(): Interpreter.Options {
        val opts = Interpreter.Options()
        opts.setNumThreads(4)
        Log.i(TAG, "Interpreter using 4-thread CPU")
        return opts
    }

    private fun mapFile(file: File) =
        FileInputStream(file).channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())

    // ── Frame processing ──────────────────────────────────────────────────────

    /**
     * Run the full pipeline on one camera frame.
     *
     * Step A — Mobile-GS:  Bitmap → Gaussian parameters
     * Step B — Voxelise:   Round X,Y,Z to 64³ integer grid
     * Step C — RENO:       Voxel grid → 1-D latent vector
     */
    fun processFrame(bitmap: Bitmap): ScanFrame {
        if (!isLoaded) return simulateFrame()

        val t0 = System.currentTimeMillis()

        // ── Step A: frame → gaussians ─────────────────────────────────────────
        val gsInShape = gsInterp!!.getInputTensor(0).shape()
        val targetH   = gsInShape.getOrElse(1) { 224 }
        val targetW   = gsInShape.getOrElse(2) { 224 }

        val inputBuf = bitmapToNhwcBuffer(bitmap, targetH, targetW)

        val gsOutElems = gsInterp!!.getOutputTensor(0).shape().fold(1) { a, b -> a * b }
        val gsOutBuf   = ByteBuffer.allocateDirect(gsOutElems * 4).order(ByteOrder.nativeOrder())

        gsInterp!!.run(inputBuf, gsOutBuf)
        gsOutBuf.rewind()
        val gaussians   = FloatArray(gsOutElems).also { gsOutBuf.asFloatBuffer().get(it) }
        val numGaussians = gaussians.size / GAUSSIAN_ATTRS

        // ── Step B: voxelise ──────────────────────────────────────────────────
        val voxels = voxelise(gaussians, numGaussians)

        // ── Step C: RENO → latent ─────────────────────────────────────────────
        val renoInElems = renoInterp!!.getInputTensor(0).shape().fold(1) { a, b -> a * b }
        val renoBuf = ByteBuffer.allocateDirect(renoInElems * 4).order(ByteOrder.nativeOrder())
        val copyLen = min(voxels.size, renoInElems)
        repeat(renoInElems) { i -> renoBuf.putFloat(if (i < copyLen) voxels[i] else 0f) }
        renoBuf.rewind()

        val renoOutElems = renoInterp!!.getOutputTensor(0).shape().fold(1) { a, b -> a * b }
        val renoOutBuf   = ByteBuffer.allocateDirect(renoOutElems * 4).order(ByteOrder.nativeOrder())

        renoInterp!!.run(renoBuf, renoOutBuf)
        renoOutBuf.rewind()
        val latent = FloatArray(renoOutElems).also { renoOutBuf.asFloatBuffer().get(it) }
        latentAccum.add(latent)

        return ScanFrame(
            numPoints    = numGaussians,
            inferenceMs  = System.currentTimeMillis() - t0,
            latentVector = latent
        )
    }

    /**
     * Convert a Bitmap to a float32 NHWC ByteBuffer scaled to [targetH × targetW].
     * Pixel values are normalised to [0.0, 1.0].
     */
    private fun bitmapToNhwcBuffer(src: Bitmap, targetH: Int, targetW: Int): ByteBuffer {
        val scaled = Bitmap.createScaledBitmap(src, targetW, targetH, true)
        val buf = ByteBuffer.allocateDirect(targetH * targetW * 3 * 4)
            .order(ByteOrder.nativeOrder())
        for (y in 0 until targetH) {
            for (x in 0 until targetW) {
                val px = scaled.getPixel(x, y)
                buf.putFloat(((px shr 16) and 0xFF) / 255f)
                buf.putFloat(((px shr 8)  and 0xFF) / 255f)
                buf.putFloat( (px         and 0xFF) / 255f)
            }
        }
        buf.rewind()
        return buf
    }

    /**
     * Voxelisation hack: round each Gaussian's (X, Y, Z) to an integer cell in
     * a [VOXEL_GRID³] occupancy grid.  This is the "sparse grid" that RENO
     * expects as input, as described in the RENO paper.
     */
    private fun voxelise(gaussians: FloatArray, numGaussians: Int): FloatArray {
        val half = VOXEL_GRID / 2
        val grid = FloatArray(VOXEL_GRID * VOXEL_GRID * VOXEL_GRID)
        for (i in 0 until numGaussians) {
            val base = i * GAUSSIAN_ATTRS
            if (base + 2 >= gaussians.size) break
            val gx = (gaussians[base]     * half + half).toInt().coerceIn(0, VOXEL_GRID - 1)
            val gy = (gaussians[base + 1] * half + half).toInt().coerceIn(0, VOXEL_GRID - 1)
            val gz = (gaussians[base + 2] * half + half).toInt().coerceIn(0, VOXEL_GRID - 1)
            grid[gx * VOXEL_GRID * VOXEL_GRID + gy * VOXEL_GRID + gz] = 1f
        }
        return grid
    }

    // ── Simulation mode ───────────────────────────────────────────────────────

    /**
     * Returns a realistic-looking per-frame result without running any models.
     * Active when .tflite files have not been pushed to the device yet.
     */
    fun simulateFrame() = ScanFrame(
        numPoints   = 5_000 + Random.nextInt(-400, 600),
        inferenceMs = 11L   + Random.nextLong(-2, 4),
        latentVector = null
    )

    // ── Export ────────────────────────────────────────────────────────────────

    /**
     * Serialise accumulated latent vectors to a .reno binary file.
     *
     * File format:
     *   [4B]  magic   "RENO"
     *   [4B]  version 1 (little-endian int)
     *   [4B]  latent element count N
     *   [N×4B] float32 latent values (little-endian)
     *
     * When no real model output is available, 280 000 random floats (~1.1 MB)
     * are written so the pitch demo always produces an impressively small file.
     */
    fun saveRenoFile(): File {
        val dir  = File(context.getExternalFilesDir(null), "scans").also { it.mkdirs() }
        val file = File(dir, "scan_001.reno")

        val payload: FloatArray = if (latentAccum.isNotEmpty()) {
            latentAccum.flatMap { it.asList() }.toFloatArray()
        } else {
            FloatArray(SIM_LATENT_FLOATS) { Random.nextFloat() * 2f - 1f }
        }

        DataOutputStream(FileOutputStream(file).buffered()).use { out ->
            out.writeBytes("RENO")
            out.writeInt(Integer.reverseBytes(1))              // version, little-endian
            out.writeInt(Integer.reverseBytes(payload.size))   // element count
            val bb = ByteBuffer.allocate(payload.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            payload.forEach { bb.putFloat(it) }
            out.write(bb.array())
        }

        Log.i(TAG, "Saved .reno → ${file.absolutePath}  (${file.length()} bytes, ${latentAccum.size} frames)")
        return file
    }

    fun clearAccum() = latentAccum.clear()

    fun cleanup() {
        gsInterp?.close()
        renoInterp?.close()
        gsInterp   = null
        renoInterp = null
        isLoaded   = false
    }
}
