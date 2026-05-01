package com.civilscan.nerf3d.pipeline

import android.content.Context
import android.os.Build
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.channels.FileChannel

/**
 * Loads LiteRT (TFLite) models and enables the best available NPU path:
 *
 *   API 27+  →  setUseNNAPI(true)  →  Android NNAPI  →  Qualcomm Hexagon HTP
 *   Fallback →  4-thread CPU
 *
 * Models live in <externalFilesDir>/models/.
 * Push with:  bash scripts/download_models.sh
 */
class ModelManager(private val context: Context) {

    companion object {
        private const val TAG        = "ModelManager"
        const val MODEL_GS           = "mobile_gs_quant.tflite"
        const val MODEL_RENO         = "reno_quant.tflite"
        const val MODEL_ANALYSIS     = "civil_analysis_quant.tflite"
    }

    data class LoadedModels(
        val gs:           Interpreter?,
        val reno:         Interpreter?,
        val analysis:     Interpreter?,
        val npuActive:    Boolean,
        val delegateType: String
    )

    fun loadModels(): LoadedModels {
        val dir = File(context.getExternalFilesDir(null), "models").also { it.mkdirs() }

        // NNAPI is available from API 27 — routes to Hexagon HTP on Snapdragon 8 Elite
        val nnApiAvailable = Build.VERSION.SDK_INT >= Build.VERSION_CODES.P

        val opts = Interpreter.Options().apply {
            numThreads = 4
            if (nnApiAvailable) setUseNNAPI(true)
        }

        val gs       = load(File(dir, MODEL_GS),       opts)
        val reno     = load(File(dir, MODEL_RENO),     opts)
        val analysis = load(File(dir, MODEL_ANALYSIS), opts)

        val delegate = if (nnApiAvailable) "NNAPI/HTP" else "CPU"
        Log.i(TAG, "$delegate  GS=${gs != null}  RENO=${reno != null}  Analysis=${analysis != null}")
        return LoadedModels(gs, reno, analysis, nnApiAvailable, delegate)
    }

    private fun load(file: File, opts: Interpreter.Options): Interpreter? {
        if (!file.exists()) { Log.w(TAG, "Missing: ${file.name}"); return null }
        return runCatching {
            val buf = FileInputStream(file).channel
                .map(FileChannel.MapMode.READ_ONLY, 0, file.length())
            Interpreter(buf, opts).also { Log.i(TAG, "Loaded ${file.name}") }
        }.getOrElse { e -> Log.e(TAG, "Failed ${file.name}: ${e.message}"); null }
    }

    fun close() { /* delegates managed internally by Interpreter */ }
}
