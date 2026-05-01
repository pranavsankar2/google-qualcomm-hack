package com.civilscan.nerf3d.pipeline

import android.content.Context
import android.os.Build
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File

class ModelManager(private val context: Context) {

    companion object {
        private const val TAG     = "ModelManager"
        const val MODEL_DEPTH = "mobile_gs_quant.tflite"
        const val MODEL_RENO  = "reno_quant.tflite"
    }

    data class LoadedModels(
        val gs:           CompiledModel?,
        val reno:         CompiledModel?,
        val npuActive:    Boolean,
        val delegateType: String
    )

    fun loadModels(): LoadedModels {
        val dir = File(context.getExternalFilesDir(null), "models").also { it.mkdirs() }
        Log.i(TAG, "Models dir: ${dir.absolutePath}  canRead=${dir.canRead()}")
        Log.i(TAG, "Device: ${Build.MANUFACTURER} ${Build.MODEL}  SoC=${Build.SOC_MODEL ?: "unknown"}")

        val depthFile = File(dir, MODEL_DEPTH)
        val renoFile  = File(dir, MODEL_RENO)

        // Try accelerators in order of preference. LiteRT 2.x bypasses NNAPI
        // entirely — NPU here means real Hexagon HTP via Google AI Edge stack.
        val preferences = listOf(Accelerator.NPU, Accelerator.GPU, Accelerator.CPU)
        for (accel in preferences) {
            val gs = loadOn(depthFile, accel) ?: continue
            val reno = loadOn(renoFile, accel)
            val npuActive = (accel == Accelerator.NPU)
            Log.i(TAG, "✓ Models loaded with accelerator=$accel  RENO=${reno != null}")
            return LoadedModels(gs, reno, npuActive, accel.toString())
        }

        Log.e(TAG, "All accelerators failed — model files present? depth=${depthFile.exists()} reno=${renoFile.exists()}")
        return LoadedModels(null, null, npuActive = false, delegateType = "FAILED")
    }

    private fun loadOn(file: File, accel: Accelerator): CompiledModel? {
        if (!file.exists()) {
            Log.w(TAG, "Missing: ${file.name}  path=${file.absolutePath}")
            return null
        }
        return runCatching {
            CompiledModel.create(file.absolutePath, CompiledModel.Options(accel)).also {
                Log.i(TAG, "✓ ${file.name}  on=$accel  (${file.length() / 1024} KB)")
            }
        }.getOrElse { e ->
            Log.w(TAG, "✗ ${file.name} on $accel: ${e.message}")
            null
        }
    }

    fun close() {
        // CompiledModel resources are released by the runtime when no longer referenced.
    }
}
