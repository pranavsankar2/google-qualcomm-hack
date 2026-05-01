package com.civilscan.nerf3d.pipeline

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.nio.channels.FileChannel

class ModelManager(private val context: Context) {

    companion object {
        private const val TAG     = "ModelManager"
        // MiDaS v2.1 Small — monocular depth estimator
        //   in : [1, 256, 256, 3] float32 ImageNet-normalised
        //   out: [1, 256, 256, 1] float32 inverse relative depth
        const val MODEL_DEPTH = "mobile_gs_quant.tflite"   // MiDaS depth estimator
        const val MODEL_RENO  = "reno_quant.tflite"         // EfficientNet encoder
    }

    data class LoadedModels(
        val gs:           Interpreter?,   // depth (MiDaS)
        val reno:         Interpreter?,   // encoder (EfficientNet)
        val npuActive:    Boolean,
        val delegateType: String
    )

    fun loadModels(): LoadedModels {
        val dir = File(context.getExternalFilesDir(null), "models").also { it.mkdirs() }
        Log.i(TAG, "Models dir: ${dir.absolutePath}  canRead=${dir.canRead()}")

        val opts = Interpreter.Options().apply { numThreads = 4 }
        val gs   = load(File(dir, MODEL_DEPTH), opts)
        val reno = load(File(dir, MODEL_RENO),  opts)

        Log.i(TAG, "MiDaS=${gs != null}  RENO=${reno != null}")
        return LoadedModels(gs, reno, false, "CPU")
    }

    private fun load(file: File, opts: Interpreter.Options): Interpreter? {
        if (!file.exists()) {
            Log.w(TAG, "Missing: ${file.name}  path=${file.absolutePath}")
            return null
        }
        return runCatching {
            val buf = FileInputStream(file).channel
                .map(FileChannel.MapMode.READ_ONLY, 0, file.length())
            Interpreter(buf, opts).also { interp ->
                val inp = interp.getInputTensor(0).shape().toList()
                val out = interp.getOutputTensor(0).shape().toList()
                Log.i(TAG, "✓ ${file.name}  (${file.length() / 1024} KB)  in=$inp  out=$out")
            }
        }.getOrElse { e ->
            Log.e(TAG, "✗ ${file.name}: ${e.message}")
            null
        }
    }

    fun close() {}
}
