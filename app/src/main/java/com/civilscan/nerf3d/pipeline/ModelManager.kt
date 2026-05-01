package com.civilscan.nerf3d.pipeline

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.io.FileInputStream
import java.nio.channels.FileChannel

class ModelManager(private val context: Context) {

    companion object {
        private const val TAG     = "ModelManager"
        const val MODEL_DEPTH = "mobile_gs_quant.tflite"   // MiDaS depth estimator
        const val MODEL_RENO  = "reno_quant.tflite"         // EfficientNet encoder
    }

    data class LoadedModels(
        val gs:           Interpreter?,   // depth (MiDaS)
        val reno:         Interpreter?,   // encoder (EfficientNet)
        val npuActive:    Boolean,
        val delegateType: String
    )

    private var nnApiDelegate: NnApiDelegate? = null

    fun loadModels(): LoadedModels {
        val dir = File(context.getExternalFilesDir(null), "models").also { it.mkdirs() }
        Log.i(TAG, "Models dir: ${dir.absolutePath}  canRead=${dir.canRead()}")

        // Attempt 1: NNAPI → Qualcomm Hexagon HTP / NPU on Snapdragon
        val nnApi = tryNnApi()
        if (nnApi != null) {
            val npuOpts = Interpreter.Options().apply { addDelegate(nnApi) }
            val gsNpu   = load(File(dir, MODEL_DEPTH), npuOpts)
            if (gsNpu != null) {
                nnApiDelegate = nnApi
                val renoNpu = load(File(dir, MODEL_RENO), npuOpts)
                Log.i(TAG, "NPU active via NNAPI  MiDaS=true  RENO=${renoNpu != null}")
                return LoadedModels(gsNpu, renoNpu, npuActive = true, delegateType = "NPU")
            }
            Log.w(TAG, "NNAPI available but model failed on NPU — falling back to CPU")
            nnApi.close()
        }

        // Fallback: CPU with 4 threads
        val cpuOpts = Interpreter.Options().apply { numThreads = 4 }
        val gs   = load(File(dir, MODEL_DEPTH), cpuOpts)
        val reno = load(File(dir, MODEL_RENO),  cpuOpts)
        Log.i(TAG, "CPU mode  MiDaS=${gs != null}  RENO=${reno != null}")
        return LoadedModels(gs, reno, npuActive = false, delegateType = "CPU")
    }

    private fun tryNnApi(): NnApiDelegate? = runCatching {
        NnApiDelegate(
            NnApiDelegate.Options()
                .setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
        ).also { Log.i(TAG, "NnApiDelegate created") }
    }.getOrElse { e ->
        Log.w(TAG, "NNAPI unavailable: ${e.message}")
        null
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

    fun close() {
        nnApiDelegate?.close()
        nnApiDelegate = null
    }
}
