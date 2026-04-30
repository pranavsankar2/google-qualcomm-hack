package com.example.qnn_litertlm_gemma

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.example.qnn_litertlm_gemma.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.NumberFormat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG        = "MainActivity"
        private const val REQ_CAMERA = 101

        // Each 3D Gaussian carries 92 float attributes in the full 3DGS
        // representation (position, scale, rotation, opacity, SH coefficients).
        // 92 × 4 bytes = 368 bytes per point.
        private const val BYTES_PER_GAUSSIAN = 92 * 4L

        // RENO paper: ~47.6× compression vs unquantised 3DGS.
        private const val COMPRESSION_RATIO = 47.6f
    }

    private lateinit var binding:        ActivityMainBinding
    private lateinit var pipeline:       ScannerPipeline
    private lateinit var cameraExecutor: ExecutorService

    private var cameraProvider: ProcessCameraProvider? = null

    // ── Scan state ────────────────────────────────────────────────────────────
    @Volatile private var isScanning     = false
    private var totalPoints              = 0L
    private var frameCount               = 0
    private var lastInferenceMs          = 0L

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        pipeline       = ScannerPipeline(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        binding.buttonScan.setOnClickListener   { toggleScan() }
        binding.buttonExport.setOnClickListener { exportScan() }
        binding.buttonExport.isEnabled = false

        loadPipelineAsync()

        if (hasCameraPermission()) startCamera() else requestCameraPermission()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        pipeline.cleanup()
    }

    // ── Pipeline init ─────────────────────────────────────────────────────────

    /**
     * Attempt to load Mobile-GS and RENO models from external storage on a
     * background thread.  Updates the status labels regardless of outcome.
     */
    private fun loadPipelineAsync() {
        lifecycleScope.launch(Dispatchers.IO) {
            val loaded = pipeline.loadModels()
            withContext(Dispatchers.Main) {
                if (loaded) {
                    binding.textBackendPill.text = "LiteRT"
                    binding.statBackend.text     = "Backend: LiteRT (4-thread CPU)"
                    binding.statModels.text      = "Models: Loaded ✓"
                } else {
                    binding.textBackendPill.text = "DEMO"
                    binding.statBackend.text     = "Backend: Demo Mode"
                    binding.statModels.text      =
                        "Push models to:\n${pipeline.modelDir().absolutePath}"
                }
            }
        }
    }

    // ── Scan control ──────────────────────────────────────────────────────────

    private fun toggleScan() {
        isScanning = !isScanning
        if (isScanning) {
            totalPoints  = 0L
            frameCount   = 0
            pipeline.clearAccum()
            binding.buttonExport.isEnabled = false
        } else {
            binding.buttonExport.isEnabled = totalPoints > 0
        }
        updateScanButton()
        updateStatusLabel()
    }

    private fun updateScanButton() {
        binding.buttonScan.text = if (isScanning) "STOP SCAN" else "START SCAN"
    }

    private fun updateStatusLabel() {
        if (isScanning) {
            binding.textScanStatus.text      = "● LIVE"
            binding.textScanStatus.setTextColor(getColor(R.color.brand_secondary))
        } else {
            binding.textScanStatus.text      = "○ READY"
            binding.textScanStatus.setTextColor(getColor(R.color.text_secondary))
        }
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    /**
     * Called from the camera analyser thread; marshals a UI update.
     *
     * Numbers are calculated to match the RENO paper's compression demo:
     *   142 000 pts × 368 B = 52.2 MB uncompressed → ÷ 47.6 ≈ 1.1 MB compressed
     */
    private fun onFrameResult(result: ScanFrame) {
        totalPoints     += result.numPoints
        frameCount++
        lastInferenceMs  = result.inferenceMs
        runOnUiThread    { refreshStatsPanel() }
    }

    private fun refreshStatsPanel() {
        val fmt          = NumberFormat.getNumberInstance()
        val uncompressed = totalPoints * BYTES_PER_GAUSSIAN
        val compressed   = (uncompressed / COMPRESSION_RATIO).toLong()

        binding.statPoints.text       = fmt.format(totalPoints)
        binding.statUncompressed.text = formatBytes(uncompressed)
        binding.statCompressed.text   = formatBytes(compressed)
        binding.statRatio.text        = String.format("%.1fx", COMPRESSION_RATIO)
        binding.statInference.text    = "${lastInferenceMs}ms"

        updateStatusLabel()
    }

    private fun formatBytes(bytes: Long): String = when {
        bytes >= 1_000_000L -> String.format("%.1f MB", bytes / 1_000_000.0)
        bytes >= 1_000L     -> String.format("%.1f KB", bytes / 1_000.0)
        else                -> "$bytes B"
    }

    // ── Export ────────────────────────────────────────────────────────────────

    /**
     * Stop the scan (if running), write scan_001.reno, and launch the Android
     * share sheet so the user can e-mail the file to their laptop.
     */
    private fun exportScan() {
        if (isScanning) {
            isScanning = false
            updateScanButton()
            updateStatusLabel()
        }
        lifecycleScope.launch(Dispatchers.IO) {
            val file = pipeline.saveRenoFile()
            withContext(Dispatchers.Main) { shareFile(file) }
        }
    }

    private fun shareFile(file: File) {
        val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
        val fmt = NumberFormat.getNumberInstance()
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "application/octet-stream"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, "3D Scan Export — ${file.name}")
            putExtra(
                Intent.EXTRA_TEXT,
                "Scene: ${fmt.format(totalPoints)} Gaussians\n" +
                "Uncompressed: ${formatBytes(totalPoints * BYTES_PER_GAUSSIAN)}\n" +
                "RENO compressed: ${formatBytes((totalPoints * BYTES_PER_GAUSSIAN / COMPRESSION_RATIO).toLong())}\n\n" +
                "Decode with:  python decode_reno.py scan_001.reno"
            )
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, "Export to Cloud Workstation"))
    }

    // ── Camera ────────────────────────────────────────────────────────────────

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED

    private fun requestCameraPermission() =
        ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_CAMERA &&
            grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            binding.textScanStatus.text = "Camera permission required"
        }
    }

    private fun startCamera() {
        ProcessCameraProvider.getInstance(this).also { future ->
            future.addListener({
                cameraProvider = future.get()
                bindCameraUseCases()
            }, ContextCompat.getMainExecutor(this))
        }
    }

    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: return

        // Preview — feeds the PreviewView on the left half of the screen
        val preview = Preview.Builder()
            .setTargetResolution(Size(640, 480))
            .build()
            .also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }

        // Analysis — delivers YUV frames to the pipeline
        val analysis = ImageAnalysis.Builder()
            .setTargetResolution(Size(640, 480))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        analysis.setAnalyzer(cameraExecutor) { proxy ->
            if (isScanning) {
                val bmp    = proxy.toBitmap()
                val result = if (pipeline.isLoaded)
                    pipeline.processFrame(bmp)
                else
                    pipeline.simulateFrame()
                onFrameResult(result)
            }
            proxy.close()
        }

        try {
            provider.unbindAll()
            provider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )
            Log.i(TAG, "Camera bound — preview + analysis active")
        } catch (e: Exception) {
            Log.e(TAG, "Camera bind failed: ${e.message}", e)
        }
    }
}
