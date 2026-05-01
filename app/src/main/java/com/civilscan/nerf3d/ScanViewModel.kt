package com.civilscan.nerf3d

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.civilscan.nerf3d.analysis.CivilAnalysisEngine
import com.civilscan.nerf3d.data.*
import com.civilscan.nerf3d.export.ExportManager
import com.civilscan.nerf3d.pipeline.ScannerPipeline
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream

class ScanViewModel(app: Application) : AndroidViewModel(app) {

    private val pipeline = ScannerPipeline(app)
    private val analysis = CivilAnalysisEngine()
    val export           = ExportManager(app)

    // ── State flows ───────────────────────────────────────────────────────────

    private val _scanState      = MutableStateFlow<ScanState>(ScanState.Idle)
    val scanState: StateFlow<ScanState> = _scanState.asStateFlow()

    private val _analysisResult = MutableStateFlow<AnalysisResult?>(null)
    val analysisResult: StateFlow<AnalysisResult?> = _analysisResult.asStateFlow()

    private val _gaussians      = MutableStateFlow<List<GaussianPoint>>(emptyList())
    val gaussians: StateFlow<List<GaussianPoint>> = _gaussians.asStateFlow()

    private val _exportedFile   = MutableStateFlow<ExportedFile?>(null)
    val exportedFile: StateFlow<ExportedFile?> = _exportedFile.asStateFlow()

    val delegateType: String  get() = pipeline.delegateType
    val npuActive: Boolean    get() = pipeline.npuActive
    val isSimulated: Boolean  get() = pipeline.isSimulated

    // ── Scan lifecycle ────────────────────────────────────────────────────────

    fun startScan() {
        pipeline.reset()
        _analysisResult.value = null
        _exportedFile.value   = null
        _scanState.value = ScanState.Scanning(delegateType = pipeline.delegateType)
    }

    /** Called from CameraX ImageAnalysis on each frame. */
    fun onFrame(image: ImageProxy) {
        val bitmap = image.toBitmap()
        image.close()
        if (bitmap == null || _scanState.value !is ScanState.Scanning) return

        viewModelScope.launch(Dispatchers.Default) {
            val result = pipeline.processFrame(bitmap)
            val current = _scanState.value
            if (current is ScanState.Scanning) {
                _scanState.value = current.copy(
                    gaussianCount = pipeline.getGaussians().size,
                    frameCount    = current.frameCount + 1,
                    inferenceMs   = result.inferenceMs,
                    isSimulated   = result.isSimulated
                )
            }
            // Throttle GL updates: every 8 frames
            val fc = (_scanState.value as? ScanState.Scanning)?.frameCount ?: 0
            if (fc % 8 == 0) _gaussians.value = pipeline.getGaussians().takeLast(50_000)
        }
    }

    fun stopAndAnalyze() {
        if (_scanState.value !is ScanState.Scanning) return
        _scanState.value = ScanState.Compressing(0f)

        viewModelScope.launch(Dispatchers.Default) {
            _scanState.value = ScanState.Compressing(0.3f)
            val scan = pipeline.compressAndFinalize()
            _scanState.value = ScanState.Compressing(0.7f)
            val gaussians = pipeline.getGaussians()
            _gaussians.value = gaussians
            val result = analysis.analyze(scan, gaussians)
            _analysisResult.value  = result
            _scanState.value = ScanState.Done(scan)
        }
    }

    // ── Export ────────────────────────────────────────────────────────────────

    fun exportReno() {
        val scan = (scanState.value as? ScanState.Done)?.scan ?: return
        viewModelScope.launch(Dispatchers.IO) {
            _exportedFile.value = export.exportReno(scan)
        }
    }

    fun exportReport() {
        val scan   = (scanState.value as? ScanState.Done)?.scan    ?: return
        val result = _analysisResult.value                          ?: return
        viewModelScope.launch(Dispatchers.IO) {
            _exportedFile.value = export.exportReport(scan, result)
        }
    }

    fun resetScan() {
        pipeline.reset()
        _scanState.value      = ScanState.Idle
        _analysisResult.value = null
        _gaussians.value      = emptyList()
        _exportedFile.value   = null
    }

    override fun onCleared() { super.onCleared(); pipeline.close() }

    // ── ImageProxy → Bitmap ───────────────────────────────────────────────────

    private fun ImageProxy.toBitmap(): Bitmap? = runCatching {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer
        val yLen    = yBuffer.remaining()
        val uLen    = uBuffer.remaining()
        val vLen    = vBuffer.remaining()
        val nv21    = ByteArray(yLen + uLen + vLen)
        yBuffer.get(nv21, 0, yLen)
        vBuffer.get(nv21, yLen, vLen)
        uBuffer.get(nv21, yLen + vLen, uLen)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out      = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 80, out)
        val bytes = out.toByteArray()
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }.getOrNull()
}
