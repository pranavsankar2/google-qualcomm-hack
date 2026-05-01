package com.civilscan.nerf3d

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.civilscan.nerf3d.data.*
import com.civilscan.nerf3d.export.ExportManager
import com.civilscan.nerf3d.pipeline.ScannerPipeline
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.util.concurrent.atomic.AtomicBoolean

class ScanViewModel(app: Application) : AndroidViewModel(app) {

    private val pipeline = ScannerPipeline(app)
    val export           = ExportManager(app)

    // ── State ─────────────────────────────────────────────────────────────────

    private val _state    = MutableStateFlow<ScanState>(ScanState.Idle)
    val scanState: StateFlow<ScanState> = _state.asStateFlow()

    private val _gaussians = MutableStateFlow<List<GaussianPoint>>(emptyList())
    val gaussians: StateFlow<List<GaussianPoint>> = _gaussians.asStateFlow()

    private val _shards = MutableStateFlow<List<ExportedFile>>(emptyList())
    val shards: StateFlow<List<ExportedFile>> = _shards.asStateFlow()

    val isSimulated:  Boolean get() = pipeline.isSimulated
    val npuActive:    Boolean get() = pipeline.npuActive
    val midasLoaded:  Boolean = !pipeline.isSimulated   // fixed at init time, no StateFlow needed
    val renoLoaded:   Boolean = pipeline.renoLoaded

    private val _depthBitmap = MutableStateFlow<Bitmap?>(null)
    val depthBitmap: StateFlow<Bitmap?> = _depthBitmap.asStateFlow()

    // ── Sensors ───────────────────────────────────────────────────────────────

    private val sensorManager = app.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val rotSensor  = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
    private val accSensor  = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
    @Volatile private var latestR:   FloatArray? = null
    @Volatile private var latestAcc: FloatArray? = null

    private val sensorListener = object : SensorEventListener {
        private val mat = FloatArray(9)
        override fun onSensorChanged(e: SensorEvent) {
            when (e.sensor.type) {
                Sensor.TYPE_GAME_ROTATION_VECTOR -> {
                    SensorManager.getRotationMatrixFromVector(mat, e.values)
                    latestR = mat.copyOf()
                }
                Sensor.TYPE_LINEAR_ACCELERATION -> latestAcc = e.values.copyOf()
            }
        }
        override fun onAccuracyChanged(s: Sensor?, a: Int) {}
    }

    init {
        sensorManager.registerListener(sensorListener, rotSensor, SensorManager.SENSOR_DELAY_GAME)
        accSensor?.let { sensorManager.registerListener(sensorListener, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    // ── Controls ──────────────────────────────────────────────────────────────

    /** Start scanning from idle, or resume from paused. */
    fun start() {
        when (val s = _state.value) {
            is ScanState.Idle -> {
                pipeline.reset()
                _gaussians.value = emptyList()
                _shards.value    = emptyList()
                framesSinceGlUpdate = 0
                _state.value = ScanState.Running()
            }
            is ScanState.Paused -> {
                _state.value = ScanState.Running(
                    pointCount  = s.pointCount,
                    shardCount  = s.shardCount
                )
            }
            else -> {}
        }
    }

    /** Pause mid-scan. If a shard is in progress let it finish first. */
    fun pause() {
        val s = _state.value
        if (s is ScanState.Running) {
            _state.value = ScanState.Paused(s.pointCount, s.shardCount)
        } else if (s is ScanState.Sharding) {
            pauseAfterShard = true
        }
    }

    /** Clear everything and start a brand-new recording from zero. */
    fun newRecording() {
        pauseAfterShard = false
        pipeline.reset()
        _gaussians.value = emptyList()
        _shards.value    = emptyList()
        framesSinceGlUpdate = 0
        _state.value = ScanState.Running()
    }

    // ── Frame processing ──────────────────────────────────────────────────────

    private var framesSinceGlUpdate = 0
    @Volatile private var pauseAfterShard = false
    private val processing = AtomicBoolean(false)

    fun onFrame(image: ImageProxy) {
        val bmp = image.toBitmap(); image.close()
        if (bmp == null || _state.value !is ScanState.Running) return
        // Drop frame if the pipeline is still working on the previous one
        if (!processing.compareAndSet(false, true)) return
        val R   = latestR
        val acc = latestAcc
        // Rotate linear acceleration from device frame to world frame: a_world = R * a_device
        val accWorld = if (R != null && acc != null) floatArrayOf(
            R[0]*acc[0] + R[1]*acc[1] + R[2]*acc[2],
            R[3]*acc[0] + R[4]*acc[1] + R[5]*acc[2],
            R[6]*acc[0] + R[7]*acc[1] + R[8]*acc[2]
        ) else null

        viewModelScope.launch(Dispatchers.Default) {
          try {
            val result  = pipeline.processFrame(bmp, R, accWorld)
            val current = _state.value as? ScanState.Running ?: return@launch
            val pts     = pipeline.getGaussians().size
            framesSinceGlUpdate++
            _depthBitmap.value = pipeline.getDepthBitmap()

            if (pts >= ScannerPipeline.SHARD_MAX_POINTS) {
                // ── Auto-shard ────────────────────────────────────────────────
                val idx   = current.shardCount + 1
                _state.value = ScanState.Sharding(idx, resumeAfter = !pauseAfterShard)

                val scan     = pipeline.compressAndFinalize()
                val exported = export.exportReno(scan, "shard_$idx")
                _shards.value = _shards.value + exported

                pipeline.resetAccumulationOnly()
                framesSinceGlUpdate = 0
                _gaussians.value = emptyList()

                if (pauseAfterShard) {
                    pauseAfterShard = false
                    _state.value = ScanState.Paused(0, idx)
                } else {
                    _state.value = ScanState.Running(
                        pointCount  = 0,
                        shardCount  = idx,
                        inferenceMs = result.inferenceMs,
                        isSimulated = result.isSimulated
                    )
                }
            } else {
                _state.value = current.copy(
                    pointCount  = pts,
                    inferenceMs = result.inferenceMs,
                    isSimulated = result.isSimulated
                )
                if (framesSinceGlUpdate % 6 == 0) {
                    _gaussians.value = pipeline.getGaussians()
                }
            }
          } finally {
            processing.set(false)
          }
        }
    }

    override fun onCleared() {
        super.onCleared()
        sensorManager.unregisterListener(sensorListener)
        pipeline.close()
    }

    // ── ImageProxy → Bitmap ───────────────────────────────────────────────────

    private fun ImageProxy.toBitmap(): Bitmap? = runCatching {
        val yB = planes[0].buffer; val uB = planes[1].buffer; val vB = planes[2].buffer
        val yL = yB.remaining(); val uL = uB.remaining(); val vL = vB.remaining()
        val nv21 = ByteArray(yL + uL + vL)
        yB.get(nv21, 0, yL); vB.get(nv21, yL, vL); uB.get(nv21, yL + vL, uL)
        val yuv = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, width, height), 80, out)
        BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
    }.getOrNull()
}
