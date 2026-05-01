package com.civilscan.nerf3d.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.opengl.GLSurfaceView
import android.util.Size
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import android.widget.ImageView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import kotlinx.coroutines.launch
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.civilscan.nerf3d.ScanViewModel
import com.civilscan.nerf3d.data.ScanState
import com.civilscan.nerf3d.renderer.GaussianRenderer
import com.civilscan.nerf3d.ui.theme.Critical
import com.civilscan.nerf3d.ui.theme.Warning
import java.util.concurrent.Executors

@Composable
fun ScanScreen(vm: ScanViewModel, requestCamera: () -> Unit) {
    val context        = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scanState      by vm.scanState.collectAsState()
    val shards         by vm.shards.collectAsState()
    val gaussians      by vm.gaussians.collectAsState()
    val executor       = remember { Executors.newSingleThreadExecutor() }
    var showDepth      by remember { mutableStateOf(false) }

    val hasPermission = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
        PackageManager.PERMISSION_GRANTED
    if (!hasPermission) { PermissionPrompt(requestCamera); return }

    Column(Modifier.fillMaxSize().background(Color.Black)) {

        // ══ Top half: 3D point cloud OR depth map (toggle) ════════════════════
        Box(Modifier.weight(1f).fillMaxWidth()) {

            if (showDepth) {
                // ── Depth view ─────────────────────────────────────────────────
                DepthMapPanel(vm)

                // Model status pills (top-left in depth mode)
                Column(
                    Modifier.align(Alignment.TopStart).padding(10.dp),
                    verticalArrangement = Arrangement.spacedBy(5.dp)
                ) {
                    Pill("DEPTH")
                    Pill(
                        if (vm.midasLoaded) "MiDaS ✓" else "MiDaS ✗ missing",
                        tint = if (vm.midasLoaded) Color(0xFF3FB950) else Color(0xFFF85149)
                    )
                    Pill(
                        if (vm.renoLoaded) "RENO ✓" else "RENO ✗",
                        tint = if (vm.renoLoaded) Color(0xFF3FB950) else Color(0xFFD29922)
                    )
                }

                // Near/far legend (bottom-right in depth mode)
                Row(
                    Modifier.align(Alignment.BottomEnd).padding(horizontal = 10.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Pill("near", tint = Color(0xFFFF6B6B))
                    Pill("far",  tint = Color(0xFF6B8EFF))
                }

            } else {
                // ── 3D point cloud ─────────────────────────────────────────────
                PointCloudPanel(vm)

                // Stats pills (top-left in 3D mode)
                Column(
                    Modifier.align(Alignment.TopStart).padding(10.dp),
                    verticalArrangement = Arrangement.spacedBy(5.dp)
                ) {
                    val pts = (scanState as? ScanState.Running)?.pointCount
                        ?: (scanState as? ScanState.Paused)?.pointCount ?: 0
                    val nShards = (scanState as? ScanState.Running)?.shardCount
                        ?: (scanState as? ScanState.Paused)?.shardCount
                        ?: (scanState as? ScanState.Sharding)?.shardIndex ?: 0
                    Pill("${pts.fmtK} pts")
                    if (nShards > 0) Pill("$nShards shard${if (nShards > 1) "s" else ""}")
                }

                // Sharding overlay
                if (scanState is ScanState.Sharding) {
                    Surface(
                        Modifier.align(Alignment.Center),
                        color = Color.Black.copy(alpha = 0.78f),
                        shape = RoundedCornerShape(14.dp)
                    ) {
                        Row(
                            Modifier.padding(horizontal = 18.dp, vertical = 11.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            CircularProgressIndicator(
                                Modifier.size(18.dp),
                                color = MaterialTheme.colorScheme.primary,
                                strokeWidth = 2.5.dp
                            )
                            Text(
                                "Saving shard ${(scanState as ScanState.Sharding).shardIndex}  •  ${ScannerPipeline.SHARD_MAX_POINTS.fmtK} pts",
                                color = Color.White, fontSize = 13.sp
                            )
                        }
                    }
                }

                // Idle hint
                if (gaussians.isEmpty() && scanState is ScanState.Idle) {
                    Text(
                        "Tap Start to begin scanning",
                        Modifier.align(Alignment.Center),
                        color = Color.White.copy(alpha = 0.4f), fontSize = 13.sp
                    )
                }

                // Shard progress bar (bottom)
                val pts = (scanState as? ScanState.Running)?.pointCount
                    ?: (scanState as? ScanState.Paused)?.pointCount ?: 0
                val progress = pts.toFloat() / ScannerPipeline.SHARD_MAX_POINTS
                if (progress > 0f) {
                    LinearProgressIndicator(
                        progress = { progress },
                        modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth().height(2.dp),
                        color    = MaterialTheme.colorScheme.primary,
                        trackColor = Color.White.copy(alpha = 0.08f)
                    )
                }
            }

            // Perf badges + view toggle (top-right, always visible)
            Row(
                Modifier.align(Alignment.TopEnd).padding(10.dp),
                horizontalArrangement = Arrangement.spacedBy(5.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Show inference perf only in 3D mode
                if (!showDepth) {
                    val ms  = (scanState as? ScanState.Running)?.inferenceMs
                    val sim = (scanState as? ScanState.Running)?.isSimulated ?: vm.isSimulated
                    if (ms != null) {
                        Pill("${ms}ms")
                        Pill(if (sim) "SIM" else "NPU",
                            tint = if (sim) Warning else Color(0xFF3FB950))
                    }
                }
                ViewToggle(showDepth) { showDepth = it }
            }
        }

        HorizontalDivider(color = Color.White.copy(alpha = 0.10f), thickness = 1.dp)

        // ══ Bottom half: camera preview ═══════════════════════════════════════
        Box(Modifier.weight(1f).fillMaxWidth()) {
            AndroidView(
                factory = { ctx ->
                    val pv = PreviewView(ctx)
                    val future = ProcessCameraProvider.getInstance(ctx)
                    future.addListener({
                        val provider = future.get()
                        val preview  = Preview.Builder().build()
                            .also { it.setSurfaceProvider(pv.surfaceProvider) }
                        val analysis = ImageAnalysis.Builder()
                            .setTargetResolution(Size(640, 480))
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .build()
                            .also { ia -> ia.setAnalyzer(executor) { img -> vm.onFrame(img) } }
                        runCatching {
                            provider.unbindAll()
                            provider.bindToLifecycle(
                                lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
                            )
                        }
                    }, ContextCompat.getMainExecutor(ctx))
                    pv
                },
                modifier = Modifier.fillMaxSize()
            )

            // Recording status dot
            Row(
                Modifier.align(Alignment.TopStart).padding(10.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Box(Modifier.size(8.dp).background(
                    when {
                        scanState is ScanState.Running -> Critical
                        scanState is ScanState.Paused  -> Warning
                        else                           -> Color.Gray
                    }, CircleShape
                ))
                Text(
                    when (scanState) {
                        is ScanState.Running  -> "SCANNING"
                        is ScanState.Paused   -> "PAUSED"
                        is ScanState.Sharding -> "SAVING SHARD"
                        else                  -> "READY"
                    },
                    color = Color.White.copy(alpha = 0.85f),
                    fontSize = 10.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.sp
                )
            }
        }

        // ══ Control bar ═══════════════════════════════════════════════════════
        Row(
            Modifier
                .fillMaxWidth()
                .background(Color(0xFF0D1117))
                .padding(horizontal = 20.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp, Alignment.CenterHorizontally),
            verticalAlignment = Alignment.CenterVertically
        ) {
            val isRunning  = scanState is ScanState.Running
            val isSharding = scanState is ScanState.Sharding
            val isPaused   = scanState is ScanState.Paused
            val isIdle     = scanState is ScanState.Idle

            CtaButton(
                label = if (isRunning || isSharding) "Pause" else "Start",
                icon  = if (isRunning || isSharding) Icons.Filled.Pause else Icons.Filled.PlayArrow,
                enabled = !isSharding,
                primary = isIdle || isPaused,
                onClick = { if (isRunning) vm.pause() else vm.start() }
            )
            CtaButton(
                label = "New", icon = Icons.Filled.RestartAlt,
                enabled = true, primary = false,
                onClick = { vm.newRecording() }
            )
            val hasShards = shards.any { it.success }
            CtaButton(
                label   = if (shards.isEmpty()) "Export" else "Export (${shards.count { it.success }})",
                icon    = Icons.Filled.FileUpload,
                enabled = hasShards, primary = false,
                onClick = {
                    val paths = shards.filter { it.success }.map { it.path }
                    vm.export.shareAllIntent(paths)?.let { intent ->
                        context.startActivity(Intent.createChooser(intent, "Export shards"))
                    }
                }
            )
        }
    }
}

// ── View toggle (3D ↔ Depth) ──────────────────────────────────────────────────

@Composable
private fun ViewToggle(showDepth: Boolean, onToggle: (Boolean) -> Unit) {
    Surface(
        color = Color.Black.copy(alpha = 0.60f),
        shape = RoundedCornerShape(8.dp)
    ) {
        Row(
            Modifier.padding(3.dp),
            horizontalArrangement = Arrangement.spacedBy(3.dp)
        ) {
            listOf(false to "3D", true to "Depth").forEach { (isDepth, label) ->
                val active = showDepth == isDepth
                Surface(
                    onClick = { onToggle(isDepth) },
                    color = if (active) MaterialTheme.colorScheme.primary
                            else Color.Transparent,
                    shape = RoundedCornerShape(6.dp)
                ) {
                    Text(
                        label,
                        Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
                        color = if (active) Color.White else Color.White.copy(alpha = 0.55f),
                        fontSize = 12.sp, fontWeight = FontWeight.SemiBold
                    )
                }
            }
        }
    }
}

// ── Depth map panel ───────────────────────────────────────────────────────────
// Uses AndroidView(ImageView) so bitmap frames go directly to the view
// without triggering Compose recomposition — much lower latency.

@Composable
private fun DepthMapPanel(vm: ScanViewModel) {
    val scope = rememberCoroutineScope()
    var hasDepth by remember { mutableStateOf(false) }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(
            factory = { ctx ->
                ImageView(ctx).apply {
                    scaleType = ImageView.ScaleType.FIT_XY
                    val iv = this
                    scope.launch {
                        vm.depthBitmap.collect { bmp ->
                            if (bmp != null) {
                                iv.setImageBitmap(bmp)
                                if (!hasDepth) hasDepth = true
                            }
                        }
                    }
                }
            },
            modifier = Modifier.fillMaxSize()
        )
        if (!hasDepth) {
            Text(
                "Depth map will appear when scanning\n(requires MiDaS model)",
                Modifier.align(Alignment.Center),
                color = Color.White.copy(alpha = 0.4f),
                fontSize = 13.sp
            )
        }
    }
}

// ── GL point cloud panel ──────────────────────────────────────────────────────

@Composable
private fun PointCloudPanel(vm: ScanViewModel) {
    val gaussians by vm.gaussians.collectAsState()
    var renderer: GaussianRenderer? by remember { mutableStateOf(null) }
    LaunchedEffect(gaussians) { renderer?.setGaussians(gaussians) }

    AndroidView(
        factory = { ctx ->
            GLSurfaceView(ctx).apply {
                setEGLContextClientVersion(3)
                val r = GaussianRenderer(ctx)
                renderer = r
                setRenderer(r)
                renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
                setOnTouchListener { _, e -> r.onTouch(e); true }
            }
        },
        modifier = Modifier.fillMaxSize()
    )
}

// ── Reusable composables ──────────────────────────────────────────────────────

@Composable
private fun CtaButton(
    label: String, icon: ImageVector,
    enabled: Boolean, primary: Boolean,
    onClick: () -> Unit
) {
    if (primary) {
        Button(
            onClick = onClick, enabled = enabled,
            modifier = Modifier.height(44.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = MaterialTheme.colorScheme.primary,
                disabledContainerColor = Color.White.copy(alpha = 0.10f)
            ),
            contentPadding = PaddingValues(horizontal = 16.dp)
        ) {
            Icon(icon, null, Modifier.size(18.dp))
            Spacer(Modifier.width(6.dp))
            Text(label, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
        }
    } else {
        OutlinedButton(
            onClick = onClick, enabled = enabled,
            modifier = Modifier.height(44.dp),
            colors = ButtonDefaults.outlinedButtonColors(
                contentColor = Color.White,
                disabledContentColor = Color.White.copy(alpha = 0.30f)
            ),
            border = androidx.compose.foundation.BorderStroke(
                1.dp,
                if (enabled) Color.White.copy(alpha = 0.35f) else Color.White.copy(alpha = 0.12f)
            ),
            contentPadding = PaddingValues(horizontal = 16.dp)
        ) {
            Icon(icon, null, Modifier.size(18.dp))
            Spacer(Modifier.width(6.dp))
            Text(label, fontSize = 13.sp)
        }
    }
}

@Composable
private fun Pill(text: String, tint: Color = Color.White) {
    Surface(color = Color.Black.copy(alpha = 0.65f), shape = RoundedCornerShape(6.dp)) {
        Text(
            text, Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
            color = tint, fontSize = 11.sp, fontWeight = FontWeight.Medium
        )
    }
}

// ── Permission prompt ─────────────────────────────────────────────────────────

@Composable
private fun PermissionPrompt(request: () -> Unit) {
    Box(
        Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier.padding(40.dp)
        ) {
            Icon(Icons.Filled.CameraAlt, null, Modifier.size(64.dp),
                tint = MaterialTheme.colorScheme.primary)
            Text("Camera Access Required", fontWeight = FontWeight.Bold, fontSize = 20.sp,
                color = MaterialTheme.colorScheme.onSurface)
            Text("Camera access is needed to capture frames for 3D reconstruction.",
                color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 14.sp)
            Button(onClick = request) { Text("Grant Permission") }
        }
    }
}

// ── Utils ─────────────────────────────────────────────────────────────────────

private val Int.fmtK get() = if (this >= 1000) "${this / 1000}k" else "$this"

private object ScannerPipeline {
    val SHARD_MAX_POINTS get() = com.civilscan.nerf3d.pipeline.ScannerPipeline.SHARD_MAX_POINTS
}
