package com.civilscan.nerf3d.ui

import android.Manifest
import android.content.pm.PackageManager
import android.util.Size
import android.view.MotionEvent
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.navigation.NavController
import com.civilscan.nerf3d.ScanViewModel
import com.civilscan.nerf3d.data.ScanState
import com.civilscan.nerf3d.ui.theme.*
import java.util.concurrent.Executors

@Composable
fun ScanScreen(vm: ScanViewModel, nav: NavController, requestCamera: () -> Unit) {
    val context       = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scanState     by vm.scanState.collectAsState()

    val hasPermission = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
        PackageManager.PERMISSION_GRANTED

    if (!hasPermission) {
        PermissionPrompt(requestCamera)
        return
    }

    val executor = remember { Executors.newSingleThreadExecutor() }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        // ── Camera preview ────────────────────────────────────────────────────
        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx)
                val cameraFuture = ProcessCameraProvider.getInstance(ctx)
                cameraFuture.addListener({
                    val provider = cameraFuture.get()
                    val preview  = Preview.Builder().build()
                        .also { it.setSurfaceProvider(previewView.surfaceProvider) }

                    val analysis = ImageAnalysis.Builder()
                        .setTargetResolution(Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                        .also { ia ->
                            ia.setAnalyzer(executor) { image -> vm.onFrame(image) }
                        }

                    runCatching {
                        provider.unbindAll()
                        provider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_BACK_CAMERA,
                            preview, analysis
                        )
                    }
                }, ContextCompat.getMainExecutor(ctx))
                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        // ── Top HUD ───────────────────────────────────────────────────────────
        Column(
            Modifier
                .fillMaxWidth()
                .background(Color.Black.copy(alpha = 0.55f))
                .padding(horizontal = 20.dp, vertical = 14.dp)
        ) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                HudChip(
                    label = if (vm.npuActive) "NPU" else "SIM",
                    color = if (vm.npuActive) Success else Warning
                )
                HudChip(label = vm.delegateType, color = InfoBlue)
                if (scanState is ScanState.Scanning) {
                    val s = scanState as ScanState.Scanning
                    HudChip(label = "${s.inferenceMs}ms", color = MaterialTheme.colorScheme.primary)
                }
            }

            AnimatedVisibility(scanState is ScanState.Scanning) {
                val s = scanState as? ScanState.Scanning ?: return@AnimatedVisibility
                Column(Modifier.padding(top = 10.dp)) {
                    Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) {
                        HudStat("Gaussians", "${s.gaussianCount}")
                        HudStat("Frames",    "${s.frameCount}")
                        HudStat("Mode",      if (s.isSimulated) "Simulated" else "Live NPU")
                    }
                }
            }
        }

        // ── Bottom controls ───────────────────────────────────────────────────
        Column(
            Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(Color.Black.copy(alpha = 0.65f))
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            when (scanState) {
                is ScanState.Idle, is ScanState.PermissionRequired -> {
                    Button(
                        onClick  = { vm.startScan() },
                        modifier = Modifier.size(80.dp),
                        shape    = CircleShape,
                        colors   = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
                    ) {
                        Icon(Icons.Filled.FiberManualRecord, null, Modifier.size(36.dp))
                    }
                    Text("Tap to start scan", color = Color.White.copy(alpha = 0.8f), fontSize = 14.sp)
                }

                is ScanState.Scanning -> {
                    val s = scanState as ScanState.Scanning
                    // Pulsing record button
                    Button(
                        onClick  = {
                            vm.stopAndAnalyze()
                            nav.navigate("analysis")
                        },
                        modifier = Modifier.size(80.dp),
                        shape    = CircleShape,
                        colors   = ButtonDefaults.buttonColors(containerColor = Critical)
                    ) {
                        Icon(Icons.Filled.Stop, null, Modifier.size(36.dp))
                    }
                    Text("${s.gaussianCount} Gaussians accumulated",
                        color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Medium)
                    Text("Tap stop to compress & analyse",
                        color = Color.White.copy(alpha=0.6f), fontSize = 12.sp)
                }

                is ScanState.Compressing -> {
                    val p = (scanState as ScanState.Compressing).progress
                    CircularProgressIndicator(
                        progress = { p },
                        modifier = Modifier.size(60.dp),
                        color    = MaterialTheme.colorScheme.primary,
                        strokeWidth = 6.dp
                    )
                    Text("Compressing with RENO…",
                        color = Color.White, fontSize = 14.sp)
                }

                is ScanState.Done -> {
                    Button(
                        onClick = { nav.navigate("analysis") },
                        colors  = ButtonDefaults.buttonColors(containerColor = Success)
                    ) {
                        Icon(Icons.Filled.Analytics, null)
                        Spacer(Modifier.width(8.dp))
                        Text("View Analysis")
                    }
                    TextButton(onClick = { vm.resetScan() }) {
                        Text("New Scan", color = Color.White.copy(alpha=0.7f))
                    }
                }

                else -> {}
            }
        }
    }
}

@Composable
private fun PermissionPrompt(request: () -> Unit) {
    Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background),
        contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier.padding(40.dp)) {
            Icon(Icons.Filled.CameraAlt, null, Modifier.size(64.dp),
                tint = MaterialTheme.colorScheme.primary)
            Text("Camera Access Required", fontWeight = FontWeight.Bold, fontSize = 20.sp,
                color = MaterialTheme.colorScheme.onSurface)
            Text("CivilScan needs the camera to capture video frames for 3D reconstruction.",
                color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 14.sp)
            Button(onClick = request) { Text("Grant Permission") }
        }
    }
}

@Composable
private fun HudChip(label: String, color: Color) {
    Surface(color = color.copy(alpha = 0.2f), shape = RoundedCornerShape(8.dp)) {
        Text(label, Modifier.padding(horizontal = 10.dp, vertical = 4.dp),
            color = color, fontSize = 12.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun HudStat(label: String, value: String) {
    Column {
        Text(value, color = Color.White, fontWeight = FontWeight.Bold, fontSize = 16.sp)
        Text(label,  color = Color.White.copy(alpha=0.6f), fontSize = 11.sp)
    }
}
