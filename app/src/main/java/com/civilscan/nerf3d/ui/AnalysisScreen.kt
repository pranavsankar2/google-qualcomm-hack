package com.civilscan.nerf3d.ui

import android.opengl.GLSurfaceView
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.navigation.NavController
import com.civilscan.nerf3d.ScanViewModel
import com.civilscan.nerf3d.data.*
import com.civilscan.nerf3d.renderer.GaussianRenderer
import com.civilscan.nerf3d.ui.theme.*

@Composable
fun AnalysisScreen(vm: ScanViewModel, nav: NavController) {
    val analysis   by vm.analysisResult.collectAsState()
    val gaussians  by vm.gaussians.collectAsState()
    val scanState  by vm.scanState.collectAsState()

    if (scanState is ScanState.Compressing || (scanState is ScanState.Idle && analysis == null)) {
        LoadingPane()
        return
    }

    Column(
        Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
    ) {
        // ── 3D Viewer ─────────────────────────────────────────────────────────
        Box(Modifier.fillMaxWidth().height(280.dp)) {
            GlViewer(vm)
            // Colour mode selector overlay
            Row(
                Modifier.align(Alignment.TopEnd).padding(10.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                GaussianRenderer.ColorMode.values().forEach { mode ->
                    SmallChip(mode.name, MaterialTheme.colorScheme.primary) {}
                }
            }
            Text("Drag to orbit  •  Pinch to zoom",
                Modifier.align(Alignment.BottomCenter).padding(8.dp),
                color = Color.White.copy(alpha=0.5f), fontSize = 11.sp)
        }

        // ── Metrics ───────────────────────────────────────────────────────────
        Column(
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            val a = analysis
            if (a == null) {
                if (scanState is ScanState.Scanning || scanState is ScanState.Compressing) {
                    LoadingPane()
                } else {
                    EmptyPane { nav.navigate("scan") }
                }
                return@Column
            }

            // Score header
            ScoreHeader(a.overallScore)

            // Flags summary
            if (a.structuralFlags.isNotEmpty()) {
                FlagsCard(a.structuralFlags)
            }

            // Dimension card
            MetricCard("Dimensions", Icons.Filled.Straighten) {
                GridRow("Length",    "${f1(a.dimensions.lengthM)} m")
                GridRow("Width",     "${f1(a.dimensions.widthM)} m")
                GridRow("Height",    "${f1(a.dimensions.heightM)} m")
                GridRow("Footprint", "${f1(a.dimensions.footprintM2)} m²")
                GridRow("Volume",    "${f1(a.dimensions.volumeM3)} m³")
            }

            // Point cloud card
            MetricCard("Point Cloud Quality", Icons.Filled.ScatterPlot) {
                GridRow("Total Points",  "${a.densityMetrics.totalPoints}")
                GridRow("Density",       "${a.densityMetrics.densityPtsPerM2.toInt()} pts/m²")
                GridRow("Coverage",      "${a.densityMetrics.coveragePct.toInt()}%")
                GridRow("Avg Spacing",   "${f1(a.densityMetrics.avgSpacingMm)} mm")
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(
                    progress = { a.densityMetrics.coveragePct / 100f },
                    modifier = Modifier.fillMaxWidth().height(6.dp),
                    color    = MaterialTheme.colorScheme.primary,
                    trackColor = MaterialTheme.colorScheme.outline
                )
            }

            // Surface card
            MetricCard("Surface Quality", Icons.Filled.Terrain) {
                GridRow("Planarity Score",   "${a.surfaceMetrics.planarityScore.toInt()} / 100")
                GridRow("Roughness (RMS)",   "${f1(a.surfaceMetrics.roughnessMmRms)} mm")
                GridRow("Normal Uniformity", "${(a.surfaceMetrics.normalUniformity * 100).toInt()}%")
                Spacer(Modifier.height(8.dp))
                // Planarity bar
                LinearProgressIndicator(
                    progress = { a.surfaceMetrics.planarityScore / 100f },
                    modifier = Modifier.fillMaxWidth().height(6.dp),
                    color    = scoreColor(a.surfaceMetrics.planarityScore),
                    trackColor = MaterialTheme.colorScheme.outline
                )
            }

            // Compression card
            MetricCard("Neural Compression (RENO)", Icons.Filled.Compress) {
                GridRow("Gaussian Count",   "${a.compressionStats.gaussianCount}")
                GridRow("Uncompressed",     "${f2(a.compressionStats.uncompressedMb)} MB")
                GridRow("Compressed",       "${f2(a.compressionStats.compressedMb)} MB")
                GridRow("Ratio",            "${f1(a.compressionStats.ratio)}×")
                GridRow("Latent Dim",       "${a.compressionStats.latentDim}")
            }

            // Export CTA
            Button(
                onClick = { nav.navigate("export") },
                modifier = Modifier.fillMaxWidth(),
                colors   = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Icon(Icons.Filled.FileDownload, null)
                Spacer(Modifier.width(8.dp))
                Text("Export .reno + Report")
            }
        }
    }
}

// ── 3D GL Viewer wrapper ──────────────────────────────────────────────────────

@Composable
private fun GlViewer(vm: ScanViewModel) {
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
                setOnTouchListener { _, event -> r.onTouch(event); true }
            }
        },
        modifier = Modifier.fillMaxSize()
    )
}

// ── Composable helpers ────────────────────────────────────────────────────────

@Composable
private fun ScoreHeader(score: Float) {
    Card(
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        shape  = RoundedCornerShape(16.dp)
    ) {
        Row(
            Modifier.padding(20.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column {
                Text("Overall Score", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text("${score.toInt()} / 100", fontSize = 42.sp, fontWeight = FontWeight.Bold,
                    color = scoreColor(score))
            }
            Icon(
                if (score >= 80) Icons.Filled.CheckCircle
                else if (score >= 60) Icons.Filled.Warning
                else Icons.Filled.Error,
                null,
                Modifier.size(48.dp),
                tint = scoreColor(score)
            )
        }
    }
}

@Composable
private fun FlagsCard(flags: List<StructuralFlag>) {
    Card(
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Critical.copy(alpha = 0.08f)),
        border = BorderStroke(1.dp, Critical.copy(alpha = 0.3f)),
        shape  = RoundedCornerShape(16.dp)
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Icon(Icons.Filled.Flag, null, tint = Critical, modifier = Modifier.size(18.dp))
                Text("Structural Flags (${flags.size})", fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface)
            }
            flags.forEach { flag ->
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.Top) {
                    Surface(
                        color = flagColor(flag.severity).copy(alpha = 0.15f),
                        shape = RoundedCornerShape(4.dp)
                    ) {
                        Text(flag.severity.name, Modifier.padding(horizontal=6.dp, vertical=2.dp),
                            fontSize = 10.sp, fontWeight = FontWeight.Bold, color = flagColor(flag.severity))
                    }
                    Text(flag.description, fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.weight(1f))
                }
            }
        }
    }
}

@Composable
private fun MetricCard(title: String, icon: ImageVector, content: @Composable ColumnScope.() -> Unit) {
    Card(
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        shape  = RoundedCornerShape(16.dp)
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Icon(icon, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(18.dp))
                Text(title, fontWeight = FontWeight.SemiBold, color = MaterialTheme.colorScheme.onSurface)
            }
            HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha=0.4f))
            content()
        }
    }
}

@Composable
private fun GridRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, fontSize = 13.sp, fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSurface)
    }
}

@Composable
private fun SmallChip(label: String, color: Color, onClick: () -> Unit) {
    Surface(
        onClick = onClick,
        color   = color.copy(alpha = 0.2f),
        shape   = RoundedCornerShape(6.dp)
    ) {
        Text(label, Modifier.padding(horizontal=8.dp, vertical=3.dp),
            color = color, fontSize = 10.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun LoadingPane() {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)) {
            CircularProgressIndicator(color = MaterialTheme.colorScheme.primary)
            Text("Compressing & analysing…", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun EmptyPane(onScan: () -> Unit) {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.padding(40.dp)) {
            Icon(Icons.Filled.ViewInAr, null, Modifier.size(72.dp),
                tint = MaterialTheme.colorScheme.primary.copy(alpha=0.5f))
            Text("No scan data yet", fontWeight = FontWeight.SemiBold, fontSize = 18.sp,
                color = MaterialTheme.colorScheme.onSurface)
            Text("Complete a scan first to see the 3D reconstruction and civil engineering analysis.",
                color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 14.sp)
            Button(onClick = onScan) {
                Icon(Icons.Filled.CameraAlt, null)
                Spacer(Modifier.width(8.dp))
                Text("Start Scan")
            }
        }
    }
}

private fun scoreColor(score: Float) = when {
    score >= 80 -> Success
    score >= 60 -> Warning
    else        -> Critical
}

private fun flagColor(severity: Severity) = when (severity) {
    Severity.CRITICAL -> Critical
    Severity.WARNING  -> Warning
    Severity.INFO     -> InfoBlue
}

private fun f1(v: Float) = "%.1f".format(v)
private fun f2(v: Float) = "%.2f".format(v)
