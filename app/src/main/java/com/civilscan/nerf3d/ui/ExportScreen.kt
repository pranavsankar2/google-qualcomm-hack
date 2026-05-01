package com.civilscan.nerf3d.ui

import android.content.Intent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.civilscan.nerf3d.ScanViewModel
import com.civilscan.nerf3d.data.*
import com.civilscan.nerf3d.ui.theme.*

@Composable
fun ExportScreen(vm: ScanViewModel) {
    val context      = LocalContext.current
    val scanState    by vm.scanState.collectAsState()
    val analysis     by vm.analysisResult.collectAsState()
    val exported     by vm.exportedFile.collectAsState()

    val scan = (scanState as? ScanState.Done)?.scan

    Column(
        Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 24.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp)
    ) {
        Text("Export", fontSize = 26.sp, fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurface)
        Text("Share the neural-compressed scan or generate a PDF analysis report.",
            fontSize = 14.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)

        if (scan == null) {
            NoScanCard()
            return@Column
        }

        // Scan summary
        ScanSummaryCard(scan, analysis)

        // Export actions
        ExportButton(
            title    = "Export .reno File",
            subtitle = "Neural-compressed scan (${(scan.compressedSizeBytes / 1024f).toInt()} KB)",
            icon     = Icons.Filled.DataObject,
            color    = MaterialTheme.colorScheme.primary,
            onClick  = { vm.exportReno() }
        )

        if (analysis != null) {
            ExportButton(
                title    = "Export Analysis Report",
                subtitle = "Full PNG report with structural metrics",
                icon     = Icons.Filled.Assessment,
                color    = InfoBlue,
                onClick  = { vm.exportReport() }
            )
        }

        // Result feedback
        exported?.let { ef ->
            ExportResultCard(ef, onShare = {
                val mime = if (ef.path.endsWith(".reno")) "*/*" else "image/png"
                context.startActivity(
                    Intent.createChooser(vm.export.shareIntent(ef.path, mime), "Share via")
                )
            })
        }

        // Previous exports
        val prevExports = remember { vm.export.listExports() }
        if (prevExports.isNotEmpty()) {
            Text("Previous Exports", fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            prevExports.take(5).forEach { file ->
                PreviousExportRow(
                    name    = file.name,
                    sizeKb  = file.length() / 1024,
                    onShare = {
                        val mime = if (file.extension == "reno") "*/*" else "image/png"
                        context.startActivity(
                            Intent.createChooser(vm.export.shareIntent(file.absolutePath, mime), "Share via")
                        )
                    }
                )
            }
        }
    }
}

@Composable
private fun NoScanCard() {
    Card(
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        shape  = RoundedCornerShape(16.dp)
    ) {
        Column(
            Modifier.padding(32.dp).fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Icon(Icons.Filled.FileDownload, null, Modifier.size(48.dp),
                tint = MaterialTheme.colorScheme.onSurfaceVariant)
            Text("No scan to export", fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface)
            Text("Complete a scan first, then return here to export the .reno file and report.",
                color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 13.sp)
        }
    }
}

@Composable
private fun ScanSummaryCard(scan: CompressedScan, analysis: AnalysisResult?) {
    Card(
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        shape  = RoundedCornerShape(16.dp)
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Icon(Icons.Filled.CheckCircle, null, tint = Success, modifier = Modifier.size(20.dp))
                Text("Scan Ready", fontWeight = FontWeight.SemiBold, color = MaterialTheme.colorScheme.onSurface)
            }
            HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha=0.4f))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Gaussians",   color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 13.sp)
                Text("${scan.gaussianCount}", fontWeight = FontWeight.Medium, color = MaterialTheme.colorScheme.onSurface, fontSize = 13.sp)
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Frames",      color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 13.sp)
                Text("${scan.frameCount}", fontWeight = FontWeight.Medium, color = MaterialTheme.colorScheme.onSurface, fontSize = 13.sp)
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Ratio",       color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 13.sp)
                Text("%.1f×".format(scan.compressionRatio), fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.primary, fontSize = 13.sp)
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("File size",   color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 13.sp)
                Text("${scan.compressedSizeBytes / 1024} KB", fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.onSurface, fontSize = 13.sp)
            }
            analysis?.let { a ->
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("Score",   color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 13.sp)
                    Text("${a.overallScore.toInt()} / 100", fontWeight = FontWeight.Bold,
                        color = if (a.overallScore >= 80) Success else Warning, fontSize = 13.sp)
                }
            }
        }
    }
}

@Composable
private fun ExportButton(title: String, subtitle: String, icon: ImageVector, color: Color, onClick: () -> Unit) {
    Button(
        onClick  = onClick,
        modifier = Modifier.fillMaxWidth().height(72.dp),
        shape    = RoundedCornerShape(16.dp),
        colors   = ButtonDefaults.buttonColors(containerColor = color)
    ) {
        Icon(icon, null, Modifier.size(24.dp))
        Spacer(Modifier.width(14.dp))
        Column(horizontalAlignment = Alignment.Start) {
            Text(title,    fontWeight = FontWeight.SemiBold, fontSize = 15.sp,
                color = MaterialTheme.colorScheme.onPrimary)
            Text(subtitle, fontSize = 11.sp,
                color = MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.75f))
        }
    }
}

@Composable
private fun ExportResultCard(exported: ExportedFile, onShare: () -> Unit) {
    Card(
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (exported.success) Success.copy(alpha=0.1f) else Critical.copy(alpha=0.1f)
        ),
        shape = RoundedCornerShape(16.dp)
    ) {
        Row(
            Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Icon(
                if (exported.success) Icons.Filled.CheckCircle else Icons.Filled.Error,
                null, tint = if (exported.success) Success else Critical, modifier = Modifier.size(28.dp)
            )
            Column(Modifier.weight(1f)) {
                Text(
                    if (exported.success) "Saved successfully" else "Export failed",
                    fontWeight = FontWeight.SemiBold, color = MaterialTheme.colorScheme.onSurface
                )
                if (exported.success) {
                    Text("${exported.sizeKb} KB · ${exported.path.substringAfterLast("/")}",
                        fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                } else {
                    Text(exported.error ?: "Unknown error",
                        fontSize = 11.sp, color = Critical)
                }
            }
            if (exported.success) {
                IconButton(onClick = onShare) {
                    Icon(Icons.Filled.Share, null, tint = MaterialTheme.colorScheme.primary)
                }
            }
        }
    }
}

@Composable
private fun PreviousExportRow(name: String, sizeKb: Long, onShare: () -> Unit) {
    Card(
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        shape  = RoundedCornerShape(12.dp)
    ) {
        Row(
            Modifier.padding(horizontal=16.dp, vertical=12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Icon(
                if (name.endsWith(".reno")) Icons.Filled.DataObject else Icons.Filled.Image,
                null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(20.dp)
            )
            Column(Modifier.weight(1f)) {
                Text(name, fontSize = 13.sp, fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.onSurface, maxLines = 1)
                Text("$sizeKb KB", fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            IconButton(onClick = onShare, modifier = Modifier.size(36.dp)) {
                Icon(Icons.Filled.Share, null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}
