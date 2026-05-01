package com.civilscan.nerf3d.ui

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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.civilscan.nerf3d.ScanViewModel
import com.civilscan.nerf3d.data.ScanState
import com.civilscan.nerf3d.ui.theme.*

@Composable
fun HomeScreen(vm: ScanViewModel, nav: NavController) {
    val scanState by vm.scanState.collectAsState()

    Column(
        Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 24.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp)
    ) {
        // App title
        Column {
            Text("CivilScan 3D", fontSize = 30.sp, fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary)
            Text("On-Device Neural 3D Reconstruction", fontSize = 14.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }

        // NPU status card
        NpuStatusCard(vm)

        // Primary action
        Button(
            onClick = { nav.navigate("scan") },
            modifier = Modifier.fillMaxWidth().height(60.dp),
            shape = RoundedCornerShape(16.dp),
            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
        ) {
            Icon(Icons.Filled.CameraAlt, null, Modifier.size(24.dp))
            Spacer(Modifier.width(12.dp))
            Text("Start New Scan", fontSize = 18.sp, fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onPrimary)
        }

        // Current scan status
        if (scanState !is ScanState.Idle && scanState !is ScanState.PermissionRequired) {
            ScanStatusCard(scanState, nav)
        }

        // Feature cards
        Text("Pipeline", fontWeight = FontWeight.SemiBold, color = MaterialTheme.colorScheme.onSurfaceVariant)
        PipelineCard()

        Text("Powered by", fontWeight = FontWeight.SemiBold, color = MaterialTheme.colorScheme.onSurfaceVariant)
        TechStackCard()
    }
}

@Composable
private fun NpuStatusCard(vm: ScanViewModel) {
    Card(
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        shape  = RoundedCornerShape(16.dp)
    ) {
        Row(
            Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Icon(Icons.Filled.Memory, null,
                tint   = if (vm.npuActive) Success else Warning,
                modifier = Modifier.size(32.dp))
            Column(Modifier.weight(1f)) {
                Text("NPU Status", fontWeight = FontWeight.SemiBold, color = MaterialTheme.colorScheme.onSurface)
                Text(
                    if (vm.npuActive) "Hexagon HTP active · ${vm.delegateType}"
                    else              "Simulation mode (push .tflite files to enable NPU)",
                    fontSize = 12.sp,
                    color    = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Surface(
                color  = if (vm.npuActive) Success.copy(alpha=0.15f) else Warning.copy(alpha=0.15f),
                shape  = RoundedCornerShape(8.dp)
            ) {
                Text(
                    if (vm.npuActive) "LIVE" else "SIM",
                    Modifier.padding(horizontal=10.dp, vertical=4.dp),
                    color       = if (vm.npuActive) Success else Warning,
                    fontWeight  = FontWeight.Bold,
                    fontSize    = 12.sp
                )
            }
        }
    }
}

@Composable
private fun ScanStatusCard(state: ScanState, nav: NavController) {
    Card(
        onClick = {
            when (state) {
                is ScanState.Scanning    -> nav.navigate("scan")
                is ScanState.Done        -> nav.navigate("analysis")
                is ScanState.Compressing -> nav.navigate("analysis")
                else -> {}
            }
        },
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        shape  = RoundedCornerShape(16.dp)
    ) {
        Row(
            Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Icon(
                when (state) {
                    is ScanState.Done        -> Icons.Filled.CheckCircle
                    is ScanState.Compressing -> Icons.Filled.Compress
                    is ScanState.Scanning    -> Icons.Filled.RadioButtonChecked
                    else                     -> Icons.Filled.Info
                },
                null,
                tint = when (state) {
                    is ScanState.Done     -> Success
                    is ScanState.Error    -> Critical
                    else                  -> MaterialTheme.colorScheme.primary
                },
                modifier = Modifier.size(28.dp)
            )
            Column(Modifier.weight(1f)) {
                Text(
                    when (state) {
                        is ScanState.Scanning    -> "Scan in progress"
                        is ScanState.Compressing -> "Compressing scan…"
                        is ScanState.Done        -> "Scan complete — tap to view"
                        is ScanState.Error       -> "Error: ${state.message}"
                        else -> ""
                    },
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface
                )
                if (state is ScanState.Scanning) {
                    Text("${state.gaussianCount} Gaussians · ${state.frameCount} frames",
                        fontSize=12.sp, color=MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
            Icon(Icons.Filled.ChevronRight, null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun PipelineCard() {
    Card(
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        shape  = RoundedCornerShape(16.dp)
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            PipelineStep("CameraX", "Live YUV video frames", Icons.Filled.Videocam)
            PipelineArrow()
            PipelineStep("Mobile-GS", "3D Gaussian Splatting · NPU", Icons.Filled.ViewInAr)
            PipelineArrow()
            PipelineStep("RENO", "Neural voxel compression · NPU", Icons.Filled.Compress)
            PipelineArrow()
            PipelineStep("Civil Analysis", "Structural metrics on-device", Icons.Filled.Engineering)
            PipelineArrow()
            PipelineStep(".reno Export", "Share neural-compressed file", Icons.Filled.FileDownload)
        }
    }
}

@Composable
private fun PipelineStep(name: String, desc: String, icon: androidx.compose.ui.graphics.vector.ImageVector) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Icon(icon, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(20.dp))
        Column {
            Text(name, fontWeight = FontWeight.SemiBold, fontSize = 14.sp, color = MaterialTheme.colorScheme.onSurface)
            Text(desc, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun PipelineArrow() {
    Text("  ↓", color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 12.sp)
}

@Composable
private fun TechStackCard() {
    Card(
        Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        shape  = RoundedCornerShape(16.dp)
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf(
                "Google LiteRT 2.1.4" to "NPU inference runtime",
                "NNAPI → Hexagon HTP" to "Samsung Galaxy S25 Ultra NPU",
                "3D Gaussian Splatting" to "Real-time scene reconstruction",
                "RENO compression" to "47×+ compression ratio",
                "OpenGL ES 3.0" to "On-device 3D point-cloud viewer"
            ).forEach { (name, desc) ->
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text(name, fontSize = 13.sp, fontWeight = FontWeight.Medium,
                        color = MaterialTheme.colorScheme.onSurface)
                    Text(desc, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
    }
}
