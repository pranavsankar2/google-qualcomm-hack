package com.civilscan.nerf3d

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.core.content.ContextCompat
import com.civilscan.nerf3d.ui.ScanScreen
import com.civilscan.nerf3d.ui.theme.CivilScanTheme

class MainActivity : ComponentActivity() {

    val viewModel: ScanViewModel by viewModels()

    private val cameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* ScanScreen re-checks on recomposition */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            CivilScanTheme {
                ScanScreen(viewModel, ::requestCamera)
            }
        }
    }

    private fun requestCamera() = cameraPermission.launch(Manifest.permission.CAMERA)
}
