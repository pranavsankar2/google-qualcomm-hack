package com.civilscan.nerf3d

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.core.content.ContextCompat
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.civilscan.nerf3d.ui.*
import com.civilscan.nerf3d.ui.theme.CivilScanTheme

class MainActivity : ComponentActivity() {

    val viewModel: ScanViewModel by viewModels()

    private val cameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* navigation handled in ScanScreen */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            CivilScanTheme {
                CivilScanApp(viewModel, ::requestCamera)
            }
        }
    }

    fun hasCameraPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED

    private fun requestCamera() = cameraPermission.launch(Manifest.permission.CAMERA)
}

// ── Navigation ────────────────────────────────────────────────────────────────

private sealed class Screen(val route: String, val label: String, val icon: ImageVector) {
    object Home     : Screen("home",     "Home",     Icons.Filled.Home)
    object Scan     : Screen("scan",     "Scan",     Icons.Filled.CameraAlt)
    object Analysis : Screen("analysis", "Analysis", Icons.Filled.Analytics)
    object Export   : Screen("export",   "Export",   Icons.Filled.FileDownload)
}

private val screens = listOf(Screen.Home, Screen.Scan, Screen.Analysis, Screen.Export)

@Composable
fun CivilScanApp(vm: ScanViewModel, requestCamera: () -> Unit) {
    val nav = rememberNavController()

    Scaffold(
        bottomBar = { BottomNavBar(nav) }
    ) { inner ->
        NavHost(nav, startDestination = Screen.Home.route, Modifier.padding(inner)) {
            composable(Screen.Home.route)     { HomeScreen(vm, nav) }
            composable(Screen.Scan.route)     { ScanScreen(vm, nav, requestCamera) }
            composable(Screen.Analysis.route) { AnalysisScreen(vm, nav) }
            composable(Screen.Export.route)   { ExportScreen(vm) }
        }
    }
}

@Composable
private fun BottomNavBar(nav: NavHostController) {
    val backStack by nav.currentBackStackEntryAsState()
    val current   = backStack?.destination?.route

    NavigationBar {
        screens.forEach { screen ->
            NavigationBarItem(
                icon     = { Icon(screen.icon, contentDescription = screen.label) },
                label    = { Text(screen.label) },
                selected = current == screen.route,
                onClick  = {
                    nav.navigate(screen.route) {
                        popUpTo(Screen.Home.route) { saveState = true }
                        launchSingleTop = true
                        restoreState    = true
                    }
                }
            )
        }
    }
}
