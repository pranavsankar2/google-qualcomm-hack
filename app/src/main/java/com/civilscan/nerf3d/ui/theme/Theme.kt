package com.civilscan.nerf3d.ui.theme

import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// Civil-engineering dark palette
val Teal400    = Color(0xFF00D4AA)
val Teal200    = Color(0xFF66EDD5)
val Navy900    = Color(0xFF0D1117)
val Navy800    = Color(0xFF161B22)
val Navy700    = Color(0xFF21262D)
val Navy600    = Color(0xFF30363D)
val TextPrimary   = Color(0xFFE6EDF3)
val TextSecondary = Color(0xFF8B949E)
val Success    = Color(0xFF3FB950)
val Warning    = Color(0xFFD29922)
val Critical   = Color(0xFFF85149)
val InfoBlue   = Color(0xFF58A6FF)

private val DarkColors = darkColorScheme(
    primary          = Teal400,
    onPrimary        = Navy900,
    primaryContainer = Navy700,
    secondary        = InfoBlue,
    onSecondary      = Navy900,
    background       = Navy900,
    onBackground     = TextPrimary,
    surface          = Navy800,
    onSurface        = TextPrimary,
    surfaceVariant   = Navy700,
    onSurfaceVariant = TextSecondary,
    error            = Critical,
    onError          = TextPrimary,
    outline          = Navy600,
)

@Composable
fun CivilScanTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = DarkColors,
        typography  = Typography(),
        content     = content
    )
}
