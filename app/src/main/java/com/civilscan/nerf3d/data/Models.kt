package com.civilscan.nerf3d.data

// ── Gaussian primitive ────────────────────────────────────────────────────────

data class GaussianPoint(
    val x: Float, val y: Float, val z: Float,
    val opacity: Float,
    val scaleX: Float, val scaleY: Float, val scaleZ: Float,
    val r: Float, val g: Float, val b: Float
)

data class BoundingBox(
    val minX: Float, val maxX: Float,
    val minY: Float, val maxY: Float,
    val minZ: Float, val maxZ: Float
) {
    val width  get() = maxX - minX
    val height get() = maxY - minY
    val depth  get() = maxZ - minZ
    val volume get() = width * height * depth
}

// ── Per-frame pipeline output ─────────────────────────────────────────────────

data class FrameResult(
    val gaussians: List<GaussianPoint>,
    val inferenceMs: Long,
    val isSimulated: Boolean
)

// ── Final compressed scan ─────────────────────────────────────────────────────

data class CompressedScan(
    val latent: FloatArray,
    val gaussianCount: Int,
    val boundingBox: BoundingBox,
    val compressionRatio: Float,
    val frameCount: Int,
    val captureTimeMs: Long
) {
    val uncompressedSizeBytes: Long get() = gaussianCount.toLong() * 92 * 4  // 92 float attrs × 4B
    val compressedSizeBytes:   Long get() = latent.size.toLong() * 4
}

// ── Scan state machine ────────────────────────────────────────────────────────

sealed class ScanState {
    object Idle : ScanState()
    object PermissionRequired : ScanState()
    data class Scanning(
        val gaussianCount: Int  = 0,
        val frameCount: Int     = 0,
        val inferenceMs: Long   = 0L,
        val delegateType: String = "NPU",
        val isSimulated: Boolean = false
    ) : ScanState()
    data class Compressing(val progress: Float = 0f) : ScanState()
    data class Done(val scan: CompressedScan) : ScanState()
    data class Error(val message: String) : ScanState()
}

// ── Civil engineering analysis output ─────────────────────────────────────────

data class AnalysisResult(
    val dimensions:       Dimensions,
    val densityMetrics:   DensityMetrics,
    val surfaceMetrics:   SurfaceMetrics,
    val structuralFlags:  List<StructuralFlag>,
    val compressionStats: CompressionStats,
    val overallScore:     Float,          // 0 – 100
    val reportText:       String
)

data class Dimensions(
    val lengthM:     Float,
    val widthM:      Float,
    val heightM:     Float,
    val footprintM2: Float,
    val volumeM3:    Float
)

data class DensityMetrics(
    val totalPoints:     Int,
    val densityPtsPerM2: Float,
    val coveragePct:     Float,
    val avgSpacingMm:    Float
)

data class SurfaceMetrics(
    val planarityScore:   Float,   // 0 – 100
    val roughnessMmRms:   Float,
    val undulationIndex:  Float,
    val normalUniformity: Float
)

data class StructuralFlag(
    val type:        FlagType,
    val severity:    Severity,
    val description: String,
    val location:    String
)

enum class FlagType {
    CRACK_SIGNATURE, LOW_DENSITY_VOID, SURFACE_DEVIATION,
    SETTLEMENT_INDICATOR, MOISTURE_STAIN
}

enum class Severity { INFO, WARNING, CRITICAL }

data class CompressionStats(
    val gaussianCount:   Int,
    val latentDim:       Int,
    val uncompressedMb:  Float,
    val compressedMb:    Float,
    val ratio:           Float
)

// ── Export ────────────────────────────────────────────────────────────────────

data class ExportedFile(
    val path:    String,
    val sizeKb:  Long,
    val success: Boolean,
    val error:   String? = null
)
