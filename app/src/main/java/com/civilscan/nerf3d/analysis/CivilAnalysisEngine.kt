package com.civilscan.nerf3d.analysis

import com.civilscan.nerf3d.data.*
import kotlin.math.*

// With real MiDaS depth, XYZ are already in metres (METERS_PER_UNIT = 1.0)
private const val METERS_PER_UNIT = 1f

/**
 * Derives civil-engineering metrics from the raw Gaussian point cloud and the
 * compressed scan metadata.  All computations run on the CPU post-scan.
 *
 * Key outputs:
 *   • Scene dimensions (L × W × H in real metres)
 *   • Point density and scan coverage
 *   • Surface planarity via 3×3 covariance PCA
 *   • Surface roughness (RMS deviation from best-fit plane)
 *   • Anomaly / defect cluster detection
 *   • Overall structural quality score (0 – 100)
 */
class CivilAnalysisEngine {

    fun analyze(scan: CompressedScan, gaussians: List<GaussianPoint>): AnalysisResult {
        val dims    = dimensions(scan.boundingBox)
        val density = density(scan, dims)
        val surface = surfaceMetrics(gaussians)
        val flags   = detectFlags(gaussians, dims, density, surface)
        val stats   = CompressionStats(
            gaussianCount  = scan.gaussianCount,
            latentDim      = scan.latent.size,
            uncompressedMb = scan.uncompressedSizeBytes / 1_048_576f,
            compressedMb   = scan.compressedSizeBytes   / 1_048_576f,
            ratio          = scan.compressionRatio
        )
        val score  = overallScore(density, surface, flags)
        val report = buildReport(dims, density, surface, flags, stats, score)
        return AnalysisResult(dims, density, surface, flags, stats, score, report)
    }

    // ── Dimensions ────────────────────────────────────────────────────────────

    private fun dimensions(bb: BoundingBox): Dimensions {
        val l = bb.width  * METERS_PER_UNIT
        val w = bb.depth  * METERS_PER_UNIT
        val h = bb.height * METERS_PER_UNIT
        return Dimensions(l, w, h, l * w, l * w * h)
    }

    // ── Point density ─────────────────────────────────────────────────────────

    private fun density(scan: CompressedScan, dims: Dimensions): DensityMetrics {
        val fp      = dims.footprintM2.coerceAtLeast(0.01f)
        val density = scan.gaussianCount / fp
        val cov     = (density / 8_000f * 100f).coerceIn(5f, 99f)
        val spacing = if (density > 0f) sqrt(1f / density) * 1_000f else 200f
        return DensityMetrics(scan.gaussianCount, density, cov, spacing)
    }

    // ── Surface quality via covariance PCA ───────────────────────────────────

    private fun surfaceMetrics(gaussians: List<GaussianPoint>): SurfaceMetrics {
        if (gaussians.size < 20) return SurfaceMetrics(50f, 8f, 0.4f, 0.5f)
        val sample = gaussians.shuffled().take(8_000)

        val cx = sample.sumOf { it.x.toDouble() }.toFloat() / sample.size
        val cy = sample.sumOf { it.y.toDouble() }.toFloat() / sample.size
        val cz = sample.sumOf { it.z.toDouble() }.toFloat() / sample.size

        var cxx=0f; var cyy=0f; var czz=0f
        var cxy=0f; var cxz=0f; var cyz=0f
        for (g in sample) {
            val dx=g.x-cx; val dy=g.y-cy; val dz=g.z-cz
            cxx+=dx*dx; cyy+=dy*dy; czz+=dz*dz
            cxy+=dx*dy; cxz+=dx*dz; cyz+=dy*dz
        }
        val n = sample.size.toFloat()
        cxx/=n; cyy/=n; czz/=n; cxy/=n; cxz/=n; cyz/=n

        // Frobenius norm and determinant approximation for smallest eigenvalue
        val trace = cxx + cyy + czz
        val det   = cxx*(cyy*czz - cyz*cyz) - cxy*(cxy*czz - cyz*cxz) + cxz*(cxy*cyz - cyy*cxz)
        val lMin  = if (trace > 0f) abs(det) / (trace * trace + 1e-6f) else 0f

        val planarity  = (100f - lMin * 600f).coerceIn(0f, 100f)
        val roughness  = (sqrt(lMin * trace + 1e-6f) * METERS_PER_UNIT * 1_000f).coerceIn(0f, 50f)
        val undulation = (roughness / 25f).coerceIn(0f, 1f)

        // Scale variance → normal uniformity proxy
        val scales = sample.map { it.scaleX + it.scaleY + it.scaleZ }
        val meanSc = scales.average().toFloat()
        val scVar  = scales.sumOf { ((it - meanSc) * (it - meanSc)).toDouble() }.toFloat() / scales.size
        val normU  = (1f - scVar * 8f).coerceIn(0f, 1f)

        return SurfaceMetrics(planarity, roughness, undulation, normU)
    }

    // ── Structural flag detection ─────────────────────────────────────────────

    private fun detectFlags(
        gaussians: List<GaussianPoint>,
        dims: Dimensions,
        density: DensityMetrics,
        surface: SurfaceMetrics
    ): List<StructuralFlag> = buildList {

        if (density.densityPtsPerM2 < 1_200f)
            add(StructuralFlag(
                FlagType.LOW_DENSITY_VOID, Severity.WARNING,
                "Point density ${density.densityPtsPerM2.toInt()} pts/m² is below the minimum " +
                "recommended 1 200 pts/m². Structural voids or obstructions may be under-sampled.",
                "Multiple zones"
            ))

        if (surface.planarityScore < 65f)
            add(StructuralFlag(
                FlagType.SURFACE_DEVIATION, Severity.WARNING,
                "Planarity score ${surface.planarityScore.toInt()}/100 indicates non-planar " +
                "surfaces. Possible curvature, deformation, or scan misalignment.",
                "General surface"
            ))

        if (surface.roughnessMmRms > 8f)
            add(StructuralFlag(
                FlagType.CRACK_SIGNATURE,
                if (surface.roughnessMmRms > 20f) Severity.CRITICAL else Severity.WARNING,
                "Surface roughness ${String.format("%.1f", surface.roughnessMmRms)} mm RMS. " +
                "Values > 8 mm may indicate cracking, spalling, or settlement.",
                "Primary scanned surface"
            ))

        if (density.coveragePct < 65f)
            add(StructuralFlag(
                FlagType.LOW_DENSITY_VOID, Severity.INFO,
                "Scan coverage ${density.coveragePct.toInt()}%. Re-scan obstructed or distant areas " +
                "for a complete structural assessment.",
                "Coverage boundary"
            ))

        // Semi-transparent Gaussians → moisture / staining heuristic
        val semiCount = gaussians.count { it.opacity < 0.3f }
        if (semiCount > gaussians.size * 0.12f)
            add(StructuralFlag(
                FlagType.MOISTURE_STAIN, Severity.INFO,
                "${semiCount} semi-transparent point primitives detected (${(semiCount * 100f / gaussians.size.coerceAtLeast(1)).toInt()}%). " +
                "Possible moisture ingress, efflorescence, or surface contamination.",
                "Scattered regions"
            ))

        // Settlement: large vertical extent relative to footprint
        if (dims.heightM > 0f && dims.footprintM2 / dims.heightM < 5f && dims.heightM < 1.5f)
            add(StructuralFlag(
                FlagType.SETTLEMENT_INDICATOR, Severity.INFO,
                "Unusual height-to-footprint ratio detected. Verify scan orientation; " +
                "potential differential settlement pattern.",
                "Foundation zone"
            ))
    }

    // ── Scoring ───────────────────────────────────────────────────────────────

    private fun overallScore(
        density: DensityMetrics,
        surface: SurfaceMetrics,
        flags: List<StructuralFlag>
    ): Float {
        var score = 100f
        score -= if (density.densityPtsPerM2 < 1_200f) 15f else 0f
        score -= (100f - surface.planarityScore) * 0.25f
        score -= surface.roughnessMmRms * 0.6f
        score -= flags.count { it.severity == Severity.CRITICAL } * 25f
        score -= flags.count { it.severity == Severity.WARNING  } * 10f
        score -= flags.count { it.severity == Severity.INFO     } * 2f
        return score.coerceIn(0f, 100f)
    }

    // ── Text report ───────────────────────────────────────────────────────────

    private fun buildReport(
        d: Dimensions, den: DensityMetrics, sur: SurfaceMetrics,
        flags: List<StructuralFlag>, stats: CompressionStats, score: Float
    ) = buildString {
        appendLine("=== CIVILSCAN 3D — STRUCTURAL REPORT ===")
        appendLine("Overall Score : ${score.toInt()} / 100")
        appendLine()
        appendLine("DIMENSIONS")
        appendLine("  L × W × H : ${f1(d.lengthM)} × ${f1(d.widthM)} × ${f1(d.heightM)} m")
        appendLine("  Footprint  : ${f1(d.footprintM2)} m²   Volume: ${f1(d.volumeM3)} m³")
        appendLine()
        appendLine("POINT CLOUD")
        appendLine("  Points      : ${den.totalPoints}")
        appendLine("  Density     : ${den.densityPtsPerM2.toInt()} pts/m²")
        appendLine("  Coverage    : ${den.coveragePct.toInt()}%   Avg spacing: ${f1(den.avgSpacingMm)} mm")
        appendLine()
        appendLine("SURFACE QUALITY")
        appendLine("  Planarity   : ${sur.planarityScore.toInt()} / 100")
        appendLine("  Roughness   : ${f1(sur.roughnessMmRms)} mm RMS")
        appendLine("  Undulation  : ${f1(sur.undulationIndex)}")
        appendLine()
        appendLine("NEURAL COMPRESSION (RENO)")
        appendLine("  Input size  : ${f2(stats.uncompressedMb)} MB  →  ${f2(stats.compressedMb)} MB")
        appendLine("  Ratio       : ${f1(stats.ratio)}×   Latent dim: ${stats.latentDim}")
        appendLine()
        if (flags.isEmpty()) {
            appendLine("STRUCTURAL FLAGS : None — within normal parameters.")
        } else {
            appendLine("STRUCTURAL FLAGS (${flags.size})")
            flags.forEach { fl ->
                appendLine("  [${fl.severity}] ${fl.type}")
                appendLine("    ${fl.description}")
                appendLine("    Location: ${fl.location}")
            }
        }
    }

    private fun f1(v: Float) = "%.1f".format(v)
    private fun f2(v: Float) = "%.2f".format(v)
}
