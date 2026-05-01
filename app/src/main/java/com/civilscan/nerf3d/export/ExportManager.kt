package com.civilscan.nerf3d.export

import android.content.Context
import android.content.Intent
import android.graphics.*
import android.os.Environment
import androidx.core.content.FileProvider
import com.civilscan.nerf3d.data.*
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.*

/**
 * Handles all export operations:
 *   • .reno  — binary neural-compressed scan file (header + latent floats)
 *   • report.png — full-resolution 1080×1920 analysis report rendered on Canvas
 *   • Android share-sheet integration via FileProvider
 *
 * .reno binary format (little-endian):
 *   4B  magic        0x52454E4F  "RENO"
 *   2B  version      2
 *   4B  gaussianCount
 *   4B  latentDim
 *   4B  frameCount
 *   8B  captureTimestamp  (epoch ms)
 *   24B boundingBox   6 × float32  (minX maxX minY maxY minZ maxZ)
 *   4B  compressionRatio  float32
 *   latentDim × 4B  latent vector  float32[]
 */
class ExportManager(private val context: Context) {

    companion object {
        private const val RENO_MAGIC   = 0x52454E4F.toInt()
        private const val RENO_VERSION = 2.toShort()
    }

    // ── .reno export ──────────────────────────────────────────────────────────

    fun exportReno(scan: CompressedScan, prefix: String = "scan"): ExportedFile {
        val file = timestampedFile("${prefix}", "reno")
        return try {
            DataOutputStream(BufferedOutputStream(FileOutputStream(file))).use { dos ->
                // Header
                dos.writeInt(RENO_MAGIC)
                dos.writeShort(RENO_VERSION.toInt())
                dos.writeInt(scan.gaussianCount)
                dos.writeInt(scan.latent.size)
                dos.writeInt(scan.frameCount)
                dos.writeLong(scan.captureTimeMs)
                // Bounding box
                with(scan.boundingBox) {
                    dos.writeFloat(minX); dos.writeFloat(maxX)
                    dos.writeFloat(minY); dos.writeFloat(maxY)
                    dos.writeFloat(minZ); dos.writeFloat(maxZ)
                }
                dos.writeFloat(scan.compressionRatio)
                // Latent vector — packed LE float array
                val buf = ByteBuffer.allocate(scan.latent.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                scan.latent.forEach { buf.putFloat(it) }
                dos.write(buf.array())
            }
            ExportedFile(file.absolutePath, file.length() / 1024, true)
        } catch (e: Exception) {
            ExportedFile("", 0, false, e.message)
        }
    }

    // ── PNG report ────────────────────────────────────────────────────────────

    fun exportReport(scan: CompressedScan, analysis: AnalysisResult): ExportedFile {
        val file = timestampedFile("report", "png")
        val w = 1080; val h = 1920
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        renderReport(canvas, w.toFloat(), h.toFloat(), analysis)
        return try {
            FileOutputStream(file).use { bmp.compress(Bitmap.CompressFormat.PNG, 95, it) }
            ExportedFile(file.absolutePath, file.length() / 1024, true)
        } catch (e: Exception) {
            ExportedFile("", 0, false, e.message)
        }
    }

    // ── Share intent ──────────────────────────────────────────────────────────

    fun shareIntent(path: String, mime: String = "*/*"): Intent {
        val uri = FileProvider.getUriForFile(
            context, "${context.packageName}.fileprovider", File(path))
        return Intent(Intent.ACTION_SEND).apply {
            type = mime
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
    }

    fun listExports(): List<File> =
        exportsDir().listFiles()
            ?.filter { it.extension in listOf("reno", "png") }
            ?.sortedByDescending { it.lastModified() }
            ?: emptyList()

    // ── Canvas report renderer ────────────────────────────────────────────────

    private fun renderReport(canvas: Canvas, w: Float, h: Float, a: AnalysisResult) {
        val bg      = Color.parseColor("#0D1117")
        val surface = Color.parseColor("#161B22")
        val accent  = Color.parseColor("#00D4AA")
        val white   = Color.parseColor("#E6EDF3")
        val gray    = Color.parseColor("#8B949E")
        val green   = Color.parseColor("#3FB950")
        val amber   = Color.parseColor("#D29922")
        val red     = Color.parseColor("#F85149")
        val blue    = Color.parseColor("#58A6FF")

        canvas.drawRect(0f, 0f, w, h, paint(bg, fill = true))

        var y = 0f

        // Header bar
        canvas.drawRect(0f, 0f, w, 160f, paint(surface, fill = true))
        y = 60f
        canvas.drawText("CivilScan 3D", 60f, y, typedPaint(white, 56f, bold = true))
        y = 115f
        canvas.drawText("Structural Analysis Report", 60f, y, typedPaint(accent, 36f))
        y = 200f

        // Score card
        val scoreColor = when {
            a.overallScore >= 80 -> green
            a.overallScore >= 60 -> amber
            else                 -> red
        }
        canvas.drawRoundRect(RectF(40f, y, w-40f, y+150f), 20f, 20f, paint(surface, fill=true))
        canvas.drawText("Overall Score", 80f, y+55f,  typedPaint(gray, 28f))
        canvas.drawText("${a.overallScore.toInt()} / 100", 80f, y+125f, typedPaint(scoreColor, 72f, bold=true))
        y += 190f

        fun section(title: String) {
            canvas.drawRect(0f, y, w, y+1.5f, paint(Color.parseColor("#21262D"), fill=true))
            canvas.drawText(title, 60f, (y + 55f).also { /* set y below */ }, typedPaint(accent, 32f, bold=true))
        }

        // Dimensions section
        section("DIMENSIONS"); y += 70f
        val d = a.dimensions
        listOf(
            "Length"    to "${f1(d.lengthM)} m",
            "Width"     to "${f1(d.widthM)} m",
            "Height"    to "${f1(d.heightM)} m",
            "Footprint" to "${f1(d.footprintM2)} m²",
            "Volume"    to "${f1(d.volumeM3)} m³"
        ).forEach { (label, value) ->
            canvas.drawText(label, 80f,  y, typedPaint(gray,  30f))
            canvas.drawText(value, 600f, y, typedPaint(white, 30f, bold=true))
            y += 48f
        }
        y += 20f

        // Point cloud section
        section("POINT CLOUD QUALITY"); y += 70f
        val den = a.densityMetrics
        listOf(
            "Total Points"  to "${den.totalPoints}",
            "Density"       to "${den.densityPtsPerM2.toInt()} pts/m²",
            "Coverage"      to "${den.coveragePct.toInt()}%",
            "Avg Spacing"   to "${f1(den.avgSpacingMm)} mm"
        ).forEach { (label, value) ->
            canvas.drawText(label, 80f,  y, typedPaint(gray,  30f))
            canvas.drawText(value, 600f, y, typedPaint(white, 30f, bold=true))
            y += 48f
        }
        y += 20f

        // Surface section
        section("SURFACE METRICS"); y += 70f
        val sur = a.surfaceMetrics
        listOf(
            "Planarity Score"   to "${sur.planarityScore.toInt()} / 100",
            "Roughness (RMS)"   to "${f1(sur.roughnessMmRms)} mm",
            "Normal Uniformity" to "${(sur.normalUniformity * 100).toInt()}%"
        ).forEach { (label, value) ->
            canvas.drawText(label, 80f,  y, typedPaint(gray,  30f))
            canvas.drawText(value, 600f, y, typedPaint(white, 30f, bold=true))
            y += 48f
        }
        y += 20f

        // Compression section
        section("NEURAL COMPRESSION"); y += 70f
        val cs = a.compressionStats
        listOf(
            "Gaussian Count" to "${cs.gaussianCount}",
            "Uncompressed"   to "${f2(cs.uncompressedMb)} MB",
            "Compressed"     to "${f2(cs.compressedMb)} MB",
            "Ratio"          to "${f1(cs.ratio)}×",
            "Latent Dim"     to "${cs.latentDim}"
        ).forEach { (label, value) ->
            canvas.drawText(label, 80f,  y, typedPaint(gray,  30f))
            canvas.drawText(value, 600f, y, typedPaint(white, 30f, bold=true))
            y += 48f
        }
        y += 20f

        // Structural flags
        section("STRUCTURAL FLAGS"); y += 70f
        if (a.structuralFlags.isEmpty()) {
            canvas.drawText("✓  No significant flags detected.", 80f, y, typedPaint(green, 30f))
            y += 50f
        } else {
            for (fl in a.structuralFlags) {
                val fc = when(fl.severity) { Severity.CRITICAL->red; Severity.WARNING->amber; else->blue }
                canvas.drawText("[${fl.severity}] ${fl.type}", 80f, y, typedPaint(fc, 28f, bold=true))
                y += 38f
                fl.description.chunked(55).forEach { line ->
                    canvas.drawText(line, 100f, y, typedPaint(gray, 24f))
                    y += 34f
                }
                y += 10f
            }
        }

        // Footer
        val ts = SimpleDateFormat("yyyy-MM-dd  HH:mm", Locale.US).format(Date())
        canvas.drawText("Generated by CivilScan 3D  •  $ts", 60f, h - 60f, typedPaint(gray, 24f))
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun paint(color: Int, fill: Boolean = false) = Paint().apply {
        this.color  = color
        style       = if (fill) Paint.Style.FILL else Paint.Style.STROKE
        isAntiAlias = true
    }

    private fun typedPaint(color: Int, size: Float, bold: Boolean = false) = Paint().apply {
        this.color  = color
        textSize    = size
        typeface    = if (bold) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
        isAntiAlias = true
    }

    private fun exportsDir() = File(
        context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), "exports"
    ).also { it.mkdirs() }

    private fun timestampedFile(prefix: String, ext: String): File {
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return File(exportsDir(), "${prefix}_${ts}.$ext")
    }

    private fun f1(v: Float) = "%.1f".format(v)
    private fun f2(v: Float) = "%.2f".format(v)
}
