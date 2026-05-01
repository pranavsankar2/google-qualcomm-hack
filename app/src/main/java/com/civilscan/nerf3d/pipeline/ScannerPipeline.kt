package com.civilscan.nerf3d.pipeline

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.civilscan.nerf3d.data.*
import kotlin.math.*
import kotlin.random.Random

/**
 * SLAM-aware 3D scanning pipeline.
 *
 * ── Depth ──────────────────────────────────────────────────────────────────────
 * MiDaS v2.1 small (mobile_gs_quant.tflite)
 *   in : [1,256,256,3] float32 ImageNet-normalised
 *   out: [1,256,256,1] float32 inverse relative depth (disparity)
 *
 * ── SLAM pose ─────────────────────────────────────────────────────────────────
 * Rotation  — Android TYPE_GAME_ROTATION_VECTOR, expressed relative to the
 *             first frame so world-Z = initial camera forward.
 * Translation — Sparse optical flow (gradient-selected features, 2-pass SAD
 *               coarse→fine) + 1-point RANSAC for outlier-robust estimation.
 *
 * ── Compression ───────────────────────────────────────────────────────────────
 * EfficientNet-Lite0 (reno_quant.tflite) — 1-shot encoder at shard finalise.
 */
class ScannerPipeline(context: Context) {

    companion object {
        private const val TAG = "ScannerPipeline"

        // MiDaS
        const val DEPTH_W = 256; const val DEPTH_H = 256
        private val MEAN  = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD   = floatArrayOf(0.229f, 0.224f, 0.225f)

        // MobileNetV3 Small image embedder (MediaPipe) — 1024-dim scene embedding
        const val ENC_W = 224; const val ENC_H = 224; const val LATENT_DIM = 1024

        // Depth inference throttle: run MiDaS every N frames, reuse last depth in between
        private const val DEPTH_SKIP = 3

        // Pinhole intrinsics for 256×256 (~65° HFOV rear camera)
        private const val FX = 190f; private const val FY = 190f
        private const val CX = 128f; private const val CY = 128f
        private const val MAX_D = 8f; private const val MIN_D = 0.3f

        private const val DEPTH_MIN_VALID = 0.35f
        private const val DEPTH_MAX_VALID = 7.75f
        private const val DEPTH_SMOOTH_ALPHA = 0.2f

        // Fixed disparity → depth mapping. Scene-independent; the constants are
        // properties of the MiDaS model's output range, not the scene.
        // Tuned for this specific quantized model: raw output spans ~[0, 1100].
        //   raw ≈ 100  → 10 m (far)
        //   raw ≈ 500  → 2 m  (mid)
        //   raw ≈ 1100 → 0.9 m (near)
        // Re-tune by reading the "MiDaS raw" log line.
        private const val RAW_DISP_MIN = 100f       // smallest disparity we trust → far
        private const val RAW_DISP_MAX = 2500f      // largest disparity we trust → near
        private const val DEPTH_K      = 1000f      // depth = DEPTH_K / disparity

        // Edge confidence: skip pixels where depth changes sharply between
        // neighbors (relative to local depth). MiDaS hallucinates around object
        // edges, so emitting Gaussians there produces "streamer" artifacts.
        private const val EDGE_REL_THRESH = 0.18f

        // Voxel-grid dedup on accumulation. ~5 cm voxels: industry-standard
        // densification bound for splat-style point clouds. Caps memory growth
        // and prevents per-frame duplicate Gaussians at the same world location.
        private const val VOXEL_SIZE = 0.05f
        private const val VOXEL_RANGE = 1024     // ±51.2 m at 5 cm voxels

        // Point cloud sampling
        private const val STRIDE = 4               // ~4 096 pts/frame

        // Optical flow
        private const val FEAT_STEP   = 20         // feature grid spacing
        private const val PATCH       = 4          // half-patch (8×8 comparison)
        private const val COARSE_HALF = 16         // coarse search ±16 px, stride 4
        private const val FINE_HALF   = 4          // fine refine ±4 px, stride 1
        private const val GRAD_THRESH = 0.004f     // min gradient² to accept feature
        private const val SAD_THRESH  = 0.10f      // max mean SAD per pixel
        private const val RANSAC_N    = 25         // RANSAC iterations
        private const val INLIER_R    = 0.12f      // 12 cm inlier radius
        private const val MIN_INLIERS = 6          // min inliers to trust estimate
        private const val MAX_DT      = 0.1f       // max translation per frame (m)

        // Pose-drift containment. The IMU integrator + optical-flow estimator
        // both accumulate error every frame; without these caps the splat slowly
        // walks away from where the user pointed the camera.
        private const val MAX_WORLD_DRIFT = 2.0f    // hard clamp on worldT magnitude (m)
        private const val VEL_DAMP_RATE   = 3.0f    // exp(-RATE·dt) — was 0.2; ~0.23s half-life
        private const val ACCEL_DEADZONE  = 0.5f    // ignore |accel| < this (m/s²); sensor noise floor

        const val SHARD_MAX_POINTS = 250_000
        const val METERS_PER_UNIT  = 1f

        // Depth colormap LUT: index 0 = near (red), 255 = far (blue)
        private val DEPTH_LUT = IntArray(256) { i ->
            android.graphics.Color.HSVToColor(floatArrayOf(i / 255f * 240f, 1f, 1f))
        }
    }

    private val manager = ModelManager(context)
    private val models  = manager.loadModels()

    val delegateType: String  get() = models.delegateType
    val npuActive:    Boolean get() = models.npuActive
    val isSimulated:  Boolean get() = models.gs == null
    val renoLoaded:   Boolean get() = models.reno != null

    // ── Inference buffers (LiteRT TensorBuffer-based) ─────────────────────────
    // Staging FloatArrays hold pixel-prepared input; we copy into LiteRT
    // TensorBuffers right before run(). Output is read back as a flat
    // FloatArray and reshaped in buildDepth().
    private val depthInArr = FloatArray(DEPTH_H * DEPTH_W * 3)
    private val depthInBufs  by lazy { models.gs?.createInputBuffers()  ?: emptyList() }
    private val depthOutBufs by lazy { models.gs?.createOutputBuffers() ?: emptyList() }
    private val encInArr = FloatArray(ENC_H * ENC_W * 3)
    private val encInBufs    by lazy { models.reno?.createInputBuffers()  ?: emptyList() }
    private val encOutBufs   by lazy { models.reno?.createOutputBuffers() ?: emptyList() }

    // ── Accumulation ──────────────────────────────────────────────────────────
    private val acc = mutableListOf<GaussianPoint>()
    private var frameCount = 0
    private var lastBmp: Bitmap? = null

    // Voxel occupancy set for dedup. Cleared on reset; pruned with `acc`.
    private val occupiedVoxels = HashSet<Long>()

    // ── SLAM state ────────────────────────────────────────────────────────────
    // World = coordinate frame of first camera frame (Z forward, Y up, X right)
    private var initR:    FloatArray? = null     // anchor: sensor rotation at frame 0
    private var worldTx = 0f; private var worldTy = 0f; private var worldTz = 0f

    // Previous-frame data for optical flow
    private var prevGray:  FloatArray?          = null
    private var prevDepth: Array<FloatArray>?   = null
    private var prevRrel:  FloatArray?          = null  // previous relative R (world_cam)

    // Depth visualization
    @Volatile private var latestDepthArray:  Array<FloatArray>? = null
    @Volatile private var cachedDepthBitmap: Bitmap?            = null

    // Accelerometer integration
    private var velX = 0f; private var velY = 0f; private var velZ = 0f
    private var lastFrameMs = 0L

    // Depth throttle counter (only run MiDaS every DEPTH_SKIP frames)
    private var depthFrameCounter = 0

    // ── Public API ────────────────────────────────────────────────────────────

    fun reset() {
        acc.clear(); occupiedVoxels.clear(); frameCount = 0; lastBmp = null
        initR = null
        worldTx = 0f; worldTy = 0f; worldTz = 0f
        prevGray = null; prevDepth = null; prevRrel = null
        velX = 0f; velY = 0f; velZ = 0f; lastFrameMs = 0L
        latestDepthArray = null; cachedDepthBitmap = null; depthFrameCounter = 0
    }

    fun resetAccumulationOnly() {
        acc.clear(); occupiedVoxels.clear(); frameCount = 0; lastBmp = null
        // SLAM pose and disparity calibration intentionally kept.
    }

    @Synchronized
    fun processFrame(bitmap: Bitmap, sensorR: FloatArray? = null, accWorld: FloatArray? = null): FrameResult {
        frameCount++; lastBmp = bitmap
        val t0 = System.currentTimeMillis()
        val raw = if (!isSimulated) realFrame(bitmap, sensorR, accWorld) else simFrame(sensorR)
        val added = ArrayList<GaussianPoint>(raw.size)
        for (g in raw) {
            val key = voxelKey(g.x, g.y, g.z)
            if (occupiedVoxels.add(key)) { acc += g; added += g }
        }
        if (acc.size > 500_000) {
            val drop = acc.size - 350_000
            for (i in 0 until drop) {
                val g = acc[i]; occupiedVoxels.remove(voxelKey(g.x, g.y, g.z))
            }
            repeat(drop) { acc.removeAt(0) }
        }
        return FrameResult(added, System.currentTimeMillis() - t0, isSimulated)
    }

    private fun voxelKey(x: Float, y: Float, z: Float): Long {
        val vx = (floor(x / VOXEL_SIZE).toInt() + VOXEL_RANGE).toLong() and 0x7FF
        val vy = (floor(y / VOXEL_SIZE).toInt() + VOXEL_RANGE).toLong() and 0x7FF
        val vz = (floor(z / VOXEL_SIZE).toInt() + VOXEL_RANGE).toLong() and 0x7FF
        return (vx shl 22) or (vy shl 11) or vz
    }

    fun compressAndFinalize(): CompressedScan {
        val gs     = acc.toList()
        val latent = if (models.reno != null && lastBmp != null) runEncoder(lastBmp!!) else simLatent()
        val bbox   = bbox(gs)
        val raw    = gs.size.toLong() * 10 * 4
        val ratio  = if (latent.isNotEmpty()) raw.toFloat() / (latent.size * 4f) else 47.6f
        Log.i(TAG, "Shard: ${gs.size} pts ratio=${ratio.f1}× T=(${worldTx.f2},${worldTy.f2},${worldTz.f2})")
        return CompressedScan(latent, gs.size, bbox, ratio, frameCount, System.currentTimeMillis())
    }

    fun getGaussians(): List<GaussianPoint> = acc.toList()
    fun close() { manager.close() }

    fun getDepthBitmap(): Bitmap? = cachedDepthBitmap

    private fun buildDepthBitmap(d: Array<FloatArray>): Bitmap {
        val bmp    = Bitmap.createBitmap(DEPTH_W, DEPTH_H, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(DEPTH_W * DEPTH_H)
        for (v in 0 until DEPTH_H) for (u in 0 until DEPTH_W) {
            val n = ((d[v][u] - MIN_D) / (MAX_D - MIN_D)).coerceIn(0f, 1f)
            pixels[v * DEPTH_W + u] = DEPTH_LUT[(n * 255).toInt()]
        }
        bmp.setPixels(pixels, 0, DEPTH_W, 0, 0, DEPTH_W, DEPTH_H)
        return bmp
    }

    // ── Real depth + SLAM ─────────────────────────────────────────────────────

    private fun realFrame(bmp: Bitmap, sensorR: FloatArray?, accWorld: FloatArray?): List<GaussianPoint> {
        val now = System.currentTimeMillis()
        val dt  = if (lastFrameMs > 0L) ((now - lastFrameMs) / 1000f).coerceAtMost(0.1f) else 0f
        lastFrameMs = now

        val scaled = Bitmap.createScaledBitmap(bmp, DEPTH_W, DEPTH_H, true)
        val gray   = toGray(scaled)

        // Only run MiDaS every DEPTH_SKIP frames; reuse last depth map in between
        depthFrameCounter++
        var depth: Array<FloatArray>
        if (depthFrameCounter % DEPTH_SKIP == 0 || latestDepthArray == null) {
            prepDepthIn(scaled)
            models.gs!!.run(depthInBufs, depthOutBufs)
            val rawOut = depthOutBufs[0].readFloat()
            depth = buildDepth(rawOut)
            val prev = prevDepth
            if (prev != null) depth = smoothTemporalDepth(depth, prev)
            latestDepthArray  = depth
            cachedDepthBitmap = buildDepthBitmap(depth)
        } else {
            depth = latestDepthArray!!
        }
        val R     = sensorR ?: prevRrel?.let { relR(it) } ?: identity3()

        if (initR == null && sensorR != null) initR = sensorR.copyOf()
        val Rrel = worldRot(R)

        // Translation: prefer accelerometer (metric), fall back to optical flow
        if (accWorld != null && dt > 0f) {
            val ax = if (abs(accWorld[0]) > ACCEL_DEADZONE) accWorld[0] else 0f
            val ay = if (abs(accWorld[1]) > ACCEL_DEADZONE) accWorld[1] else 0f
            val az = if (abs(accWorld[2]) > ACCEL_DEADZONE) accWorld[2] else 0f
            velX += ax * dt; velY += ay * dt; velZ += az * dt
            // Aggressive velocity decay — IMU bias would otherwise integrate
            // into unbounded drift even when the phone is stationary.
            val damp = exp(-VEL_DAMP_RATE * dt)
            velX *= damp; velY *= damp; velZ *= damp
            worldTx += velX * dt; worldTy += velY * dt; worldTz += velZ * dt
        } else {
            val pG = prevGray; val pD = prevDepth; val pRrel = prevRrel
            if (pG != null && pD != null && pRrel != null) {
                val (dx, dy, dz) = estimateDelta(pG, pD, pRrel, gray, depth, Rrel)
                worldTx += dx; worldTy += dy; worldTz += dz
            }
        }
        // Hard clamp on accumulated world translation. Whatever the IMU /
        // optical-flow estimator says, the splat can't walk further than
        // MAX_WORLD_DRIFT from the origin where the first frame anchored.
        worldTx = worldTx.coerceIn(-MAX_WORLD_DRIFT, MAX_WORLD_DRIFT)
        worldTy = worldTy.coerceIn(-MAX_WORLD_DRIFT, MAX_WORLD_DRIFT)
        worldTz = worldTz.coerceIn(-MAX_WORLD_DRIFT, MAX_WORLD_DRIFT)

        prevGray = gray; prevDepth = depth; prevRrel = Rrel
        return unproject(depth, scaled, Rrel)
    }

    /**
     * Compute R_world_cam = R_relative_to_init * cam_flip.
     *
     * R_init anchors the world frame so Z = initial camera forward.
     * cam_flip = diag(1,−1,−1) converts camera axes (X right, Y down, Z fwd)
     * to Android body frame (X right, Y up, Z toward user).
     *
     * R_world_cam[row][col]:
     *   col 0 unchanged (cam X = body X)
     *   col 1 negated   (cam Y down = −body Y up)
     *   col 2 negated   (cam Z fwd = −body Z toward-user)
     */
    private fun worldRot(sensorR: FloatArray): FloatArray {
        // Relative sensor rotation: R_rel = R_init^T * R_current
        val base = initR ?: sensorR
        val Rrel = mat3Mul(mat3T(base), sensorR)
        // Apply cam→body flip (negate columns 1 and 2)
        return FloatArray(9) { i -> if (i % 3 == 0) Rrel[i] else -Rrel[i] }
    }

    // Fallback when we have Rrel but no fresh sensorR
    private fun relR(Rrel: FloatArray) = FloatArray(9) { i ->
        // undo the flip to recover a fake sensorR — simpler: just return Rrel as-is
        Rrel[i]
    }

    /**
     * Unproject depth map to world-space Gaussians: P_world = Rrel * P_cam + T.
     *
     * Per-pixel filters:
     *  • depth ∈ [DEPTH_MIN_VALID, MAX_D − 0.1)
     *  • relative depth gradient < EDGE_REL_THRESH (skip object boundaries
     *    where MiDaS hallucinates streamer points)
     *
     * Splat sizing: each Gaussian covers the world-space footprint of its
     * source pixel block at depth d → scale = d · STRIDE / FX. This is the
     * standard depth-map-to-3DGS initializer (used in 3DGS / Mip-Splatting
     * preprocessing).
     */
    private fun unproject(
        depth: Array<FloatArray>, colorBmp: Bitmap, Rrel: FloatArray
    ): List<GaussianPoint> {
        val R  = Rrel
        val px = IntArray(DEPTH_W * DEPTH_H)
        colorBmp.getPixels(px, 0, DEPTH_W, 0, 0, DEPTH_W, DEPTH_H)
        val out = ArrayList<GaussianPoint>((DEPTH_W / STRIDE) * (DEPTH_H / STRIDE))
        val pixelScale = STRIDE.toFloat() / FX     // angular size of one stride-step
        val uMin = STRIDE; val uMax = DEPTH_W - STRIDE
        val vMin = STRIDE; val vMax = DEPTH_H - STRIDE
        for (v in 0 until DEPTH_H step STRIDE) {
            for (u in 0 until DEPTH_W step STRIDE) {
                val d = depth[v][u]
                if (d > MAX_D - 0.1f || d < DEPTH_MIN_VALID) continue

                // Relative depth gradient — skip depth discontinuities.
                if (u in uMin until uMax && v in vMin until vMax) {
                    val gx = depth[v][u + STRIDE] - depth[v][u - STRIDE]
                    val gy = depth[v + STRIDE][u] - depth[v - STRIDE][u]
                    val rel = (abs(gx) + abs(gy)) / d
                    if (rel > EDGE_REL_THRESH) continue
                }

                val cx = (u - CX) * d / FX
                val cy = (v - CY) * d / FY
                val s  = d * pixelScale
                val argb = px[v * DEPTH_W + u]
                out += GaussianPoint(
                    x = R[0]*cx + R[1]*cy + R[2]*d + worldTx,
                    y = R[3]*cx + R[4]*cy + R[5]*d + worldTy,
                    z = R[6]*cx + R[7]*cy + R[8]*d + worldTz,
                    opacity = 0.9f,
                    scaleX = s, scaleY = s, scaleZ = s,
                    r = (argb shr 16 and 0xFF) / 255f,
                    g = (argb shr  8 and 0xFF) / 255f,
                    b = (argb        and 0xFF) / 255f
                )
            }
        }
        return out
    }

    // ── Optical flow translation estimator ────────────────────────────────────

    /**
     * Estimate world-space translation delta between two frames.
     *
     * For each matched feature (u0,v0)→(u1,v1):
     *   P0_world = pRrel * P0_cam   (no T — we solve for delta only)
     *   P1_world = cRrel * P1_cam
     *   dT_i = P0_world − P1_world
     *
     * 1-point RANSAC over all dT_i → robust consensus translation.
     */
    private fun estimateDelta(
        pGray: FloatArray, pDepth: Array<FloatArray>, pRrel: FloatArray,
        cGray: FloatArray, cDepth: Array<FloatArray>, cRrel: FloatArray
    ): Triple<Float, Float, Float> {
        val pGrad = gradMag(pGray)
        val dx = ArrayList<Float>(); val dy = ArrayList<Float>(); val dz = ArrayList<Float>()

        val u0min = PATCH + COARSE_HALF; val u0max = DEPTH_W - PATCH - COARSE_HALF
        val v0min = PATCH + COARSE_HALF; val v0max = DEPTH_H - PATCH - COARSE_HALF

        var u0 = u0min
        while (u0 < u0max) {
            var v0 = v0min
            while (v0 < v0max) {
                val d0 = pDepth[v0][u0]
                if (d0 >= MAX_D - 0.1f || pGrad[v0 * DEPTH_W + u0] < GRAD_THRESH) {
                    v0 += FEAT_STEP; continue
                }

                // ── Pass 1: coarse SAD (stride 4) ─────────────────────────────
                var bestSad = Float.MAX_VALUE; var buC = 0; var bvC = 0
                for (dv in -COARSE_HALF..COARSE_HALF step 4) {
                    for (du in -COARSE_HALF..COARSE_HALF step 4) {
                        val u1 = u0 + du; val v1 = v0 + dv
                        if (u1 < PATCH || u1 >= DEPTH_W - PATCH ||
                            v1 < PATCH || v1 >= DEPTH_H - PATCH) continue
                        var sad = 0f
                        for (pv in -PATCH until PATCH) for (pu in -PATCH until PATCH)
                            sad += abs(pGray[(v0+pv)*DEPTH_W+(u0+pu)] - cGray[(v1+pv)*DEPTH_W+(u1+pu)])
                        if (sad < bestSad) { bestSad = sad; buC = du; bvC = dv }
                    }
                }

                // ── Pass 2: fine SAD (stride 1) around coarse winner ──────────
                var bu = buC; var bv = bvC
                bestSad = Float.MAX_VALUE
                for (dv in (bvC - FINE_HALF)..(bvC + FINE_HALF)) {
                    for (du in (buC - FINE_HALF)..(buC + FINE_HALF)) {
                        val u1 = u0 + du; val v1 = v0 + dv
                        if (u1 < PATCH || u1 >= DEPTH_W - PATCH ||
                            v1 < PATCH || v1 >= DEPTH_H - PATCH) continue
                        var sad = 0f
                        for (pv in -PATCH until PATCH) for (pu in -PATCH until PATCH)
                            sad += abs(pGray[(v0+pv)*DEPTH_W+(u0+pu)] - cGray[(v1+pv)*DEPTH_W+(u1+pu)])
                        if (sad < bestSad) { bestSad = sad; bu = du; bv = dv }
                    }
                }

                val patchPx = (PATCH * 2f) * (PATCH * 2)
                if (bestSad / patchPx > SAD_THRESH) { v0 += FEAT_STEP; continue }

                val u1 = u0 + bu; val v1 = v0 + bv
                val d1 = cDepth[v1][u1]; if (d1 >= MAX_D - 0.1f) { v0 += FEAT_STEP; continue }

                // Back-project both to world (excluding translation)
                val p0cx = (u0 - CX) * d0 / FX; val p0cy = (v0 - CY) * d0 / FY
                val p1cx = (u1 - CX) * d1 / FX; val p1cy = (v1 - CY) * d1 / FY

                val w0x = pRrel[0]*p0cx + pRrel[1]*p0cy + pRrel[2]*d0
                val w0y = pRrel[3]*p0cx + pRrel[4]*p0cy + pRrel[5]*d0
                val w0z = pRrel[6]*p0cx + pRrel[7]*p0cy + pRrel[8]*d0
                val w1x = cRrel[0]*p1cx + cRrel[1]*p1cy + cRrel[2]*d1
                val w1y = cRrel[3]*p1cx + cRrel[4]*p1cy + cRrel[5]*d1
                val w1z = cRrel[6]*p1cx + cRrel[7]*p1cy + cRrel[8]*d1

                dx += w0x - w1x; dy += w0y - w1y; dz += w0z - w1z
                v0 += FEAT_STEP
            }
            u0 += FEAT_STEP
        }

        return ransac(dx, dy, dz)
    }

    /** 1-point RANSAC: sample one correspondence, count inliers within INLIER_R. */
    private fun ransac(
        dx: List<Float>, dy: List<Float>, dz: List<Float>
    ): Triple<Float, Float, Float> {
        if (dx.size < MIN_INLIERS) return Triple(0f, 0f, 0f)

        var bestCount = 0; var bx = 0f; var by = 0f; var bz = 0f
        val thresh2 = INLIER_R * INLIER_R

        repeat(RANSAC_N) {
            val i = Random.nextInt(dx.size)
            val tx = dx[i]; val ty = dy[i]; val tz = dz[i]

            var cnt = 0; var sx = 0f; var sy = 0f; var sz = 0f
            for (j in dx.indices) {
                val ex = dx[j]-tx; val ey = dy[j]-ty; val ez = dz[j]-tz
                if (ex*ex + ey*ey + ez*ez < thresh2) {
                    cnt++; sx += dx[j]; sy += dy[j]; sz += dz[j]
                }
            }
            if (cnt > bestCount) { bestCount = cnt; bx = sx/cnt; by = sy/cnt; bz = sz/cnt }
        }

        if (bestCount < MIN_INLIERS) return Triple(0f, 0f, 0f)
        val mag = sqrt(bx*bx + by*by + bz*bz)
        return if (mag > MAX_DT) Triple(bx*MAX_DT/mag, by*MAX_DT/mag, bz*MAX_DT/mag)
               else Triple(bx, by, bz)
    }

    // ── Depth map ─────────────────────────────────────────────────────────────

    private fun buildDepth(raw: FloatArray): Array<FloatArray> {
        // Direct fixed-mapping disparity → depth. NO per-frame normalization:
        // that's what made the cone (flat-scene noise stretched across the full
        // depth range). Same MiDaS raw output → same metric depth, every frame.
        var rawMin = Float.MAX_VALUE; var rawMax = -Float.MAX_VALUE
        val depth = Array(DEPTH_H) { v ->
            FloatArray(DEPTH_W) { u ->
                val r = raw[v * DEPTH_W + u]
                if (r < rawMin) rawMin = r
                if (r > rawMax) rawMax = r
                val disp = r.coerceIn(RAW_DISP_MIN, RAW_DISP_MAX)
                (DEPTH_K / disp).coerceIn(DEPTH_MIN_VALID, DEPTH_MAX_VALID)
            }
        }
        // Diagnostic: read this once after a scan to tune RAW_DISP_MIN/MAX/K.
        // If raw range is tiny (e.g., [0.4, 0.6]) the model's output is in a
        // narrow band — adjust the constants so 1/disp lands in your target
        // depth range. If raw range is huge (e.g., [0, 200]) divide by a scale
        // before mapping.
        if (frameCount % 30 == 1) {
            val dNear = DEPTH_K / rawMax
            val dFar  = DEPTH_K / rawMin.coerceAtLeast(1e-4f)
            Log.d(TAG, "MiDaS raw=[%.3f, %.3f] → depth=[%.2f, %.2f]m"
                .format(rawMin, rawMax, dNear, dFar))
        }
        return medianFilterDepth(depth)
    }

    /**
     * Median filter: smooth depth map to reduce noise and isolated pixels.
     * Uses 3×3 neighborhood median.
     */
    private fun medianFilterDepth(depth: Array<FloatArray>): Array<FloatArray> {
        val result = Array(DEPTH_H) { FloatArray(DEPTH_W) }
        val neighborhood = FloatArray(9)

        for (v in 0 until DEPTH_H) {
            for (u in 0 until DEPTH_W) {
                var n = 0
                for (dv in -1..1) {
                    for (du in -1..1) {
                        val nv = (v + dv).coerceIn(0, DEPTH_H - 1)
                        val nu = (u + du).coerceIn(0, DEPTH_W - 1)
                        neighborhood[n++] = depth[nv][nu]
                    }
                }
                neighborhood.sort()
                result[v][u] = neighborhood[4]
            }
        }
        return result
    }

    private fun smoothTemporalDepth(
        current: Array<FloatArray>,
        previous: Array<FloatArray>
    ): Array<FloatArray> {
        val smoothed = Array(DEPTH_H) { FloatArray(DEPTH_W) }
        for (v in 0 until DEPTH_H) {
            for (u in 0 until DEPTH_W) {
                smoothed[v][u] = lerp(previous[v][u], current[v][u], DEPTH_SMOOTH_ALPHA)
            }
        }
        return smoothed
    }

    private fun lerp(a: Float, b: Float, t: Float) = a + (b - a) * t

    private fun prepDepthIn(bmp: Bitmap) {
        val px = IntArray(DEPTH_W * DEPTH_H); bmp.getPixels(px, 0, DEPTH_W, 0, 0, DEPTH_W, DEPTH_H)
        var i = 0
        for (p in px) {
            depthInArr[i++] = ((p shr 16 and 0xFF) / 255f - MEAN[0]) / STD[0]
            depthInArr[i++] = ((p shr  8 and 0xFF) / 255f - MEAN[1]) / STD[1]
            depthInArr[i++] = ((p        and 0xFF) / 255f - MEAN[2]) / STD[2]
        }
        depthInBufs[0].writeFloat(depthInArr)
    }

    private fun toGray(bmp: Bitmap): FloatArray {
        val px = IntArray(DEPTH_W * DEPTH_H); bmp.getPixels(px, 0, DEPTH_W, 0, 0, DEPTH_W, DEPTH_H)
        return FloatArray(px.size) { i ->
            val p = px[i]
            ((p shr 16 and 0xFF) * 0.299f + (p shr 8 and 0xFF) * 0.587f + (p and 0xFF) * 0.114f) / 255f
        }
    }

    private fun gradMag(gray: FloatArray): FloatArray {
        val g = FloatArray(DEPTH_W * DEPTH_H)
        for (v in 1 until DEPTH_H - 1) for (u in 1 until DEPTH_W - 1) {
            val gx = gray[v*DEPTH_W + u+1] - gray[v*DEPTH_W + u-1]
            val gy = gray[(v+1)*DEPTH_W + u] - gray[(v-1)*DEPTH_W + u]
            g[v*DEPTH_W + u] = gx*gx + gy*gy
        }
        return g
    }

    // ── Encoder ───────────────────────────────────────────────────────────────

    private fun runEncoder(bmp: Bitmap): FloatArray {
        val scaled = Bitmap.createScaledBitmap(bmp, ENC_W, ENC_H, true)
        val px = IntArray(ENC_W * ENC_H); scaled.getPixels(px, 0, ENC_W, 0, 0, ENC_W, ENC_H)
        var i = 0
        for (p in px) {
            encInArr[i++] = (p shr 16 and 0xFF) / 255f
            encInArr[i++] = (p shr  8 and 0xFF) / 255f
            encInArr[i++] = (p        and 0xFF) / 255f
        }
        return runCatching {
            encInBufs[0].writeFloat(encInArr)
            models.reno!!.run(encInBufs, encOutBufs)
            encOutBufs[0].readFloat()
        }.getOrElse { simLatent() }
    }

    // ── Simulation fallback ───────────────────────────────────────────────────

    private fun simFrame(sensorR: FloatArray?): List<GaussianPoint> {
        if (initR == null && sensorR != null) initR = sensorR.copyOf()
        val Rrel = sensorR?.let { worldRot(it) }
        val pts  = ArrayList<GaussianPoint>(600)

        fun pt(cx: Float, cy: Float, cz: Float, r: Float, g: Float, b: Float) {
            val wx: Float; val wy: Float; val wz: Float
            if (Rrel != null) {
                wx = Rrel[0]*cx + Rrel[1]*cy + Rrel[2]*cz + worldTx
                wy = Rrel[3]*cx + Rrel[4]*cy + Rrel[5]*cz + worldTy
                wz = Rrel[6]*cx + Rrel[7]*cy + Rrel[8]*cz + worldTz
            } else { wx = cx + worldTx; wy = cy + worldTy; wz = cz + worldTz }
            pts += GaussianPoint(wx, wy, wz, 0.85f, 0.04f, 0.04f, 0.04f, r, g, b)
        }

        for (i in -8..8 step 1) for (j in 0..12 step 1)
            pt(i * 0.5f, 1.4f, j * 0.5f + 1f, 0.76f, 0.71f, 0.64f)  // floor
        for (i in -8..8 step 1) for (j in -2..4 step 1)
            pt(i * 0.5f, j * 0.4f, 5.5f, 0.58f, 0.68f, 0.82f)        // wall
        repeat(60) {
            pt(Random.nextFloat()*8f-4f, Random.nextFloat()*2.5f,
               1f + Random.nextFloat()*4f,
               Random.nextFloat(), Random.nextFloat(), Random.nextFloat())
        }
        return pts
    }

    private fun simLatent() = FloatArray(LATENT_DIM) { Random.nextFloat() * 2f - 1f }

    // ── Matrix utils ──────────────────────────────────────────────────────────

    private fun identity3() = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)

    private fun mat3T(A: FloatArray) = FloatArray(9) { i -> A[(i % 3) * 3 + i / 3] }

    private fun mat3Mul(A: FloatArray, B: FloatArray): FloatArray {
        val C = FloatArray(9)
        for (r in 0..2) for (c in 0..2) for (k in 0..2) C[r*3+c] += A[r*3+k] * B[k*3+c]
        return C
    }

    private fun bbox(gs: List<GaussianPoint>): BoundingBox {
        if (gs.isEmpty()) return BoundingBox(0f, 1f, 0f, 1f, 0f, 1f)
        var x0 = Float.MAX_VALUE; var x1 = -Float.MAX_VALUE
        var y0 = Float.MAX_VALUE; var y1 = -Float.MAX_VALUE
        var z0 = Float.MAX_VALUE; var z1 = -Float.MAX_VALUE
        for (g in gs) {
            if (g.x < x0) x0 = g.x; if (g.x > x1) x1 = g.x
            if (g.y < y0) y0 = g.y; if (g.y > y1) y1 = g.y
            if (g.z < z0) z0 = g.z; if (g.z > z1) z1 = g.z
        }
        return BoundingBox(x0, x1, y0, y1, z0, z1)
    }

    private val Float.f1 get() = "%.1f".format(this)
    private val Float.f2 get() = "%.2f".format(this)
}
