#!/usr/bin/env python3
"""
Download MiDaS v2.1 Small TFLite (monocular depth estimation) and
EfficientNet-Lite0 feature-vector TFLite (scene encoder) for CivilScan 3D.

  depth_model.tflite  — MiDaS v2.1 Small
      in : [1, 256, 256, 3]  float32  ImageNet-normalised
      out: [1, 256, 256, 1]  float32  inverse relative disparity
      size: ~5 MB

  reno_quant.tflite   — EfficientNet-Lite0 feature vectors
      in : [1, 224, 224, 3]  float32  [0, 1] normalised
      out: [1, 1280]         float32  pooled feature vector
      size: ~18 MB
"""

import os, sys, subprocess, time, urllib.request

MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
DEVICE_PATH  = "/sdcard/Android/data/com.civilscan.nerf3d/files/models/"
PACKAGE      = "com.civilscan.nerf3d"

# MiDaS v2.1 Small from TF Hub (Intel published, 256×256, ~5 MB quantised)
MIDAS_URLS = [
    # Primary — TF Hub lite-model storage bucket
    "https://storage.googleapis.com/tfhub-lite-models/intel/lite-model/midas/v2_1_small/1/lite/2.tflite",
    # Fallback — TF Hub redirect
    "https://tfhub.dev/intel/lite-model/midas/v2_1_small/1/lite/2?lite-format=tflite",
]

EFFICIENTNET_URLS = [
    "https://storage.googleapis.com/tfhub-lite-models/tensorflow/lite-model/efficientnet/lite0/feature_vectors/2/default/1.tflite",
    "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/feature_vectors/2/default/1?lite-format=tflite",
]

os.makedirs(MODELS_DIR, exist_ok=True)


def download(urls: list, dest: str, label: str) -> bool:
    print(f"\n── {label}")
    for url in urls:
        print(f"   trying: {url}")
        def progress(b, bs, total):
            if total > 0:
                print(f"\r   {min(b*bs*100//total,100)}%  ", end="", flush=True)
        try:
            urllib.request.urlretrieve(url, dest, reporthook=progress)
            kb = os.path.getsize(dest) // 1024
            print(f"\r   ✓ {kb:,} KB → {dest}")
            return True
        except Exception as e:
            print(f"\r   ✗ {e}")
    return False


def make_stub_depth():
    """
    Generates a MiDaS-shaped stub model using tensorflow.
    Output is a constant gradient so the depth map renders visually.
    pip install tensorflow
    """
    print("\n── Generating MiDaS stub (requires tensorflow) ───────────────────")
    try:
        import tensorflow as tf
        import numpy as np
    except ImportError:
        print("   pip install tensorflow  then re-run with --stub")
        return False

    @tf.function(input_signature=[tf.TensorSpec([1, 256, 256, 3], tf.float32)])
    def midas_stub(x):
        # Return horizontal gradient so depth map shows colour variation
        col = tf.cast(tf.range(256), tf.float32) / 255.0
        depth = tf.reshape(col, [1, 1, 256, 1])
        depth = tf.tile(depth, [1, 256, 1, 1])
        return depth

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [midas_stub.get_concrete_function()])
    tflite = converter.convert()
    dest = os.path.join(MODELS_DIR, "depth_model.tflite")
    with open(dest, "wb") as f:
        f.write(tflite)
    print(f"   ✓ {len(tflite)//1024} KB stub → {dest}")
    return True


def fix_device_permissions():
    """Delete the shell-owned models dir so the app can recreate it as the app user."""
    print("\n── Fixing device directory permissions ───────────────────────────")
    subprocess.run(["adb", "shell", f"rm -rf {DEVICE_PATH}"], capture_output=True)
    print("   Launching app to create models dir as app user…")
    subprocess.run([
        "adb", "shell", "am", "start", "-n", f"{PACKAGE}/.MainActivity"
    ], capture_output=True)
    time.sleep(5)
    result = subprocess.run(
        ["adb", "shell", f"ls -la {DEVICE_PATH}"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("   ✓ models dir exists and is app-owned:")
        for line in result.stdout.splitlines()[:3]:
            print("    ", line)
    else:
        print("   ⚠ dir not created yet — make sure app is open, then push manually")


def push_models():
    print("\n── Pushing models to device ──────────────────────────────────────")
    for fname in ["depth_model.tflite", "reno_quant.tflite"]:
        src = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(src):
            print(f"   skip {fname} (not found locally)")
            continue
        r = subprocess.run(
            ["adb", "push", src, DEVICE_PATH],
            capture_output=True, text=True
        )
        if r.returncode == 0:
            print(f"   ✓ {fname}  ({os.path.getsize(src)//1024:,} KB)")
        else:
            print(f"   ✗ {fname}: {r.stderr.strip()}")


def restart_app():
    print("\n── Restarting app ────────────────────────────────────────────────")
    subprocess.run(["adb", "shell", "am", "force-stop", PACKAGE])
    time.sleep(1)
    subprocess.run(["adb", "shell", "am", "start", "-n", f"{PACKAGE}/.MainActivity"])
    print("   App restarted. Check logcat:")
    print("   adb logcat -s ModelManager:*")
    print("   Expect: ✓ Loaded depth_model.tflite  in=[1, 256, 256, 3]  out=[1, 256, 256, 1]")


def verify_local():
    print("\n── Local models ──────────────────────────────────────────────────")
    files = {
        "depth_model.tflite": (1_000, 15_000),   # 1–15 MB for MiDaS small
        "reno_quant.tflite":  (5_000, 25_000),   # 5–25 MB for EfficientNet
    }
    for fname, (lo, hi) in files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            kb = os.path.getsize(path) // 1024
            warn = " ⚠ size looks wrong (wrong model?)" if kb < lo or kb > hi else ""
            print(f"   ✓ {fname}  {kb:,} KB{warn}")
        else:
            print(f"   ✗ {fname}  NOT FOUND")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stub",      action="store_true", help="Generate stub depth model (needs tensorflow)")
    p.add_argument("--push-only", action="store_true", help="Skip download, just push existing local files")
    args = p.parse_args()

    if args.push_only:
        verify_local()
        fix_device_permissions()
        push_models()
        restart_app()
        sys.exit(0)

    midas_ok = download(MIDAS_URLS, os.path.join(MODELS_DIR, "depth_model.tflite"),
                        "MiDaS v2.1 Small (depth estimator)")
    if not midas_ok:
        if args.stub:
            midas_ok = make_stub_depth()
        else:
            print("\n   Download failed. Try: python3 scripts/setup_models.py --stub")

    reno_ok = download(EFFICIENTNET_URLS, os.path.join(MODELS_DIR, "reno_quant.tflite"),
                       "EfficientNet-Lite0 feature vectors (scene encoder)")

    verify_local()
    fix_device_permissions()
    push_models()
    restart_app()
