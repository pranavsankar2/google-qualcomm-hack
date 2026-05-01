"""
Convert Mobile-GS and RENO PyTorch checkpoints to INT8-quantised TFLite
for on-device NPU inference via Google LiteRT / NNAPI.

Usage:
    pip install tensorflow torch onnx onnx-tf
    python scripts/convert_to_tflite.py --gs  path/to/mobile_gs.pt
                                         --reno path/to/reno.pt

If you only want to verify the NPU pipeline with stub models (correct I/O shapes,
garbage outputs) run without arguments:
    python scripts/convert_to_tflite.py --stub
"""

import argparse
import sys
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Stub model generator (no PyTorch required) ────────────────────────────────

def make_stub_models():
    """
    Creates minimal TFLite models with the exact I/O shapes CivilScan expects.
    Outputs are scaled versions of the input mean — garbage data but valid tensors.
    Use these to verify the NNAPI/HTP delegate path fires without real checkpoints.
    """
    try:
        import tensorflow as tf
    except ImportError:
        sys.exit("pip install tensorflow")

    print("Building stub Mobile-GS  [1,224,224,3] → [1,85000] …")

    @tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32)])
    def gs_fn(x):
        # Pool → scale → tile to 85 000 (= 5 000 gaussians × 17 attrs)
        mean = tf.reduce_mean(x)                        # scalar
        return tf.fill([1, 85_000], mean * 0.1)         # all attrs ~0 → sigmoid(0)=0.5

    _export_tflite(gs_fn.get_concrete_function(), "mobile_gs_quant.tflite")

    print("Building stub RENO  [1,262144] → [1,1024] …")

    @tf.function(input_signature=[tf.TensorSpec([1, 262_144], tf.float32)])
    def reno_fn(x):
        mean = tf.reduce_mean(x)
        return tf.fill([1, 1_024], mean)

    _export_tflite(reno_fn.get_concrete_function(), "reno_quant.tflite")

    _push_to_device()


def _export_tflite(concrete_fn, filename):
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    path = os.path.join(OUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(tflite_model)
    kb = os.path.getsize(path) // 1024
    print(f"  ✓ {path}  ({kb} KB)")


# ── Real model conversion via ONNX ────────────────────────────────────────────

def convert_gs(checkpoint_path: str):
    """PyTorch Mobile-GS → ONNX → TFLite INT8."""
    try:
        import torch, onnx
        import tensorflow as tf
    except ImportError:
        sys.exit("pip install torch onnx onnx-tf tensorflow")

    print(f"Loading Mobile-GS from {checkpoint_path} …")
    model = torch.load(checkpoint_path, map_location="cpu")
    model.eval()

    dummy = torch.zeros(1, 3, 224, 224)
    onnx_path = os.path.join(OUT_DIR, "mobile_gs.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes=None,
        opset_version=17
    )
    print(f"  ONNX written → {onnx_path}")
    _onnx_to_tflite(onnx_path, "mobile_gs_quant.tflite", [1, 224, 224, 3])


def convert_reno(checkpoint_path: str):
    """PyTorch RENO → ONNX → TFLite INT8."""
    try:
        import torch, onnx
        import tensorflow as tf
    except ImportError:
        sys.exit("pip install torch onnx onnx-tf tensorflow")

    print(f"Loading RENO from {checkpoint_path} …")
    model = torch.load(checkpoint_path, map_location="cpu")
    model.eval()

    dummy = torch.zeros(1, 1, 64, 64, 64)   # voxel grid [B,C,D,H,W]
    onnx_path = os.path.join(OUT_DIR, "reno.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["voxels"], output_names=["latent"],
        dynamic_axes=None, opset_version=17
    )
    print(f"  ONNX written → {onnx_path}")
    # RENO input flattened to [1,262144] in the app
    _onnx_to_tflite(onnx_path, "reno_quant.tflite", [1, 262_144])


def _onnx_to_tflite(onnx_path: str, out_name: str, input_shape: list):
    """ONNX → SavedModel → INT8 TFLite via onnx-tf."""
    import subprocess, tensorflow as tf
    from onnx_tf.backend import prepare
    import onnx

    sm_dir = onnx_path.replace(".onnx", "_savedmodel")
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(sm_dir)
    print(f"  SavedModel → {sm_dir}")

    # Quantise to INT8
    def representative_dataset():
        import numpy as np
        for _ in range(100):
            yield [np.random.rand(*input_shape).astype("float32")]

    converter = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    path = os.path.join(OUT_DIR, out_name)
    with open(path, "wb") as f:
        f.write(tflite_model)
    mb = os.path.getsize(path) / 1_048_576
    print(f"  ✓ INT8 TFLite → {path}  ({mb:.1f} MB)")


# ── ADB push helper ───────────────────────────────────────────────────────────

def _push_to_device():
    import subprocess
    device_path = "/sdcard/Android/data/com.civilscan.nerf3d/files/models/"
    print(f"\nPushing models to device …  ({device_path})")
    subprocess.run(["adb", "shell", "mkdir", "-p", device_path])
    for fname in ["mobile_gs_quant.tflite", "reno_quant.tflite"]:
        src = os.path.join(OUT_DIR, fname)
        if os.path.exists(src):
            r = subprocess.run(["adb", "push", src, device_path], capture_output=True, text=True)
            if r.returncode == 0:
                print(f"  ✓ {fname}")
            else:
                print(f"  ✗ {fname}: {r.stderr.strip()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub", action="store_true",
        help="Generate stub models with correct I/O shapes (no checkpoint required)")
    parser.add_argument("--gs",   metavar="PATH", help="Path to Mobile-GS .pt checkpoint")
    parser.add_argument("--reno", metavar="PATH", help="Path to RENO .pt checkpoint")
    parser.add_argument("--push", action="store_true", default=True,
        help="Auto-push to connected device after conversion (default: true)")
    args = parser.parse_args()

    if args.stub:
        make_stub_models()
    else:
        if not args.gs and not args.reno:
            print(__doc__)
            print("\nTip: run with --stub to generate test models without checkpoints.")
            sys.exit(0)
        if args.gs:
            convert_gs(args.gs)
        if args.reno:
            convert_reno(args.reno)
        if args.push:
            _push_to_device()

    print("\nDone. Restart the app — NPU indicator should show NNAPI/HTP.")
