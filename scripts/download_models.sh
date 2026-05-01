#!/bin/bash
# Download plug-and-play TFLite models for CivilScan 3D and push to device.
#
# Model 1 — MiDaS v2.1 small  (depth estimation → 3D point cloud)
#   Input  : [1, 256, 256, 3]  float32  HWC  (ImageNet-normalised)
#   Output : [1, 256, 256, 1]  float32  (inverse relative depth)
#
# Model 2 — EfficientNet-Lite0 FP32  (scene encoder / neural compression)
#   Input  : [1, 224, 224, 3]  float32  HWC  (0-1 range)
#   Output : [1, 1280]         float32  (pooled feature vector)
#
# Both route to the Qualcomm Hexagon HTP via NNAPI on the Galaxy S25 Ultra.

set -e
MODELS_DIR="$(dirname "$0")/../models"
DEVICE_PATH="/sdcard/Android/data/com.civilscan.nerf3d/files/models/"

mkdir -p "$MODELS_DIR"

echo ""
echo "── Model 1: MiDaS v2.1 small (depth estimation) ──────────────────────────"
curl -L --progress-bar \
  "https://github.com/isl-org/MiDaS/releases/download/v2_1/model_opt.tflite" \
  -o "$MODELS_DIR/mobile_gs_quant.tflite"
echo "   ✓ $(du -sh "$MODELS_DIR/mobile_gs_quant.tflite" | cut -f1)  →  mobile_gs_quant.tflite"

echo ""
echo "── Model 2: EfficientNet-Lite0 FP32 (scene encoder) ──────────────────────"
curl -L --progress-bar \
  "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/fp32/2?lite-format=tflite" \
  -o "$MODELS_DIR/reno_quant.tflite"
echo "   ✓ $(du -sh "$MODELS_DIR/reno_quant.tflite" | cut -f1)  →  reno_quant.tflite"

echo ""
echo "── Pushing to device ──────────────────────────────────────────────────────"
adb shell mkdir -p "$DEVICE_PATH"
adb push "$MODELS_DIR/mobile_gs_quant.tflite" "$DEVICE_PATH"
adb push "$MODELS_DIR/reno_quant.tflite"      "$DEVICE_PATH"

echo ""
echo "Done. Restart the app — the HUD should show  NNAPI/HTP  instead of  SIM."
