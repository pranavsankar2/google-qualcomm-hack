# On-Device 3D Spatial Mapping

## Application Description

**On-Device 3D Spatial Mapping** is an Android mobile application that enables rapid structural engineering assessment through AI-powered 3D scanning. It captures high-fidelity point clouds using mobile device cameras and analyzes them for civil engineering metrics to support infrastructure inspection, damage assessment, and structural monitoring.

### Key Features

- **Real-Time 3D Scanning**: Captures 3D point clouds via camera + MiDaS depth estimation
- **SLAM-Based Pose Tracking**: Tracks camera position using game rotation vectors and optical flow
- **Neural Compression**: Compresses point clouds 8-12× using EfficientNet encoder
- **Structural Analysis**: Computes:
  - Scene dimensions (length, width, height, volume)
  - Point density and scan coverage metrics
  - Surface planarity & roughness via PCA covariance
  - Structural defect detection (cracks, moisture, voids, settlement)
- **NPU Acceleration**: Offloads LiteRT model inference to Qualcomm Hexagon HTP for efficient on-device processing
- **Detailed Reporting**: Generates structural quality score (0–100) with flagged defects and severity levels

### Technology Stack

- **Language**: Kotlin + Jetpack Compose (UI)
- **Cameras & Video**: CameraX
- **ML Inference**: LiteRT with NPU acceleration
- **Models**: MiDaS v2.1 (depth), EfficientNet-Lite0 (compression)
- **Async**: Kotlin Coroutines
- **Min SDK**: Android 8.0 (API 26) | Target SDK: Android 15 (API 36)
- **Architecture**: ARM64 (arm64-v8a)

---

## Team

| Name | Email |
|------|-------|
| [Pranav Sankar] | [pranavsankar2@gmail.com] |
| [Akshar Nana] | [akshar.nana.za@gmail.com] |
| [Anirudh Kondapaneni] | [anirudhk3002@gmail.com]

---

## Setup Instructions

### Prerequisites

- **Android Studio** (2024.1 or later)
- **Android SDK**: API 36 (latest) with build tools
- **JDK 11** or later
- **Gradle 8.7+** (included via Gradle wrapper)
- **Android Device or Emulator**: API 26+ with ARM64 support (recommended: Snapdragon device for NPU acceleration)

### Step 1: Clone the Repository

```bash
git clone https://github.com/pranavsankar2/google-qualcomm-hack.git
cd google-qualcomm-hack
```

### Step 2: Models

The LiteRT models are placed in the `models/` directory:

### Step 3: Build the Application

```bash
# Build via Gradle wrapper
./gradlew :app:assembleDebug
```

### Step 5: Install Dependencies

The project uses managed dependencies via Gradle (see `app/build.gradle.kts`). Key libraries:

- Jetpack Compose (UI framework)
- CameraX (camera access)
- LiteRT (ML inference)
- Kotlin Coroutines (async runtime)

These are automatically resolved during the build.

---

## Run and Usage Instructions

### Running on Device
#### Via Command Line

```bash
# Push the models to the phone
adb push models/mobile_gs_quant.tflite /sdcard/Android/data/com.civilscan.nerf3d/files/models/

adb push models/reno_quant.tflite /sdcard/Android/data/com.civilscan.nerf3d/files/models/

# Install the app on phone
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### Using the Application

1. **Launch the app** on your device
2. **Grant camera permissions** when prompted
3. **Aim at the structure** you want to scan (building facade, pavement, foundation, etc.)
4. **Capture frames** by holding the camera steadily aimed at the area (5–30 seconds recommended)
5. **Processing** occurs automatically:
   - Depth frames are estimated
   - Point cloud is built incrementally
   - SLAM tracks camera pose
6. **View results**:
   - Live point cloud visualization
7. **Export** results for structural reports

### References
1. MiDaS v2.1 - Depth Estimation

Paper: "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer"
Authors: Ranftl, R., Lasinger, K., Hafner, D., et al.
URL: https://arxiv.org/abs/1907.01341
Used for: Real-time monocular depth estimation from RGB images
Model: mobile_gs_quant.tflite (MobileNet backbone variant)
EfficientNet-Lite0 - Compression Encoder

2. Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
Authors: Tan, M., & Le, Q. V.
URL: https://arxiv.org/abs/1905.11946
Used for: Scene embedding & point cloud compression (RENO encoder)
Variant: EfficientNet-Lite (mobile-optimized)
