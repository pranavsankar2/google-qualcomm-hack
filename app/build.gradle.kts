plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace   = "com.civilscan.nerf3d"
    compileSdk  = 36

    defaultConfig {
        applicationId   = "com.civilscan.nerf3d"
        minSdk          = 26
        targetSdk       = 36
        versionCode     = 1
        versionName     = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions { jvmTarget = "11" }

    buildFeatures { compose = true }

    packaging {
        jniLibs {
            useLegacyPackaging = true
            pickFirsts += "**/libLiteRt.so"
            pickFirsts += "**/libLiteRtClGlAccelerator.so"
        }
    }
}

// litert-api ships stub classes duplicated by litert core — drop it globally.
configurations.all {
    exclude(group = "com.google.ai.edge.litert", module = "litert-api")
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)

    // Jetpack Compose
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.graphics)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons)
    implementation(libs.compose.activity)
    implementation(libs.compose.viewmodel)
    implementation(libs.compose.navigation)
    implementation(libs.compose.runtime)

    // Lifecycle
    implementation(libs.lifecycle.runtime.ktx)

    // CameraX
    implementation(libs.camerax.camera2)
    implementation(libs.camerax.lifecycle)
    implementation(libs.camerax.view)

    // Google LiteRT (TFLite successor) — NPU inference via NNAPI → Hexagon HTP
    implementation(libs.litert.core)

    // Coroutines
    implementation(libs.kotlinx.coroutines.android)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    debugImplementation(libs.compose.ui.tooling)
    debugImplementation(libs.compose.ui.test.manifest)
}
