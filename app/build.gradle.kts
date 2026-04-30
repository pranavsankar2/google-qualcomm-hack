plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.qnn_litertlm_gemma"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.qnn_litertlm_gemma"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        viewBinding = true
    }
    packaging {
        jniLibs {
            useLegacyPackaging = true
            // litert:2.1.4 and litertlm-android bundle the same libLiteRt*.so files
            // across every ABI directory.  Use ** to match arm64-v8a, x86_64, etc.
            pickFirsts += "**/libLiteRt.so"
            pickFirsts += "**/libLiteRtClGlAccelerator.so"
        }
    }
}

// litert-api ships the same org.tensorflow.lite.* stubs that litert:2.1.4 ships as
// full classes — drop the stub artifact globally to eliminate the duplicate-class error.
configurations.all {
    exclude(group = "com.google.ai.edge.litert", module = "litert-api")
}

dependencies {
    // Core Android
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.recyclerview)
    implementation(libs.androidx.security.crypto)

    // LiteRT-LM — powers the LLM Engine (unchanged)
    implementation(libs.litertlm.android)

    // LiteRT core — provides org.tensorflow.lite.Interpreter for scanner models.
    // The two native .so files it adds are deduplicated via packagingOptions.pickFirsts above.
    implementation(libs.litert.core)

    // CameraX — continuous YUV frame capture
    implementation(libs.camerax.camera2)
    implementation(libs.camerax.lifecycle)
    implementation(libs.camerax.view)

    // Coroutines
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.androidx.lifecycle.runtime.ktx)

    // Testing
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
