plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.rtst.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.rtst.app"
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
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
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    sourceSets {
        getByName("main") {
            kotlin.srcDir("src/main/kotlin")
        }
    }
}

dependencies {
    // ExecuTorch runtime (Maven Central — stable)
    // To use nightly: download AAR from https://ossci-android.s3.amazonaws.com/executorch/release/snapshot-{YYYYMMDD}/executorch.aar
    //   put in app/libs/, and use: implementation(files("libs/executorch.aar"))
    //   plus: implementation("com.facebook.soloader:soloader:0.10.5")
    //   plus: implementation("com.facebook.fbjni:fbjni:0.7.0")
    // implementation("org.pytorch:executorch-android:1.1.0")
    implementation("org.pytorch:executorch-android-vulkan:1.1.0")

    // CameraX (for real-time video style transfer)
    val cameraXVersion = "1.4.1"
    implementation("androidx.camera:camera-core:$cameraXVersion")
    implementation("androidx.camera:camera-camera2:$cameraXVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraXVersion")
    implementation("androidx.camera:camera-view:$cameraXVersion")

    // AndroidX
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.2.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}
