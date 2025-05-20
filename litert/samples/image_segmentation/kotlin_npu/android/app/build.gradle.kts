/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.jetbrains.kotlin.android)
  alias(libs.plugins.undercouch.download)
  alias(libs.plugins.compose.compiler)
}

android {
  namespace = "com.google.ai.edge.examples.image_segmentation"
  compileSdk = 34

  defaultConfig {
    applicationId = "com.google.ai.edge.examples.image_segmentation"
    minSdk = 31
    targetSdk = 33
    versionCode = 1
    versionName = "1.0"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    vectorDrawables { useSupportLibrary = true }

    // NPU only supports arm64-v8a
    ndk { abiFilters.add("arm64-v8a") }
    // Needed for Qualcomm NPU runtimes
    packaging { jniLibs { useLegacyPackaging = true } }
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_1_8
    targetCompatibility = JavaVersion.VERSION_1_8
  }
  kotlinOptions { jvmTarget = "1.8" }
  buildFeatures { compose = true }
  packaging { resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" } }

  // AI packs
  assetPacks.add(":ai_pack:selfie_multiclass")
  assetPacks.add(":ai_pack:selfie_multiclass_mtk")

  // NPU runtime libraries
  dynamicFeatures.add(":litert_npu_runtime_libraries:mediatek_runtime")
  dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v69")
  dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v73")
  dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v75")
  dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v79")

  bundle {
    deviceTargetingConfig = file("device_targeting_configuration.xml")
    deviceGroup {
      enableSplit = true // split bundle by #group
      defaultGroup = "other" // group used for standalone APKs
    }
  }

  // Disable lint analysis to avoid build failures due to lint errors.
  lint {
    disable.add("CoroutineCreationDuringComposition")
    disable.add("FlowOperatorInvokedInComposition")
    disable.add("Aligned16KB")
  }
}

// Import DownloadModels task
project.extensions.extraProperties["ASSET_DIR"] = "$projectDir/src/main/assets"

dependencies {
  // Strings for NPU runtime libraries
  implementation(project(":litert_npu_runtime_libraries:runtime_strings"))

  implementation(libs.litert)

  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.viewmodel.compose)
  implementation(libs.androidx.activity.compose)
  implementation(platform(libs.androidx.compose.bom))
  implementation(libs.androidx.ui)
  implementation(libs.androidx.ui.graphics)
  implementation(libs.androidx.ui.tooling.preview)
  implementation(libs.androidx.material.icons.core)
  implementation(libs.androidx.material.icons.extended)
  implementation(libs.androidx.material2)
  implementation(libs.androidx.camera.core)
  implementation(libs.androidx.camera.lifecycle)
  implementation(libs.androidx.camera.view)
  implementation(libs.androidx.camera.camera2)
  implementation(libs.coil.compose)
  implementation(libs.androidx.compose.runtime.livedata)
  implementation(libs.android.play.ai.delivery)
  implementation(platform(libs.kotlinx.coroutines.bom))
  implementation(libs.kotlinx.coroutines.android)

  testImplementation(libs.junit)
  androidTestImplementation(libs.androidx.junit)
  androidTestImplementation(libs.androidx.espresso.core)
  androidTestImplementation(platform(libs.androidx.compose.bom))
  androidTestImplementation(libs.androidx.ui.test.junit4)
  debugImplementation(libs.androidx.ui.tooling)
  debugImplementation(libs.androidx.ui.test.manifest)
}
