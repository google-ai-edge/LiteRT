/*
 * Copyright 2025 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.litert

import android.content.Context
import android.os.Build

/**
 * An interface to check if the device is compatible with NPU.
 *
 * Developers can implement this interface to provide their own compatibility check logic, if the
 * default logic is not sufficient.
 */
interface NpuCompatibilityChecker {
  fun isDeviceSupported(): Boolean

  companion object {
    private const val GOOGLE_TENSOR_MIN_SDK_INT = 36

    private fun isQualcommDevice(): Boolean {
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val manufacturer = Build.SOC_MANUFACTURER.trim()
        return "Qualcomm".equals(manufacturer, ignoreCase = true) ||
          "QTI".equals(manufacturer, ignoreCase = true)
      }
      return false
    }

    private fun isMediaTekDevice(): Boolean {
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val manufacturer = Build.SOC_MANUFACTURER.trim()
        return "Mediatek".equals(manufacturer, ignoreCase = true) ||
          "MTK".equals(manufacturer, ignoreCase = true)
      }
      return false
    }

    private fun isGoogleTensorDevice(): Boolean {
      // Google Tensor NPU is only supported on Android 16+ devices (API level 36).
      // BP2A is the only Android 16 build ID that does not support NPU.
      if (Build.VERSION.SDK_INT >= GOOGLE_TENSOR_MIN_SDK_INT && !Build.ID.startsWith("BP2A")) {
        val manufacturer = Build.SOC_MANUFACTURER.trim()
        val model = Build.SOC_MODEL.trim()
        val hardware = Build.HARDWARE
        return "Google".equals(manufacturer, ignoreCase = true) ||
          model.contains("Tensor", ignoreCase = true) ||
          hardware.contains("tensor", ignoreCase = true)
      }
      return false
    }

    private fun isSamsungDevice(): Boolean {
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val manufacturer = Build.SOC_MANUFACTURER.trim()
        return manufacturer.startsWith("Samsung", ignoreCase = true) ||
          "Exynos".equals(manufacturer, ignoreCase = true)
      }
      return false
    }

    /** Qualcomm NPU compatibility checker. */
    val Qualcomm =
      object : NpuCompatibilityChecker {
        override fun isDeviceSupported(): Boolean = isQualcommDevice()
      }

    /** Mediatek NPU compatibility checker. */
    val Mediatek =
      object : NpuCompatibilityChecker {
        override fun isDeviceSupported(): Boolean = isMediaTekDevice()
      }

    /** Google Tensor NPU compatibility checker. */
    val GoogleTensor =
      object : NpuCompatibilityChecker {
        override fun isDeviceSupported(): Boolean = isGoogleTensorDevice()
      }

    /** Samsung NPU compatibility checker. */
    val Samsung =
      object : NpuCompatibilityChecker {
        override fun isDeviceSupported(): Boolean = isSamsungDevice()
      }

    /** Default NPU compatibility checker for all vendors. */
    val Default =
      object : NpuCompatibilityChecker {
        override fun isDeviceSupported(): Boolean {
          return Qualcomm.isDeviceSupported() ||
            Mediatek.isDeviceSupported() ||
            GoogleTensor.isDeviceSupported() ||
            Samsung.isDeviceSupported()
        }
      }
  }
}

/** An interface to provide the NPU libraries. */
interface NpuAcceleratorProvider {
  /** Returns true if the device is compatible with NPU library. */
  fun isDeviceSupported(): Boolean

  /** Returns true if the NPU library is ready to use. */
  fun isLibraryReady(): Boolean

  /** Downloads the NPU library if needed. */
  suspend fun downloadLibrary()

  /** Returns the local directory of the NPU library. */
  fun getLibraryDir(): String
}

/**
 * An implementation of [NpuAcceleratorProvider], which provides the NPU libraries without dynamic
 * downloading.
 *
 * This implementation is for apps with built-in NPU libraries, or with NPU libraries delivered as
 * "install-time" Google Play Feature modules.
 */
class BuiltinNpuAcceleratorProvider
@JvmOverloads
constructor(
  private val context: Context,
  private val npuCompatibilityChecker: NpuCompatibilityChecker = NpuCompatibilityChecker.Default,
) : NpuAcceleratorProvider {
  override fun isDeviceSupported(): Boolean {
    return npuCompatibilityChecker.isDeviceSupported()
  }

  override fun isLibraryReady() = true

  override suspend fun downloadLibrary() {}

  override fun getLibraryDir(): String {
    return context.applicationInfo.nativeLibraryDir
  }
}
