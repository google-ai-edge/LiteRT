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
 * An interface to checks if the device is compatible with NPU.
 *
 * Developers can implement this interface to provide their own compatibility check logic, if the
 * default logic is not sufficient.
 */
interface NpuCompatibilityChecker {
  fun isDeviceSupported(): Boolean

  companion object {
    internal val SUPPORTED_QUALCOMM_SOCS =
      setOf(
        Pair("QTI", "SM8750"), // Samsung S25
        Pair("Qualcomm", "SM8750"), // Samsung S25
        Pair("QTI", "SM8650"), // Samsung S24
        Pair("Qualcomm", "SM8650"), // Samsung S24
        Pair("QTI", "SM8550"), // Samsung S23
        Pair("Qualcomm", "SM8550"), // Samsung S23
      )

    /** Qualcomm NPU compatibility checker. */
    val Qualcomm =
      object : NpuCompatibilityChecker {
        override fun isDeviceSupported(): Boolean {
          if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            return SUPPORTED_QUALCOMM_SOCS.contains(Pair(Build.SOC_MANUFACTURER, Build.SOC_MODEL))
          }
          return false
        }
      }

    // Medatek SOCs are only supported on Android 15 devices (API level 35), for now.
    internal val SUPPORTED_MEDIATEK_SOCS =
      setOf(
        Triple("Mediatek", "MT6878", 35),
        Triple("Mediatek", "MT6897", 35),
        Triple("Mediatek", "MT6983", 35),
        Triple("Mediatek", "MT6985", 35),
        Triple("Mediatek", "MT6989", 35),
        Triple("Mediatek", "MT6991", 35),
      )

    /** Mediatek NPU compatibility checker. */
    val Mediatek =
      object : NpuCompatibilityChecker {
        override fun isDeviceSupported(): Boolean {
          if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            return SUPPORTED_MEDIATEK_SOCS.contains(
              Triple(Build.SOC_MANUFACTURER, Build.SOC_MODEL, Build.VERSION.SDK_INT)
            )
          }
          return false
        }
      }

    internal val SUPPORTED_GOOGLE_SOCS =
      setOf(Pair("Google", "Tensor G3"), Pair("Google", "Tensor G4"), Pair("Google", "Tensor G5"))

    /** Google Tensor NPU compatibility checker. */
    val GoogleTensor =
      object : NpuCompatibilityChecker {
        override fun isDeviceSupported(): Boolean {
          // Google Tensor NPU is only supported on Android 16+ devices (API level 36).
          if (Build.VERSION.SDK_INT >= 36) {
            // BP2A is the only Android 16 build ID that does not support NPU.
            return SUPPORTED_GOOGLE_SOCS.contains(Pair(Build.SOC_MANUFACTURER, Build.SOC_MODEL)) &&
              !Build.ID.startsWith("BP2A")
          }
          return false
        }
      }

    /** Default NPU compatibility checker for all vendors. */
    val Default =
      object : NpuCompatibilityChecker {
        override fun isDeviceSupported(): Boolean {
          return Qualcomm.isDeviceSupported() ||
            Mediatek.isDeviceSupported() ||
            GoogleTensor.isDeviceSupported()
        }
      }
  }
}

/** An interface to provide the NPU libraries. */
interface NpuAcceleratorProvider {
  /** Returns true if the device is compatible with NPU library.. */
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
