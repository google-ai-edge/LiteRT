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
}

/**
 * The default implementation of [NpuCompatibilityChecker], which is based on the SoC list supported
 * by vendors.
 */
internal class DefaultNpuCompatibilityChecker : NpuCompatibilityChecker {
  override fun isDeviceSupported(): Boolean {
    // Build.SOC_MANUFACTURER and Build.SOC_MODEL is only available on Android S+
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
      return SUPPORTED_SOCS.contains(Pair(Build.SOC_MANUFACTURER, Build.SOC_MODEL))
    }
    return false
  }

  companion object {
    // TODO(niuchl): get full list of supported SoCs.
    private val SUPPORTED_SOCS =
      setOf(
        // Pair("Google", "Tensor G3"), // Pixel 8
        // Pair("Google", "Tensor G4"), // Pixel 9
        Pair("QTI", "SM8650"), // Samsung S24
        Pair("QTI", "SM8550"), // Samsung S23
      )
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
  private val npuCompatibilityChecker: NpuCompatibilityChecker = DefaultNpuCompatibilityChecker(),
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
