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

/** Environment to hold configuration options for LiteRT runtime. */
class Environment private constructor(handle: Long) : JniHandle(handle) {

  /**
   * Options configurable in LiteRT environment.
   *
   * NOTE: the values need to be consistent with the constants in `LiteRtEnvOptionTag` in
   * litert/c/litert_environment_options.h.
   */
  // LINT.IfChange(Option)
  enum class Option private constructor(val value: Int) {
    CompilerPluginLibraryDir(0),
    DispatchLibraryDir(1),
    /**
     * This for internal use only.
     *
     * @suppress
     */
    SystemRuntimeHandle(23),
  }
  // LINT.ThenChange(../../../../../../../../../c/litert_environment_options.h)

  override protected fun destroy() {
    nativeDestroy(handle)
  }

  /** Returns the set of accelerators available in the environment. */
  @Throws(LiteRtException::class)
  fun getAvailableAccelerators(): Set<Accelerator> {
    assertNotDestroyed()

    val accelerators = nativeGetAvailableAccelerators(handle)
    return accelerators.map { Accelerator.of(it) }.toSet()
  }

  companion object {
    // LINT.IfChange(ENV_OPTION_TAG_COMPILER_CACHE_DIR)
    private const val ENV_OPTION_TAG_COMPILER_CACHE_DIR = 18
    // LINT.ThenChange(../../../../../../../../../c/litert_environment_options.h)

    init {
      System.loadLibrary("litert_jni")
    }

    /**
     * Creates an environment.
     *
     * @param context The Android context.
     * @param options The options to configure the environment.
     * @param enableCompilerCache Whether to enable the compiler cache.
     */
    @Throws(LiteRtException::class)
    @JvmOverloads
    @JvmStatic
    fun create(
      context: Context,
      options: Map<Option, String> = mapOf(),
      enableCompilerCache: Boolean = true,
    ): Environment {
      val keys = options.keys.map { it.value }.toMutableList()
      val values = options.values.toMutableList()

      if (enableCompilerCache) {
        context.cacheDir?.absolutePath?.let { path ->
          keys.add(ENV_OPTION_TAG_COMPILER_CACHE_DIR)
          values.add(path)
        }
      }

      return Environment(nativeCreate(keys.toIntArray(), values.toTypedArray()))
    }

    /**
     * Creates an environment.
     *
     * Note: This version does not enable compiler caching. Use [create(Context, Map, Boolean)]
     * to enable caching.
     *
     * @param options The options to configure the environment.
     */
    @Throws(LiteRtException::class)
    @JvmOverloads
    @JvmStatic
    fun create(options: Map<Option, String> = mapOf()): Environment {
      return Environment(
        nativeCreate(options.keys.map { it.value }.toIntArray(), options.values.toTypedArray())
      )
    }

    // TODO(niuchl): Add support for downloading NPU library.
    /**
     * Creates an environment with a [NpuAcceleratorProvider], which provides the NPU libraries.
     *
     * @param context The Android context.
     * @param npuAcceleratorProvider The NPU accelerator provider.
     * @param options The options to configure the environment.
     * @param enableCompilerCache Whether to enable the compiler cache.
     */
    @Throws(LiteRtException::class)
    @JvmOverloads
    @JvmStatic
    fun create(
      context: Context,
      npuAcceleratorProvider: NpuAcceleratorProvider,
      options: Map<Option, String> = mapOf(),
      enableCompilerCache: Boolean = true,
    ): Environment {
      val mutableOptions = options.toMutableMap()
      if (npuAcceleratorProvider.isDeviceSupported() && npuAcceleratorProvider.isLibraryReady()) {
        mutableOptions[Option.DispatchLibraryDir] = npuAcceleratorProvider.getLibraryDir()
        mutableOptions[Option.CompilerPluginLibraryDir] = npuAcceleratorProvider.getLibraryDir()
      }

      val keys = mutableOptions.keys.map { it.value }.toMutableList()
      val values = mutableOptions.values.toMutableList()

      if (enableCompilerCache) {
        context.cacheDir?.absolutePath?.let { path ->
          keys.add(ENV_OPTION_TAG_COMPILER_CACHE_DIR)
          values.add(path)
        }
      }

      return Environment(
        nativeCreate(
          keys.toIntArray(),
          values.toTypedArray(),
        )
      )
    }

    /**
     * Creates an environment with a [NpuAcceleratorProvider].
     *
     * Note: This version does not enable compiler caching. Use [create(Context, NpuAcceleratorProvider, Map, Boolean)]
     * to enable caching.
     *
     * @param npuAcceleratorProvider The NPU accelerator provider.
     * @param options The options to configure the environment.
     */
    @Throws(LiteRtException::class)
    @JvmOverloads
    @JvmStatic
    fun create(
      npuAcceleratorProvider: NpuAcceleratorProvider,
      options: Map<Option, String> = mapOf(),
    ): Environment {
      val mutableOptions = options.toMutableMap()
      if (npuAcceleratorProvider.isDeviceSupported() && npuAcceleratorProvider.isLibraryReady()) {
        mutableOptions[Option.DispatchLibraryDir] = npuAcceleratorProvider.getLibraryDir()
        mutableOptions[Option.CompilerPluginLibraryDir] = npuAcceleratorProvider.getLibraryDir()
      }

      return Environment(
        nativeCreate(
          mutableOptions.keys.map { it.value }.toIntArray(),
          mutableOptions.values.toTypedArray(),
        )
      )
    }

    @JvmStatic private external fun nativeCreate(keys: IntArray, values: Array<String>): Long

    @JvmStatic private external fun nativeGetAvailableAccelerators(handle: Long): IntArray

    @JvmStatic private external fun nativeDestroy(handle: Long)
  }
}
