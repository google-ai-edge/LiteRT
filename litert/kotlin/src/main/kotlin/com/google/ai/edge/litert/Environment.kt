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
    /**
     * This for internal use only. Reserved for use by LiteRT in Play services.
     *
     * @suppress
     */
    SystemContextHandle(28),
    /**
     * This for internal use only.
     *
     * @suppress
     */
    SystemGpuAcceleratorHandle(30),
  }

  // LINT.ThenChange(../../../../../../../../../c/litert_environment_options.h)

  private class NativeOptionsBuilder : JniHandle(nativeCreateOptions()) {
    fun addStringOption(tag: Int, value: String) {
      assertNotDestroyed()
      nativeAddStringOption(handle, tag, value)
    }

    fun addPointerOption(tag: Int, pointer: Long) {
      assertNotDestroyed()
      nativeAddPointerOption(handle, tag, pointer)
    }

    override fun destroy() {
      nativeDestroyOptions(handle)
    }
  }

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
      return createWithOptions(
        options = options,
        compilerCacheDir = if (enableCompilerCache) context.cacheDir?.absolutePath else null,
      )
    }

    /**
     * Creates an environment.
     *
     * Note: This version does not enable compiler caching. Use [create(Context, Map, Boolean)] to
     * enable caching.
     *
     * @param options The options to configure the environment.
     */
    @Throws(LiteRtException::class)
    @JvmOverloads
    @JvmStatic
    fun create(options: Map<Option, String> = mapOf()): Environment {
      return createWithOptions(options = options)
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
      return createWithOptions(
        options = options,
        npuAcceleratorProvider = npuAcceleratorProvider,
        compilerCacheDir = if (enableCompilerCache) context.cacheDir?.absolutePath else null,
      )
    }

    /**
     * Creates an environment with a [NpuAcceleratorProvider].
     *
     * Note: This version does not enable compiler caching. Use
     * [create(Context, NpuAcceleratorProvider, Map, Boolean)] to enable caching.
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
      return createWithOptions(options = options, npuAcceleratorProvider = npuAcceleratorProvider)
    }

    private fun isPointerOption(option: Option): Boolean =
      when (option) {
        Option.SystemRuntimeHandle,
        Option.SystemContextHandle,
        Option.SystemGpuAcceleratorHandle -> true
        Option.CompilerPluginLibraryDir,
        Option.DispatchLibraryDir -> false
      }

    private fun createWithOptions(
      options: Map<Option, String>,
      npuAcceleratorProvider: NpuAcceleratorProvider? = null,
      compilerCacheDir: String? = null,
    ): Environment {
      val effectiveOptions =
        if (
          npuAcceleratorProvider != null &&
            npuAcceleratorProvider.isDeviceSupported() &&
            npuAcceleratorProvider.isLibraryReady()
        ) {
          val libraryDir = npuAcceleratorProvider.getLibraryDir()
          options.toMutableMap().apply {
            this[Option.DispatchLibraryDir] = libraryDir
            this[Option.CompilerPluginLibraryDir] = libraryDir
          }
        } else {
          options
        }

      NativeOptionsBuilder().use { builder ->
        for ((option, value) in effectiveOptions) {
          if (isPointerOption(option)) {
            val pointer =
              value.trim().toLongOrNull()
                ?: throw LiteRtException(
                  Status.ErrorInvalidArgument,
                  "Invalid pointer value for ${option.name}: $value",
                )
            builder.addPointerOption(option.value, pointer)
          } else {
            builder.addStringOption(option.value, value)
          }
        }
        if (compilerCacheDir != null) {
          builder.addStringOption(ENV_OPTION_TAG_COMPILER_CACHE_DIR, compilerCacheDir)
        }
        return Environment(nativeCreate(builder.handle))
      }
    }

    @JvmStatic private external fun nativeCreateOptions(): Long

    @JvmStatic private external fun nativeDestroyOptions(handle: Long)

    @JvmStatic private external fun nativeAddStringOption(handle: Long, tag: Int, value: String)

    @JvmStatic private external fun nativeAddPointerOption(handle: Long, tag: Int, pointer: Long)

    @JvmStatic private external fun nativeCreate(optionsHandle: Long): Long

    @JvmStatic private external fun nativeGetAvailableAccelerators(handle: Long): IntArray

    @JvmStatic private external fun nativeDestroy(handle: Long)
  }
}
