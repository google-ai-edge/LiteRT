package com.google.ai.edge.litert

/** Environment to hold configuration options for LiteRT runtime. */
class Environment private constructor(handle: Long) : JniHandle(handle) {

  /** Options configurable in LiteRT environment. */
  enum class Option private constructor(val value: Int) {
    CompilerPluginLibraryDir(0),
    DispatchLibraryDir(1),
  }

  override protected fun destroy() {
    nativeDestroy(handle)
  }

  /** Returns the set of accelerators available in the environment. */
  fun getAvailableAccelerators(): Set<Accelerator> {
    assertNotDestroyed()

    val accelerators = nativeGetAvailableAccelerators(handle)
    return accelerators
      .map { Accelerator.of(it) }
      .toMutableSet()
      .apply { add(Accelerator.CPU) } // CPU is always available.
      .toSet()
  }

  companion object {
    init {
      System.loadLibrary("litert_jni")
    }

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
     * @param npuAcceleratorProvider The NPU accelerator provider.
     * @param options The options to configure the environment.
     */
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
