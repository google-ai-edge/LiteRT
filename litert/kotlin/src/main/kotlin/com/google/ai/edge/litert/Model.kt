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

import android.content.res.AssetManager

/** The type of a tensor, including its element type and layout. */
data class TensorType
@JvmOverloads
constructor(val elementType: ElementType, val layout: Layout? = null) {

  /** Data type of tensor elements. */
  // TODO(niuchl): Add support for more element types.
  enum class ElementType {
    INT,
    FLOAT,
    INT8,
    BOOLEAN,
    INT64,
  }

  /** Layout of a tensor. */
  data class Layout
  @JvmOverloads
  constructor(val dimensions: List<Int>, val strides: List<Int> = listOf()) {

    @JvmOverloads
    constructor(
      dimensions: IntArray,
      strides: IntArray = intArrayOf(),
    ) : this(dimensions.toList(), strides.toList()) {}

    val rank: Int
      get() = dimensions.size

    val hasStrides: Boolean
      get() = strides.isNotEmpty()
  }
}

/** Model represents a LiteRT model file. */
class Model private constructor(handle: Long) : JniHandle(handle) {

  protected override fun destroy() {
    nativeDestroy(handle)
  }

  fun getInputTensorType(inputName: String, signature: String? = null): TensorType {
    assertNotDestroyed()

    return nativeGetInputTensorType(handle, inputName, signature)
  }

  fun getOutputTensorType(outputName: String, signature: String? = null): TensorType {
    assertNotDestroyed()

    return nativeGetOutputTensorType(handle, outputName, signature)
  }

  companion object {
    init {
      System.loadLibrary("litert_jni")
    }

    @Throws(LiteRtException::class)
    @JvmStatic
    fun load(assetManager: AssetManager, assetName: String): Model {
      return Model(nativeLoadAsset(assetManager, assetName))
    }

    @Throws(LiteRtException::class)
    @JvmStatic
    fun load(filePath: String): Model {
      return Model(nativeLoadFile(filePath))
    }

    @JvmStatic
    private external fun nativeLoadAsset(assetManager: AssetManager, assetName: String): Long

    @JvmStatic private external fun nativeLoadFile(filePath: String): Long

    @JvmStatic private external fun nativeDestroy(handle: Long)

    @JvmStatic
    private external fun nativeGetInputTensorType(
      handle: Long,
      inputName: String,
      signature: String?,
    ): TensorType

    @JvmStatic
    private external fun nativeGetOutputTensorType(
      handle: Long,
      outputName: String,
      signature: String?,
    ): TensorType
  }
}

/** Class that represents a compiled LiteRT model. */
class CompiledModel
private constructor(
  handle: Long,
  private val model: Model,
  private val env: Environment,
  private val modelManaged: Boolean = false,
  private val envManaged: Boolean = false,
) : JniHandle(handle) {

  /** Options to specify CPU acceleration for compiling a model. */
  data class CpuOptions
  constructor(
    val numThreads: Int? = null,
    val xnnPackFlags: Int? = null,
    val xnnPackWeightCachePath: String? = null,
  ) {
    // Keys for passing the CPU options to the native layer.
    internal enum class Key constructor(val value: Int) {
      NUM_THREADS(0),
      XNNPACK_FLAGS(1),
      XNNPACK_WEIGHT_CACHE_PATH(2),
    }

    // Converts the options to a map, with all values converted to strings.
    internal fun toMap(): Map<Key, String> {
      val map = mutableMapOf<Key, String>()
      if (numThreads != null) {
        map[Key.NUM_THREADS] = numThreads.toString()
      }
      if (xnnPackFlags != null) {
        map[Key.XNNPACK_FLAGS] = xnnPackFlags.toString()
      }
      if (xnnPackWeightCachePath != null) {
        map[Key.XNNPACK_WEIGHT_CACHE_PATH] = xnnPackWeightCachePath
      }
      return map.toMap()
    }
  }

  /** Options to specify GPU acceleration for compiling a model. */
  data class GpuOptions
  constructor(
    val constantTensorSharing: Boolean? = null,
    val infiniteFloatCapping: Boolean? = null,
    val benchmarkMode: Boolean? = null,
    val allowSrcQuantizedFcConvOps: Boolean? = null,
    val precision: Precision? = null,
    val bufferStorageType: BufferStorageType? = null,
  ) {
    /** Precision for GPU options. */
    enum class Precision constructor(val value: Int) {
      DEFAULT(0),
      FP16(1),
      FP32(2),
    }

    /** Buffer storage type for GPU options. */
    enum class BufferStorageType constructor(val value: Int) {
      DEFAUTL(0),
      BUFFER(1),
      TEXTURE_2D(2),
    }

    // Keys for passing the GPU options to the native layer.
    internal enum class Key constructor(val value: Int) {
      CONSTANT_TENSOR_SHARING(0),
      INFINITE_FLOAT_CAPPING(1),
      BENCHMARK_MODE(2),
      ALLOW_SRC_QUANTIZED_FC_CONV_OPS(3),
      PRECISION(4),
      BUFFER_STORAGE_TYPE(5),
    }

    // Converts the options to a map, with all values converted to strings.
    internal fun toMap(): Map<Key, String> {
      val map = mutableMapOf<Key, String>()
      if (constantTensorSharing != null) {
        map[Key.CONSTANT_TENSOR_SHARING] = constantTensorSharing.toString()
      }
      if (infiniteFloatCapping != null) {
        map[Key.INFINITE_FLOAT_CAPPING] = infiniteFloatCapping.toString()
      }
      if (benchmarkMode != null) {
        map[Key.BENCHMARK_MODE] = benchmarkMode.toString()
      }
      if (allowSrcQuantizedFcConvOps != null) {
        map[Key.ALLOW_SRC_QUANTIZED_FC_CONV_OPS] = allowSrcQuantizedFcConvOps.toString()
      }
      if (precision != null) {
        map[Key.PRECISION] = precision.value.toString()
      }
      if (bufferStorageType != null) {
        map[Key.BUFFER_STORAGE_TYPE] = bufferStorageType.value.toString()
      }
      return map.toMap()
    }
  }

  /** Options to specify hardware acceleration for compiling a model. */
  class Options constructor(internal val accelerators: Set<Accelerator>) {

    constructor(vararg accelerators: Accelerator) : this(setOf(*accelerators)) {}

    var cpuOptions: CpuOptions? = null
    var gpuOptions: GpuOptions? = null

    companion object {
      @JvmStatic val CPU = Options(Accelerator.CPU)
    }
  }

  @Throws(LiteRtException::class)
  fun createInputBuffer(inputName: String, signature: String? = null): TensorBuffer {
    assertNotDestroyed()

    val tb = nativeCreateInputBuffer(handle, model.handle, signature, inputName)
    return TensorBuffer(tb)
  }

  @Throws(LiteRtException::class)
  fun getInputBufferRequirements(
    inputName: String,
    signature: String? = null,
  ): TensorBufferRequirements {
    assertNotDestroyed()

    return nativeGetInputBufferRequirements(handle, model.handle, signature, inputName)
  }

  @Throws(LiteRtException::class)
  fun createOutputBuffer(outputName: String, signature: String? = null): TensorBuffer {
    assertNotDestroyed()

    val tb = nativeCreateOutputBuffer(handle, model.handle, signature, outputName)
    return TensorBuffer(tb)
  }

  @Throws(LiteRtException::class)
  fun getOutputBufferRequirements(
    outputName: String,
    signature: String? = null,
  ): TensorBufferRequirements {
    assertNotDestroyed()

    return nativeGetOutputBufferRequirements(handle, model.handle, signature, outputName)
  }

  @Throws(LiteRtException::class)
  @JvmOverloads
  fun createInputBuffers(signatureIndex: Int = 0): List<TensorBuffer> {
    assertNotDestroyed()

    val handles = nativeCreateInputBuffers(handle, model.handle, signatureIndex)
    return handles.map { TensorBuffer(it) }
  }

  @Throws(LiteRtException::class)
  fun createInputBuffers(signature: String): List<TensorBuffer> {
    assertNotDestroyed()

    val handles = nativeCreateInputBuffersBySignature(handle, model.handle, signature)
    return handles.map { TensorBuffer(it) }
  }

  @Throws(LiteRtException::class)
  @JvmOverloads
  fun createOutputBuffers(signatureIndex: Int = 0): List<TensorBuffer> {
    assertNotDestroyed()

    val handles = nativeCreateOutputBuffers(handle, model.handle, signatureIndex)
    return handles.map { TensorBuffer(it) }
  }

  @Throws(LiteRtException::class)
  fun createOutputBuffers(signature: String): List<TensorBuffer> {
    assertNotDestroyed()

    val handles = nativeCreateOutputBuffersBySignature(handle, model.handle, signature)
    return handles.map { TensorBuffer(it) }
  }

  @Throws(LiteRtException::class)
  @JvmOverloads
  fun run(inputs: List<TensorBuffer>, signatureIndex: Int = 0): List<TensorBuffer> {
    assertNotDestroyed()

    val outputs = createOutputBuffers(signatureIndex)
    run(inputs, outputs, signatureIndex)
    return outputs
  }

  @Throws(LiteRtException::class)
  fun run(inputs: List<TensorBuffer>, signature: String): List<TensorBuffer> {
    assertNotDestroyed()

    val outputs = createOutputBuffers(signature)
    run(inputs, outputs, signature)
    return outputs
  }

  @Throws(LiteRtException::class)
  @JvmOverloads
  fun run(inputs: List<TensorBuffer>, outputs: List<TensorBuffer>, signatureIndex: Int = 0) {
    assertNotDestroyed()

    nativeRun(
      handle,
      model.handle,
      signatureIndex,
      inputs.map { it.handle }.toLongArray(),
      outputs.map { it.handle }.toLongArray(),
    )
  }

  @Throws(LiteRtException::class)
  fun run(inputs: List<TensorBuffer>, outputs: List<TensorBuffer>, signature: String) {
    assertNotDestroyed()

    nativeRunBySignature(
      handle,
      model.handle,
      signature,
      inputs.map { it.handle }.toLongArray(),
      outputs.map { it.handle }.toLongArray(),
    )
  }

  @Throws(LiteRtException::class)
  fun run(
    inputs: Map<String, TensorBuffer>,
    outputs: Map<String, TensorBuffer>,
    signature: String? = null,
  ) {
    assertNotDestroyed()

    nativeRunBySignatureWithMap(
      handle,
      model.handle,
      signature,
      inputs.keys.toTypedArray(),
      inputs.values.map { it.handle }.toLongArray(),
      outputs.keys.toTypedArray(),
      outputs.values.map { it.handle }.toLongArray(),
    )
  }

  protected override fun destroy() {
    nativeDestroy(handle)
    if (modelManaged) {
      model.close()
    }
    if (envManaged) {
      env.close()
    }
  }

  companion object {
    init {
      System.loadLibrary("litert_jni")
    }

    private fun create(
      model: Model,
      options: Options = Options.CPU,
      optionalEnv: Environment? = null,
      modelManaged: Boolean,
      envManaged: Boolean = optionalEnv == null,
    ): CompiledModel {
      val env = optionalEnv ?: Environment.create()
      val accelerators =
        if (options.accelerators.size == 1 && options.accelerators.first() == Accelerator.NPU)
        // If NPU is the only accelerator, CPU is added to support partially compiled models.
        // TODO(niuchl): Document this behavior in the AOT flow.
        setOf(Accelerator.NPU, Accelerator.CPU)
        else options.accelerators

      val cpuOptionsMap = options.cpuOptions?.toMap() ?: mapOf()
      val gpuOptionsMap = options.gpuOptions?.toMap() ?: mapOf()
      return CompiledModel(
        nativeCreate(
          env.handle,
          model.handle,
          accelerators.map { it.value }.toIntArray(),
          cpuOptionsMap.keys.map { it.value }.toIntArray(),
          cpuOptionsMap.values.toTypedArray(),
          gpuOptionsMap.keys.map { it.value }.toIntArray(),
          gpuOptionsMap.values.toTypedArray(),
        ),
        model,
        env,
        modelManaged,
        envManaged,
      )
    }

    @Throws(LiteRtException::class)
    @JvmOverloads
    @JvmStatic
    fun create(
      model: Model,
      options: Options = Options.CPU,
      optionalEnv: Environment? = null,
    ): CompiledModel {
      return create(model, options, optionalEnv, modelManaged = false)
    }

    @Throws(LiteRtException::class)
    @JvmOverloads
    @JvmStatic
    fun create(
      assetManager: AssetManager,
      assetName: String,
      options: Options = Options.CPU,
      optionalEnv: Environment? = null,
    ): CompiledModel {
      return create(Model.load(assetManager, assetName), options, optionalEnv, modelManaged = true)
    }

    @Throws(LiteRtException::class)
    @JvmOverloads
    @JvmStatic
    fun create(
      filePath: String,
      options: Options = Options.CPU,
      optionalEnv: Environment? = null,
    ): CompiledModel {
      return create(Model.load(filePath), options, optionalEnv, modelManaged = true)
    }

    @JvmStatic
    private external fun nativeCreate(
      envHandle: Long,
      modelHandle: Long,
      accelerators: IntArray,
      cpuOptionsKeys: IntArray,
      cpuOptionsValues: Array<String>,
      gpuOptionsKeys: IntArray,
      gpuOptionsValues: Array<String>,
    ): Long

    @JvmStatic
    private external fun nativeCreateInputBuffer(
      compiledModelHandle: Long,
      modelHandle: Long,
      signature: String?,
      inputName: String,
    ): Long

    @JvmStatic
    private external fun nativeGetInputBufferRequirements(
      compiledModelHandle: Long,
      modelHandle: Long,
      signature: String?,
      inputName: String,
    ): TensorBufferRequirements

    @JvmStatic
    private external fun nativeCreateOutputBuffer(
      compiledModelHandle: Long,
      modelHandle: Long,
      signature: String?,
      outputName: String,
    ): Long

    @JvmStatic
    private external fun nativeGetOutputBufferRequirements(
      compiledModelHandle: Long,
      modelHandle: Long,
      signature: String?,
      outputName: String,
    ): TensorBufferRequirements

    @JvmStatic
    private external fun nativeCreateInputBuffers(
      compiledModelHandle: Long,
      modelHandle: Long,
      signatureIndex: Int,
    ): LongArray

    @JvmStatic
    private external fun nativeCreateInputBuffersBySignature(
      compiledModelHandle: Long,
      modelHandle: Long,
      signature: String,
    ): LongArray

    @JvmStatic
    private external fun nativeCreateOutputBuffers(
      compiledModelHandle: Long,
      modelHandle: Long,
      signatureIndex: Int,
    ): LongArray

    @JvmStatic
    private external fun nativeCreateOutputBuffersBySignature(
      compiledModelHandle: Long,
      modelHandle: Long,
      signature: String,
    ): LongArray

    @JvmStatic
    private external fun nativeRun(
      compiledModelHandle: Long,
      modelHandle: Long,
      signatureIndex: Int,
      inputBuffers: LongArray,
      outputBuffers: LongArray,
    )

    @JvmStatic
    private external fun nativeRunBySignature(
      compiledModelHandle: Long,
      modelHandle: Long,
      signature: String,
      inputBuffers: LongArray,
      outputBuffers: LongArray,
    )

    @JvmStatic
    private external fun nativeRunBySignatureWithMap(
      compiledModelHandle: Long,
      modelHandle: Long,
      signature: String?,
      inputKeys: Array<String>,
      inputBuffers: LongArray,
      outputKeys: Array<String>,
      outputBuffers: LongArray,
    )

    @JvmStatic private external fun nativeDestroy(handle: Long)
  }
}
