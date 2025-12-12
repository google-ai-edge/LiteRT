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

/** Class that represents a compiled LiteRT model. */
class CompiledModel
private constructor(
  handle: Long,
  private val env: Environment,
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
    val allowSrcQuantizedFcConvOps: Boolean? = null,
    val precision: Precision? = null,
    val bufferStorageType: BufferStorageType? = null,
    val preferTextureWeights: Boolean? = null,
    val serializationDir: String? = null,
    val modelCacheKey: String? = null,
    val serializeProgramCache: Boolean? = null,
    val serializeExternalTensors: Boolean? = null,
    val externalTensorsMode: Boolean? = null,
    val externalTensorPattern: String? = null,
    val backend: Backend? = null,
    val priority: Priority? = null,
    val numStepsOfCommandBufferPreparations: Int? = null,
  ) {
    /** Precision for GPU options. */
    enum class Precision constructor(val value: Int) {
      DEFAULT(0),
      FP16(1),
      FP32(2),
    }

    /** Buffer storage type for GPU options. */
    enum class BufferStorageType constructor(val value: Int) {
      DEFAULT(0),
      BUFFER(1),
      TEXTURE_2D(2),
    }

    /** Backend for GPU options. */
    enum class Backend constructor(val value: Int) {
      AUTOMATIC(0),
      OPENCL(1),
      WEBGPU(2),
    }

    /** Priority for GPU options. */
    enum class Priority constructor(val value: Int) {
      DEFAULT(0),
      LOW(1),
      NORMAL(2),
      HIGH(3),
    }

    // Keys for passing the GPU options to the native layer.
    internal enum class Key constructor(val value: Int) {
      CONSTANT_TENSOR_SHARING(0),
      INFINITE_FLOAT_CAPPING(1),
      ALLOW_SRC_QUANTIZED_FC_CONV_OPS(2),
      PRECISION(3),
      BUFFER_STORAGE_TYPE(4),
      PREFER_TEXTURE_WEIGHTS(5),
      SERIALIZATION_DIR(6),
      MODEL_CACHE_KEY(7),
      SERIALIZE_PROGRAM_CACHE(8),
      SERIALIZE_EXTERNAL_TENSORS(9),
      EXTERNAL_TENSORS_MODE(10),
      EXTERNAL_TENSOR_PATTERN(11),
      BACKEND(12),
      PRIORITY(13),
      NUM_STEPS_OF_COMMAND_BUFFER_PREPARATIONS(14),
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
      if (allowSrcQuantizedFcConvOps != null) {
        map[Key.ALLOW_SRC_QUANTIZED_FC_CONV_OPS] = allowSrcQuantizedFcConvOps.toString()
      }
      if (precision != null) {
        map[Key.PRECISION] = precision.value.toString()
      }
      if (bufferStorageType != null) {
        map[Key.BUFFER_STORAGE_TYPE] = bufferStorageType.value.toString()
      }
      if (preferTextureWeights != null) {
        map[Key.PREFER_TEXTURE_WEIGHTS] = preferTextureWeights.toString()
      }
      if (serializationDir != null) {
        map[Key.SERIALIZATION_DIR] = serializationDir
      }
      if (modelCacheKey != null) {
        map[Key.MODEL_CACHE_KEY] = modelCacheKey
      }
      if (serializeProgramCache != null) {
        map[Key.SERIALIZE_PROGRAM_CACHE] = serializeProgramCache.toString()
      }
      if (serializeExternalTensors != null) {
        map[Key.SERIALIZE_EXTERNAL_TENSORS] = serializeExternalTensors.toString()
      }
      if (externalTensorsMode != null) {
        map[Key.EXTERNAL_TENSORS_MODE] = externalTensorsMode.toString()
      }
      if (externalTensorPattern != null) {
        map[Key.EXTERNAL_TENSOR_PATTERN] = externalTensorPattern
      }
      if (backend != null) {
        map[Key.BACKEND] = backend.value.toString()
      }
      if (priority != null) {
        map[Key.PRIORITY] = priority.value.toString()
      }
      if (numStepsOfCommandBufferPreparations != null) {
        map[Key.NUM_STEPS_OF_COMMAND_BUFFER_PREPARATIONS] =
          numStepsOfCommandBufferPreparations.toString()
      }
      return map.toMap()
    }
  }

  /** Options to specify Qualcomm options for compiling a model. */
  data class QualcommOptions
  constructor(
    val logLevel: LogLevel? = null,
    val useHtpPreference: Boolean? = null,
    val useQint16AsQuint16: Boolean? = null,
    val enableWeightSharing: Boolean? = null,
    val dumpTensorIds: List<Int>? = null,
    val useConvHmx: Boolean? = null,
    val useFoldRelu: Boolean? = null,
    val htpPerformanceMode: HtpPerformanceMode? = null,
    val profiling: Profiling? = null,
    val irJsonDir: String? = null,
    val dlcDir: String? = null,
    val vtcmSize: Int? = null,
    val numHvxThreads: Int? = null,
    val optimizationLevel: OptimizationLevel? = null,
  ) {
    /** Log level for Qualcomm options. */
    enum class LogLevel constructor(val value: Int) {
      OFF(0),
      ERROR(1),
      WARN(2),
      INFO(3),
      VERBOSE(4),
      DEBUG(5),
    }

    /** HTP performance mode for Qualcomm options. */
    enum class HtpPerformanceMode constructor(val value: Int) {
      DEFAULT(0),
      SUSTAINED_HIGH_PERFORMANCE(1),
      BURST(2),
      HIGH_PERFORMANCE(3),
      POWER_SAVER(4),
      LOW_POWER_SAVER(5),
      HIGH_POWER_SAVER(6),
      LOW_BALANCED(7),
      BALANCED(8),
      EXTREME_POWER_SAVER(9),
    }

    /** Profiling for Qualcomm options. */
    enum class Profiling constructor(val value: Int) {
      OFF(0),
      BASIC(1),
      DETAILED(2),
      LINTING(3),
      OPTRACE(4),
    }

    /** Optimization level for Qualcomm options. */
    enum class OptimizationLevel constructor(val value: Int) {
      HTP_OPTIMIZE_FOR_INFERENCE(0),
      HTP_OPTIMIZE_FOR_PREPARE(1),
      HTP_OPTIMIZE_FOR_INFERENCE_O3(2),
    }

    // Keys for passing the Qualcomm options to the native layer.
    internal enum class Key constructor(val value: Int) {
      LOG_LEVEL(0),
      USE_HTP_PREFERENCE(1),
      USE_QINT16_AS_QUINT16(2),
      ENABLE_WEIGHT_SHARING(3),
      DUMP_TENSOR_IDS(4),
      USE_CONV_HMX(5),
      USE_FOLD_RELU(6),
      HTP_PERFORMANCE_MODE(7),
      PROFILING(8),
      IR_JSON_DIR(9),
      DLC_DIR(10),
      VTCM_SIZE(11),
      NUM_HVX_THREADS(12),
      OPTIMIZATION_LEVEL(13),
    }

    // Converts the options to a map, with all values converted to strings.
    internal fun toMap(): Map<Key, String> {
      val map = mutableMapOf<Key, String>()
      if (logLevel != null) {
        map[Key.LOG_LEVEL] = logLevel.value.toString()
      }
      if (useHtpPreference != null) {
        map[Key.USE_HTP_PREFERENCE] = useHtpPreference.toString()
      }
      if (useQint16AsQuint16 != null) {
        map[Key.USE_QINT16_AS_QUINT16] = useQint16AsQuint16.toString()
      }
      if (enableWeightSharing != null) {
        map[Key.ENABLE_WEIGHT_SHARING] = enableWeightSharing.toString()
      }
      if (dumpTensorIds != null) {
        map[Key.DUMP_TENSOR_IDS] = dumpTensorIds.joinToString(",")
      }
      if (useConvHmx != null) {
        map[Key.USE_CONV_HMX] = useConvHmx.toString()
      }
      if (useFoldRelu != null) {
        map[Key.USE_FOLD_RELU] = useFoldRelu.toString()
      }
      if (htpPerformanceMode != null) {
        map[Key.HTP_PERFORMANCE_MODE] = htpPerformanceMode.value.toString()
      }
      if (profiling != null) {
        map[Key.PROFILING] = profiling.value.toString()
      }
      if (irJsonDir != null) {
        map[Key.IR_JSON_DIR] = irJsonDir
      }
      if (dlcDir != null) {
        map[Key.DLC_DIR] = dlcDir
      }
      if (vtcmSize != null) {
        map[Key.VTCM_SIZE] = vtcmSize.toString()
      }
      if (numHvxThreads != null) {
        map[Key.NUM_HVX_THREADS] = numHvxThreads.toString()
      }
      if (optimizationLevel != null) {
        map[Key.OPTIMIZATION_LEVEL] = optimizationLevel.value.toString()
      }
      return map.toMap()
    }
  }

  /** Options to specify hardware acceleration for compiling a model. */
  class Options constructor(internal val accelerators: Set<Accelerator>) {

    constructor(vararg accelerators: Accelerator) : this(setOf(*accelerators)) {}

    var cpuOptions: CpuOptions? = null
    var gpuOptions: GpuOptions? = null
    var qualcommOptions: QualcommOptions? = null

    companion object {
      @JvmStatic val CPU = Options(Accelerator.CPU)
    }
  }

  @Throws(LiteRtException::class)
  fun createInputBuffer(inputName: String, signature: String = ""): TensorBuffer {
    assertNotDestroyed()

    val tb = nativeCreateInputBuffer(handle, signature, inputName)
    return TensorBuffer(tb)
  }

  @Throws(LiteRtException::class)
  fun getInputBufferRequirements(
    inputName: String,
    signature: String = "",
  ): TensorBufferRequirements {
    assertNotDestroyed()

    return nativeGetInputBufferRequirements(handle, signature, inputName)
  }

  @Throws(LiteRtException::class)
  fun createOutputBuffer(outputName: String, signature: String = ""): TensorBuffer {
    assertNotDestroyed()

    val tb = nativeCreateOutputBuffer(handle, signature, outputName)
    return TensorBuffer(tb)
  }

  @Throws(LiteRtException::class)
  fun getOutputBufferRequirements(
    outputName: String,
    signature: String = "",
  ): TensorBufferRequirements {
    assertNotDestroyed()

    return nativeGetOutputBufferRequirements(handle, signature, outputName)
  }

  @Throws(LiteRtException::class)
  @JvmOverloads
  fun createInputBuffers(signatureIndex: Int = 0): List<TensorBuffer> {
    assertNotDestroyed()

    val handles = nativeCreateInputBuffers(handle, signatureIndex)
    return handles.map { TensorBuffer(it) }
  }

  @Throws(LiteRtException::class)
  fun createInputBuffers(signature: String): List<TensorBuffer> {
    assertNotDestroyed()

    val handles = nativeCreateInputBuffersBySignature(handle, signature)
    return handles.map { TensorBuffer(it) }
  }

  @Throws(LiteRtException::class)
  @JvmOverloads
  fun createOutputBuffers(signatureIndex: Int = 0): List<TensorBuffer> {
    assertNotDestroyed()

    val handles = nativeCreateOutputBuffers(handle, signatureIndex)
    return handles.map { TensorBuffer(it) }
  }

  @Throws(LiteRtException::class)
  fun createOutputBuffers(signature: String): List<TensorBuffer> {
    assertNotDestroyed()

    val handles = nativeCreateOutputBuffersBySignature(handle, signature)
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
      signature,
      inputs.map { it.handle }.toLongArray(),
      outputs.map { it.handle }.toLongArray(),
    )
  }

  @Throws(LiteRtException::class)
  fun run(
    inputs: Map<String, TensorBuffer>,
    outputs: Map<String, TensorBuffer>,
    signature: String = "",
  ) {
    assertNotDestroyed()

    nativeRunBySignatureWithMap(
      handle,
      signature,
      inputs.keys.toTypedArray(),
      inputs.values.map { it.handle }.toLongArray(),
      outputs.keys.toTypedArray(),
      outputs.values.map { it.handle }.toLongArray(),
    )
  }

  fun getInputTensorType(inputName: String, signature: String = ""): TensorType {
    assertNotDestroyed()

    return nativeGetInputTensorType(handle, inputName, signature)
  }

  fun getOutputTensorType(outputName: String, signature: String = ""): TensorType {
    assertNotDestroyed()

    return nativeGetOutputTensorType(handle, outputName, signature)
  }

  override fun destroy() {
    nativeDestroy(handle)
    if (envManaged) {
      env.close()
    }
  }

  companion object {
    init {
      System.loadLibrary("LiteRt")
    }

    private fun createFromAsset(
      assetManager: AssetManager,
      assetName: String,
      options: Options = Options.CPU,
      optionalEnv: Environment? = null,
      envManaged: Boolean = optionalEnv == null,
    ): CompiledModel {
      val env = optionalEnv ?: Environment.create()
      val accelerators =
        if (options.accelerators.size == 1 && options.accelerators.first() == Accelerator.NPU) {
          // If NPU is the only accelerator, CPU is added to support partially compiled models.
          // TODO(niuchl): Document this behavior in the AOT flow.
          setOf(Accelerator.NPU, Accelerator.CPU)
        } else {
          options.accelerators
        }

      val cpuOptionsMap = options.cpuOptions?.toMap() ?: mapOf()
      val gpuOptionsMap = options.gpuOptions?.toMap() ?: mapOf()
      val qualcommOptionsMap = options.qualcommOptions?.toMap() ?: mapOf()
      return CompiledModel(
        nativeCreateFromAsset(
          env.handle,
          assetManager,
          assetName,
          accelerators.map { it.value }.toIntArray(),
          cpuOptionsMap.keys.map { it.value }.toIntArray(),
          cpuOptionsMap.values.toTypedArray(),
          gpuOptionsMap.keys.map { it.value }.toIntArray(),
          gpuOptionsMap.values.toTypedArray(),
          qualcommOptionsMap.keys.map { it.value }.toIntArray(),
          qualcommOptionsMap.values.toTypedArray(),
        ),
        env,
        envManaged,
      )
    }

    private fun createFromFile(
      filePath: String,
      options: Options = Options.CPU,
      optionalEnv: Environment? = null,
      envManaged: Boolean = optionalEnv == null,
    ): CompiledModel {
      val env = optionalEnv ?: Environment.create()
      val accelerators =
        if (options.accelerators.size == 1 && options.accelerators.first() == Accelerator.NPU) {
          // If NPU is the only accelerator, CPU is added to support partially compiled models.
          // TODO(niuchl): Document this behavior in the AOT flow.
          setOf(Accelerator.NPU, Accelerator.CPU)
        } else {
          options.accelerators
        }

      val cpuOptionsMap = options.cpuOptions?.toMap() ?: mapOf()
      val gpuOptionsMap = options.gpuOptions?.toMap() ?: mapOf()
      val qualcommOptionsMap = options.qualcommOptions?.toMap() ?: mapOf()
      return CompiledModel(
        nativeCreateFromFile(
          env.handle,
          filePath,
          accelerators.map { it.value }.toIntArray(),
          cpuOptionsMap.keys.map { it.value }.toIntArray(),
          cpuOptionsMap.values.toTypedArray(),
          gpuOptionsMap.keys.map { it.value }.toIntArray(),
          gpuOptionsMap.values.toTypedArray(),
          qualcommOptionsMap.keys.map { it.value }.toIntArray(),
          qualcommOptionsMap.values.toTypedArray(),
        ),
        env,
        envManaged,
      )
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
      return createFromAsset(assetManager, assetName, options, optionalEnv)
    }

    @Throws(LiteRtException::class)
    @JvmOverloads
    @JvmStatic
    fun create(
      filePath: String,
      options: Options = Options.CPU,
      optionalEnv: Environment? = null,
    ): CompiledModel {
      return createFromFile(filePath, options, optionalEnv)
    }

    @JvmStatic
    private external fun nativeCreateFromAsset(
      envHandle: Long,
      assetManager: AssetManager,
      assetName: String,
      accelerators: IntArray,
      cpuOptionsKeys: IntArray,
      cpuOptionsValues: Array<String>,
      gpuOptionsKeys: IntArray,
      gpuOptionsValues: Array<String>,
      qualcommOptionsKeys: IntArray,
      qualcommOptionsValues: Array<String>,
    ): Long

    @JvmStatic
    private external fun nativeCreateFromFile(
      envHandle: Long,
      filePath: String,
      accelerators: IntArray,
      cpuOptionsKeys: IntArray,
      cpuOptionsValues: Array<String>,
      gpuOptionsKeys: IntArray,
      gpuOptionsValues: Array<String>,
      qualcommOptionsKeys: IntArray,
      qualcommOptionsValues: Array<String>,
    ): Long

    @JvmStatic
    private external fun nativeCreateInputBuffer(
      compiledModelHandle: Long,
      signature: String,
      inputName: String,
    ): Long

    @JvmStatic
    private external fun nativeGetInputBufferRequirements(
      compiledModelHandle: Long,
      signature: String,
      inputName: String,
    ): TensorBufferRequirements

    @JvmStatic
    private external fun nativeCreateOutputBuffer(
      compiledModelHandle: Long,
      signature: String,
      outputName: String,
    ): Long

    @JvmStatic
    private external fun nativeGetOutputBufferRequirements(
      compiledModelHandle: Long,
      signature: String,
      outputName: String,
    ): TensorBufferRequirements

    @JvmStatic
    private external fun nativeCreateInputBuffers(
      compiledModelHandle: Long,
      signatureIndex: Int,
    ): LongArray

    @JvmStatic
    private external fun nativeCreateInputBuffersBySignature(
      compiledModelHandle: Long,
      signature: String,
    ): LongArray

    @JvmStatic
    private external fun nativeCreateOutputBuffers(
      compiledModelHandle: Long,
      signatureIndex: Int,
    ): LongArray

    @JvmStatic
    private external fun nativeCreateOutputBuffersBySignature(
      compiledModelHandle: Long,
      signature: String,
    ): LongArray

    @JvmStatic
    private external fun nativeRun(
      compiledModelHandle: Long,
      signatureIndex: Int,
      inputBuffers: LongArray,
      outputBuffers: LongArray,
    )

    @JvmStatic
    private external fun nativeRunBySignature(
      compiledModelHandle: Long,
      signature: String,
      inputBuffers: LongArray,
      outputBuffers: LongArray,
    )

    @JvmStatic
    private external fun nativeRunBySignatureWithMap(
      compiledModelHandle: Long,
      signature: String,
      inputKeys: Array<String>,
      inputBuffers: LongArray,
      outputKeys: Array<String>,
      outputBuffers: LongArray,
    )

    @JvmStatic
    private external fun nativeGetInputTensorType(
      handle: Long,
      inputName: String,
      signature: String,
    ): TensorType

    @JvmStatic
    private external fun nativeGetOutputTensorType(
      handle: Long,
      outputName: String,
      signature: String,
    ): TensorType

    @JvmStatic private external fun nativeDestroy(handle: Long)
  }
}
