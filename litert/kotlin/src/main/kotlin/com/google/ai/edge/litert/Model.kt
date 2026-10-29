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
import androidx.annotation.VisibleForTesting

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
    INT16,
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

  internal class NativeOptionsBuilder(accelerators: IntArray) :
    JniHandle(nativeCreateOptions(accelerators)) {

    fun writeBool(key: String, value: Boolean) {
      assertNotDestroyed()
      nativeWriteBoolOption(handle, key, value)
    }

    fun writeInt(key: String, value: Long) {
      assertNotDestroyed()
      nativeWriteIntOption(handle, key, value)
    }

    fun writeString(key: String, value: String) {
      assertNotDestroyed()
      nativeWriteStringOption(handle, key, value)
    }

    fun writeStringArray(key: String, values: Array<String>) {
      assertNotDestroyed()
      nativeWriteStringArrayOption(handle, key, values)
    }

    fun flush(identifier: String) {
      assertNotDestroyed()
      nativeFlushOpaqueOptions(handle, identifier)
    }

    override fun destroy() {
      nativeDestroyOptions(handle)
    }
  }

  /** Options to specify CPU acceleration for compiling a model. */
  data class CpuOptions
  constructor(
    val numThreads: Int? = null,
    val xnnPackFlags: Int? = null,
    val xnnPackWeightCachePath: String? = null,
  ) {
    internal fun writeTo(builder: NativeOptionsBuilder) {
      numThreads?.let { builder.writeInt("num_threads", it.toLong()) }
      xnnPackFlags?.let { builder.writeInt("flags", it.toLong() and 0xFFFFFFFFL) }
      xnnPackWeightCachePath?.let { builder.writeString("weight_cache_file_path", it) }
      builder.flush("cpu_options")
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
      FP16_WITH_FP32_ACCUM(3),
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
      @Deprecated("Use AUTOMATIC instead.") WEBGPU(2),
      OPENGL(3),
    }

    /** Priority for GPU options. */
    enum class Priority constructor(val value: Int) {
      DEFAULT(0),
      LOW(1),
      NORMAL(2),
      HIGH(3),
    }

    internal fun writeTo(builder: NativeOptionsBuilder) {
      constantTensorSharing?.let { builder.writeBool("enable_constant_tensors_sharing", it) }
      infiniteFloatCapping?.let { builder.writeBool("enable_infinite_float_capping", it) }
      allowSrcQuantizedFcConvOps?.let { builder.writeBool("allow_src_quantized_fc_conv_ops", it) }
      precision?.let { builder.writeInt("precision", it.value.toLong()) }
      bufferStorageType?.let { builder.writeInt("buffer_storage_type", it.value.toLong()) }
      preferTextureWeights?.let { builder.writeBool("prefer_texture_weights", it) }
      serializationDir?.let { builder.writeString("serialization_dir", it) }
      modelCacheKey?.let { builder.writeString("model_cache_key", it) }
      serializeProgramCache?.let { builder.writeBool("serialize_program_cache", it) }
      serializeExternalTensors?.let { builder.writeBool("serialize_external_tensors", it) }
      externalTensorsMode?.let { builder.writeBool("external_tensors_mode", it) }
      externalTensorPattern?.let {
        builder.writeStringArray("external_tensor_patterns", arrayOf(it))
      }
      backend?.let {
        @Suppress("DEPRECATION")
        val nativeBackend = if (it == Backend.WEBGPU) Backend.AUTOMATIC else it
        builder.writeInt("backend", nativeBackend.value.toLong())
      }
      priority?.let { builder.writeInt("priority", it.value.toLong()) }
      numStepsOfCommandBufferPreparations?.let {
        builder.writeInt("num_steps_of_command_buffer_preparations", it.toLong())
      }
      builder.flush("gpu_options")
    }
  }

  /** Options to specify Qualcomm options for compiling a model. */
  data class QualcommOptions
  constructor(
    val logLevel: LogLevel? = null,
    @Deprecated("This option is deprecated and will be no-op.")
    val useHtpPreference: Boolean? = null,
    @Deprecated("This option is deprecated and will be no-op.")
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
    val htpDlbc: Boolean? = null,
    val htpDlbcWeights: Boolean? = null,
    val graphPriority: GraphPriority? = null,
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

    /** Graph priority for Qualcomm options. */
    enum class GraphPriority constructor(val value: Int) {
      DEFAULT(0),
      LOW(1),
      NORMAL(2),
      NORMAL_HIGH(3),
      HIGH(4),
    }

    internal fun writeTo(builder: NativeOptionsBuilder) {
      logLevel?.let { builder.writeInt("log_level", it.value.toLong()) }
      enableWeightSharing?.let { builder.writeBool("enable_weight_sharing", it) }
      dumpTensorIds?.let { ids ->
        builder.writeStringArray("dump_tensor_ids", ids.map { it.toString() }.toTypedArray())
      }
      useConvHmx?.let { builder.writeBool("use_conv_hmx", it) }
      useFoldRelu?.let { builder.writeBool("use_fold_relu", it) }
      htpDlbc?.let { builder.writeBool("htp_dlbc", it) }
      htpDlbcWeights?.let { builder.writeBool("htp_dlbc_weights", it) }
      htpPerformanceMode?.let { builder.writeInt("htp_performance_mode", it.value.toLong()) }
      profiling?.let { builder.writeInt("profiling", it.value.toLong()) }
      irJsonDir?.let { builder.writeString("ir_json_dir", it) }
      dlcDir?.let { builder.writeString("dlc_dir", it) }
      vtcmSize?.let { builder.writeInt("vtcm_size", it.toLong() and 0xFFFFFFFFL) }
      numHvxThreads?.let { builder.writeInt("num_hvx_threads", it.toLong() and 0xFFFFFFFFL) }
      optimizationLevel?.let { builder.writeInt("optimization_level", it.value.toLong()) }
      graphPriority?.let { builder.writeInt("graph_priority", it.value.toLong()) }
      builder.flush("qualcomm")
    }
  }

  /**
   * Options to configure the LiteRT compiler plugin when compiling a model.
   *
   * Input shape overrides refine dynamic input dimensions before on-device (JIT) compilation. For
   * each signature, use exactly one addressing mode (positional, tensor name, or signature input
   * name); for the named modes, the last override for a given name wins. Each dimension must be
   * positive, or `-1` to keep it dynamic.
   *
   * Note: Overrides are applied only during on-device (JIT) compilation by a compiler plugin, which
   * requires a compiler plugin library directory in the [Environment] and a model that is not
   * pre-compiled. Otherwise (e.g. CPU-only execution), they are ignored. Dimensions are validated
   * when an override is added, names and signature keys when [CompiledModel.create] is called, and
   * addressing modes only during compilation.
   *
   * The overrides are copied when [CompiledModel.create] is called, so modifying this object
   * afterwards does not affect already created models.
   */
  class CompilerOptions {
    private enum class InputShapeMode {
      POSITIONAL,
      TENSOR,
      SIGNATURE,
    }

    private class InputShapeOverride(
      val mode: InputShapeMode,
      val signature: String,
      val name: String,
      val dimensions: List<Int>,
    )

    private val inputShapeOverrides = mutableListOf<InputShapeOverride>()

    /**
     * Appends a positional input shape override for [signature]. Call once per input, in subgraph
     * input order.
     *
     * @param dimensions Target dimensions for the input (use an empty list for a 0-D scalar, or
     *   `-1` for dimensions that should remain dynamic).
     * @param signature Target signature key, or `""` for the default signature.
     * @return This [CompilerOptions] instance for chaining.
     * @throws IllegalArgumentException if any dimension is neither positive nor `-1`.
     */
    @JvmOverloads
    fun addPositionalInputShape(dimensions: List<Int>, signature: String = ""): CompilerOptions {
      inputShapeOverrides.add(
        InputShapeOverride(InputShapeMode.POSITIONAL, signature, "", validated(dimensions))
      )
      return this
    }

    /**
     * Appends an input shape override addressed by internal tensor name in [signature]'s subgraph.
     *
     * @param tensorName Internal graph tensor name of the subgraph input.
     * @param dimensions Target dimensions for the input (use an empty list for a 0-D scalar, or
     *   `-1` for dimensions that should remain dynamic).
     * @param signature Target signature key, or `""` for the default signature.
     * @return This [CompilerOptions] instance for chaining.
     * @throws IllegalArgumentException if any dimension is neither positive nor `-1`.
     */
    @JvmOverloads
    fun addTensorInputShape(
      tensorName: String,
      dimensions: List<Int>,
      signature: String = "",
    ): CompilerOptions {
      inputShapeOverrides.add(
        InputShapeOverride(InputShapeMode.TENSOR, signature, tensorName, validated(dimensions))
      )
      return this
    }

    /**
     * Appends an input shape override addressed by signature input name in [signature].
     *
     * @param inputName Signature input argument name.
     * @param dimensions Target dimensions for the input (use an empty list for a 0-D scalar, or
     *   `-1` for dimensions that should remain dynamic).
     * @param signature Target signature key, or `""` for the default signature.
     * @return This [CompilerOptions] instance for chaining.
     * @throws IllegalArgumentException if any dimension is neither positive nor `-1`.
     */
    @JvmOverloads
    fun addSignatureInputShape(
      inputName: String,
      dimensions: List<Int>,
      signature: String = "",
    ): CompilerOptions {
      inputShapeOverrides.add(
        InputShapeOverride(InputShapeMode.SIGNATURE, signature, inputName, validated(dimensions))
      )
      return this
    }

    // Returns a copy of the dimensions after checking that each one is positive or -1 (dynamic).
    private fun validated(dimensions: List<Int>): List<Int> {
      require(dimensions.all { it > 0 || it == -1 }) {
        "Dimensions must be positive or -1 (dynamic), got $dimensions"
      }
      return dimensions.toList()
    }

    // Matches the characters rejected for signature keys and tensor/input names by the native
    // LrtAddCompilerOptions*InputShape setters (HasInvalidTomlChars in litert_compiler_options.cc).
    private fun hasInvalidOptionChars(s: String): Boolean = s.any {
      it == '"' || it == '\'' || it == '\\' || it == '\n' || it == '\r'
    }

    private fun invalidOverride(
      index: Int,
      override: InputShapeOverride,
      reason: String,
    ): LiteRtException =
      LiteRtException(
        Status.ErrorInvalidArgument,
        "Invalid input shape override #$index (signature='${override.signature}', " +
          "name='${override.name}'): $reason.",
      )

    internal fun writeTo(builder: NativeOptionsBuilder) {
      if (inputShapeOverrides.isEmpty()) {
        return
      }
      val positional = mutableListOf<String>()
      val tensor = mutableListOf<String>()
      val signature = mutableListOf<String>()
      for ((i, override) in inputShapeOverrides.withIndex()) {
        if (hasInvalidOptionChars(override.signature) || '@' in override.signature) {
          throw invalidOverride(i, override, "invalid signature key")
        }
        val formattedDims = override.dimensions.joinToString(":")
        when (override.mode) {
          InputShapeMode.POSITIONAL -> {
            positional.add("${override.signature}@$formattedDims")
          }
          InputShapeMode.TENSOR -> {
            if (override.name.isEmpty() || hasInvalidOptionChars(override.name)) {
              throw invalidOverride(i, override, "invalid tensor name")
            }
            tensor.add("${override.signature}@${override.name}@$formattedDims")
          }
          InputShapeMode.SIGNATURE -> {
            if (override.name.isEmpty() || hasInvalidOptionChars(override.name)) {
              throw invalidOverride(i, override, "invalid input name")
            }
            signature.add("${override.signature}@${override.name}@$formattedDims")
          }
        }
      }
      if (positional.isNotEmpty()) {
        builder.writeStringArray("positional_input_shapes", positional.toTypedArray())
      }
      if (tensor.isNotEmpty()) {
        builder.writeStringArray("tensor_input_shapes", tensor.toTypedArray())
      }
      if (signature.isNotEmpty()) {
        builder.writeStringArray("signature_input_shapes", signature.toTypedArray())
      }
      builder.flush("compiler_options_string")
    }
  }

  /** Options to specify hardware acceleration for compiling a model. */
  class Options constructor(private val accelerators: Set<Accelerator>) {

    constructor(vararg accelerators: Accelerator) : this(setOf(*accelerators)) {}

    private var autoFallbackToCpu: Boolean = true

    var cpuOptions: CpuOptions? = null
    var gpuOptions: GpuOptions? = null
    var qualcommOptions: QualcommOptions? = null
    var compilerOptions: CompilerOptions? = null

    /**
     * Disables automatically fall back to CPU when the accelerator is not available. The default is
     * true.
     *
     * @return The options.
     * @suppress
     */
    @VisibleForTesting(VisibleForTesting.NONE)
    fun disableAutoFallbackToCpu(): Options {
      this.autoFallbackToCpu = false
      return this
    }

    /**
     * Returns the set of accelerators for the options.
     *
     * @return The set of accelerators.
     * @suppress
     */
    @VisibleForTesting(VisibleForTesting.PACKAGE_PRIVATE)
    fun getAccelerators(): Set<Accelerator> {
      // If NPU is the only accelerator, CPU is added to support partially compiled models.
      // TODO(niuchl): Document this behavior in the AOT flow.
      if (autoFallbackToCpu && accelerators.size == 1 && accelerators.first() == Accelerator.NPU) {
        return setOf(Accelerator.NPU, Accelerator.CPU)
      }
      return accelerators
    }

    companion object {
      @JvmStatic val CPU = Options(Accelerator.CPU)
    }
  }

  /**
   * The duration of the most recent native inference execution, in nanoseconds, or a negative value
   * (`-1L`) if this model has not been run yet or if the most recent run failed.
   *
   * The measured interval covers the native execution of the compiled model graph by the underlying
   * runtime (and hardware accelerator synchronization), matching the timing scope of TFLite's
   * `Interpreter.getLastNativeInferenceDurationNanoseconds()` API. It excludes buffer registration,
   * host memory locking/mapping, tensor allocation, and data marshalling between Kotlin and native
   * memory.
   */
  val lastNativeInferenceDurationNanoseconds: Long
    get() {
      assertNotDestroyed()
      return nativeGetLastInferenceDurationNanoseconds(handle)
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
  fun resizeInputTensor(inputName: String, dimensions: IntArray, signature: String = "") {
    assertNotDestroyed()

    nativeResizeInputTensor(handle, signature, inputName, dimensions)
  }

  @Throws(LiteRtException::class)
  @JvmOverloads
  fun resizeInputTensorNonStrict(inputName: String, dimensions: IntArray, signature: String = "") {
    assertNotDestroyed()

    nativeResizeInputTensorNonStrict(handle, signature, inputName, dimensions)
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
      System.loadLibrary("litert_jni")
    }

    @VisibleForTesting(otherwise = VisibleForTesting.PRIVATE)
    @JvmName("createNativeOptions")
    internal fun createNativeOptions(options: Options): NativeOptionsBuilder {
      val builder = NativeOptionsBuilder(options.getAccelerators().map { it.value }.toIntArray())
      try {
        options.cpuOptions?.writeTo(builder)
        options.gpuOptions?.writeTo(builder)
        options.qualcommOptions?.writeTo(builder)
        options.compilerOptions?.writeTo(builder)
        return builder
      } catch (e: Throwable) {
        builder.close()
        throw e
      }
    }

    private fun createFromAsset(
      assetManager: AssetManager,
      assetName: String,
      options: Options = Options.CPU,
      optionalEnv: Environment? = null,
      envManaged: Boolean = optionalEnv == null,
    ): CompiledModel {
      createNativeOptions(options).use { nativeOptions ->
        val env = optionalEnv ?: Environment.create()
        return CompiledModel(
          nativeCreateFromAsset(env.handle, assetManager, assetName, nativeOptions.handle),
          env,
          envManaged,
        )
      }
    }

    private fun createFromFile(
      filePath: String,
      options: Options = Options.CPU,
      optionalEnv: Environment? = null,
      envManaged: Boolean = optionalEnv == null,
    ): CompiledModel {
      createNativeOptions(options).use { nativeOptions ->
        val env = optionalEnv ?: Environment.create()
        return CompiledModel(
          nativeCreateFromFile(env.handle, filePath, nativeOptions.handle),
          env,
          envManaged,
        )
      }
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
      optionsHandle: Long,
    ): Long

    @JvmStatic
    private external fun nativeCreateFromFile(
      envHandle: Long,
      filePath: String,
      optionsHandle: Long,
    ): Long

    @JvmStatic private external fun nativeCreateOptions(accelerators: IntArray): Long

    @JvmStatic private external fun nativeDestroyOptions(optionsHandle: Long)

    @JvmStatic
    private external fun nativeWriteBoolOption(optionsHandle: Long, key: String, value: Boolean)

    @JvmStatic
    private external fun nativeWriteIntOption(optionsHandle: Long, key: String, value: Long)

    @JvmStatic
    private external fun nativeWriteStringOption(optionsHandle: Long, key: String, value: String)

    @JvmStatic
    private external fun nativeWriteStringArrayOption(
      optionsHandle: Long,
      key: String,
      values: Array<String>,
    )

    @JvmStatic
    private external fun nativeFlushOpaqueOptions(optionsHandle: Long, identifier: String)

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
    private external fun nativeResizeInputTensor(
      compiledModelHandle: Long,
      signature: String,
      inputName: String,
      dimensions: IntArray,
    )

    @JvmStatic
    private external fun nativeResizeInputTensorNonStrict(
      compiledModelHandle: Long,
      signature: String,
      inputName: String,
      dimensions: IntArray,
    )

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

    @JvmStatic
    private external fun nativeGetLastInferenceDurationNanoseconds(compiledModelHandle: Long): Long

    @JvmStatic private external fun nativeDestroy(handle: Long)
  }
}
