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

/** Model represents a LiteRT model file. */
class Model private constructor(handle: Long) : JniHandle(handle) {

  protected override fun destroy() {
    nativeDestroy(handle)
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

  /** Options to specify hardware acceleration for compiling a model. */
  class Options constructor(internal vararg val accelerators: Accelerator) {

    companion object {
      @JvmStatic val CPU = Options(Accelerator.CPU)
    }
  }

  @Throws(LiteRtException::class)
  fun createInputBuffer(inputName: String, signature: String? = null): TensorBuffer {
    assertNotDestroyed()

    val handle = nativeCreateInputBuffer(handle, model.handle, signature, inputName)
    return TensorBuffer(handle)
  }

  @Throws(LiteRtException::class)
  fun createOutputBuffer(outputName: String, signature: String? = null): TensorBuffer {
    assertNotDestroyed()

    val handle = nativeCreateOutputBuffer(handle, model.handle, signature, outputName)
    return TensorBuffer(handle)
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
      return CompiledModel(
        nativeCreate(env.handle, model.handle, options.accelerators.map { it.value }.toIntArray()),
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
    private external fun nativeCreate(envHandle: Long, modelHandle: Long, options: IntArray): Long

    @JvmStatic
    private external fun nativeCreateInputBuffer(
      compiledModelHandle: Long,
      modelHandle: Long,
      signature: String?,
      inputName: String,
    ): Long

    @JvmStatic
    private external fun nativeCreateOutputBuffer(
      compiledModelHandle: Long,
      modelHandle: Long,
      signature: String?,
      outputName: String,
    ): Long

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
