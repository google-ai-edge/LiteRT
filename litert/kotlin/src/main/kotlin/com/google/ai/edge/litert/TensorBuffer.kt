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

/** TensorBuffer represents the raw memory where tensor data is stored. */
class TensorBuffer internal constructor(handle: Long) : JniHandle(handle) {
  // TODO(niuchl): Add support for different types of tensor buffers.
  // TODO(niuchl): Add tests for different element types.

  @Throws(LiteRtException::class)
  fun writeInt(data: IntArray) {
    assertNotDestroyed()

    nativeWriteInt(handle, data)
  }

  @Throws(LiteRtException::class)
  fun writeFloat(data: FloatArray) {
    assertNotDestroyed()

    nativeWriteFloat(handle, data)
  }

  @Throws(LiteRtException::class)
  fun writeInt8(data: ByteArray) {
    assertNotDestroyed()

    nativeWriteInt8(handle, data)
  }

  @Throws(LiteRtException::class)
  fun writeBoolean(data: BooleanArray) {
    assertNotDestroyed()

    nativeWriteBoolean(handle, data)
  }

  @Throws(LiteRtException::class)
  fun readInt(): IntArray {
    assertNotDestroyed()

    return nativeReadInt(handle)
  }

  @Throws(LiteRtException::class)
  fun readFloat(): FloatArray {
    assertNotDestroyed()

    return nativeReadFloat(handle)
  }

  @Throws(LiteRtException::class)
  fun readInt8(): ByteArray {
    assertNotDestroyed()

    return nativeReadInt8(handle)
  }

  @Throws(LiteRtException::class)
  fun readBoolean(): BooleanArray {
    assertNotDestroyed()

    return nativeReadBoolean(handle)
  }

  protected override fun destroy() {
    nativeDestroy(handle)
  }

  companion object {
    init {
      System.loadLibrary("litert_jni")
    }

    @JvmStatic private external fun nativeWriteInt(handle: Long, data: IntArray)

    @JvmStatic private external fun nativeWriteFloat(handle: Long, data: FloatArray)

    @JvmStatic private external fun nativeWriteInt8(handle: Long, data: ByteArray)

    @JvmStatic private external fun nativeWriteBoolean(handle: Long, data: BooleanArray)

    @JvmStatic private external fun nativeReadInt(handle: Long): IntArray

    @JvmStatic private external fun nativeReadFloat(handle: Long): FloatArray

    @JvmStatic private external fun nativeReadInt8(handle: Long): ByteArray

    @JvmStatic private external fun nativeReadBoolean(handle: Long): BooleanArray

    @JvmStatic private external fun nativeDestroy(handle: Long)
  }
}

/** The type of the tensor buffer. */
enum class TensorBufferType(private val type: Int) {
  // LINT.IfChange(tensor_buffer_types)
  Unknown(0),
  HostMemory(1),
  Ahwb(2),
  Ion(3),
  DmaBuf(4),
  FastRpc(5),
  GlBuffer(6),
  GlTexture(7),

  // 10-19 are reserved for OpenCL memory objects.
  OpenClBuffer(10),
  OpenClBufferFp16(11),
  OpenClTexture(12),
  OpenClTextureFp16(13),
  OpenClImageBuffer(14),
  OpenClImageBufferFp16(15);

  // LINT.ThenChange(../../../../../../../../../c/litert_tensor_buffer_types.h:tensor_buffer_types)

  companion object {
    fun of(type: Int): TensorBufferType {
      return values().firstOrNull { it.type == type } ?: Unknown
    }
  }
}

/** Requirements for allocating a TensorBuffer. */
class TensorBufferRequirements internal constructor(handle: Long) : JniHandle(handle) {
  override fun destroy() {
    nativeDestroy(handle)
  }

  /** Returns all the types supported by the tensor buffer requirements. */
  @Throws(LiteRtException::class)
  fun supportedTypes(): List<TensorBufferType> {
    val types = nativeGetSupportedTypes(handle)
    return types.map { TensorBufferType.of(it) }
  }

  /** Returns the size of the tensor buffer requirements in bytes. */
  @Throws(LiteRtException::class)
  fun bufferSize(): Int {
    return nativeBufferSize(handle)
  }

  /** Returns the strides of the tensor buffer requirements. */
  @Throws(LiteRtException::class)
  fun strides(): IntArray {
    return nativeGetStrides(handle)
  }

  companion object {
    init {
      System.loadLibrary("litert_jni")
    }

    @JvmStatic private external fun nativeGetSupportedTypes(handle: Long): IntArray

    @JvmStatic private external fun nativeBufferSize(handle: Long): Int

    @JvmStatic private external fun nativeGetStrides(handle: Long): IntArray

    @JvmStatic private external fun nativeDestroy(handle: Long)
  }
}
