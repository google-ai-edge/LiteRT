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

import java.util.concurrent.atomic.AtomicBoolean

/** Hardware accelerators supported by LiteRT. */
enum class Accelerator private constructor(val value: Int) {
  NONE(0),
  CPU(1),
  GPU(2),
  NPU(3);

  companion object {
    /** Converts an integer value to an [Accelerator]. */
    internal fun of(value: Int): Accelerator {
      return when (value) {
        NONE.value -> NONE
        CPU.value -> CPU
        GPU.value -> GPU
        NPU.value -> NPU
        else -> throw IllegalArgumentException("Invalid accelerator value: $value")
      }
    }
  }
}

/** A base class for all Kotlin types that wrap a JNI handle. */
abstract class JniHandle internal constructor(internal val handle: Long) : AutoCloseable {
  /** Whether the handle has been destroyed. */
  private val destroyed = AtomicBoolean(false)

  /** Asserts that the handle is not destroyed, otherwise throws an [IllegalStateException]. */
  protected fun assertNotDestroyed() {
    if (destroyed.get()) {
      throw IllegalStateException("The handle has been destroyed.")
    }
  }

  /** Clean up resources associated with the handle. */
  protected abstract fun destroy()

  /** Clean up the handle safely to avoid releasing the same JNI handle multiple times. */
  override final fun close() {
    if (destroyed.compareAndSet(false, true)) {
      destroy()
    }
  }
}
