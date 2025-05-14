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

import android.util.Log
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.coroutineScope
import com.google.common.util.concurrent.ListenableFuture
import kotlinx.coroutines.guava.future

/** A model provider that provides a model file and relevant information. */
interface ModelProvider {

  /** Model files could be either an asset or a file. */
  enum class Type {
    ASSET,
    FILE,
  }

  fun getType(): Type

  /** Returns true if the model is ready to be used. */
  fun isReady(): Boolean

  /** Returns the path to the model asset or file. */
  fun getPath(): String

  /** Returns the set of accelerators that the model is compatible with. */
  fun getCompatibleAccelerators(): Set<Accelerator>

  /** Downloads the model if it is not available on the device. */
  suspend fun download()

  /**
   * Returns a future that completes when the model is downloaded.
   *
   * NOTE: Kotlin callers should use [download] instead.
   */
  fun downloadFuture(lifecycleOwner: LifecycleOwner): ListenableFuture<Void?> {
    return lifecycleOwner.lifecycle.coroutineScope.future {
      download()
      null
    }
  }

  companion object {
    /** Creates a model provider that represents a model available on the device. */
    @JvmStatic
    fun staticModel(type: Type, path: String, vararg accelerators: Accelerator): ModelProvider {
      return object : ModelProvider {
        override fun getType() = type

        override fun isReady() = true

        override fun getPath() = path

        // TODO(niuchl): TBD for the default accelerator(?).
        override fun getCompatibleAccelerators() =
          if (accelerators.isEmpty()) setOf(Accelerator.CPU) else setOf(*accelerators)

        override suspend fun download() {}
      }
    }
  }
}

/** ModelSelector allows to dynamically select a [ModelProvider] from a given set of providers. */
class ModelSelector constructor(private val modelProviders: Set<ModelProvider>) {

  constructor(vararg modelProviders: ModelProvider) : this(setOf(*modelProviders)) {}

  /**
   * Selects a [ModelProvider] from the given set based on the availability of model files and
   * accelerators, in the order of NPU, GPU, CPU.
   *
   * @return the selected [ModelProvider], which is guaranteed to be ready to use.
   * @throws IllegalStateException if no model is available.
   */
  @Throws(IllegalStateException::class)
  suspend fun selectModel(env: Environment): ModelProvider {
    val accelerators = env.getAvailableAccelerators()

    val npuModelProvider =
      modelProviders.firstOrNull { it.getCompatibleAccelerators().contains(Accelerator.NPU) }
    if (npuModelProvider != null && accelerators.contains(Accelerator.NPU)) {
      try {
        if (!npuModelProvider.isReady()) {
          npuModelProvider.download()
        }
        return ensureModelFileAvailable(npuModelProvider)
      } catch (e: IllegalStateException) {
        Log.e(TAG, "Failed to download NPU model: ${e.message}")
      }
    }

    val gpuModelProvider =
      modelProviders.firstOrNull {
        it.getCompatibleAccelerators().contains(Accelerator.GPU) && it != npuModelProvider
      }
    if (gpuModelProvider != null && accelerators.contains(Accelerator.GPU)) {
      try {
        if (!gpuModelProvider.isReady()) {
          gpuModelProvider.download()
        }
        return ensureModelFileAvailable(gpuModelProvider)
      } catch (e: IllegalStateException) {
        Log.e(TAG, "Failed to download GPU model: ${e.message}")
      }
    }

    val cpuModelProvider =
      modelProviders.firstOrNull {
        it.getCompatibleAccelerators().contains(Accelerator.CPU) &&
          it != npuModelProvider &&
          it != gpuModelProvider
      }
    if (cpuModelProvider != null && accelerators.contains(Accelerator.CPU)) {
      try {
        if (!cpuModelProvider.isReady()) {
          cpuModelProvider.download()
        }
        return ensureModelFileAvailable(cpuModelProvider)
      } catch (e: IllegalStateException) {
        Log.e(TAG, "Failed to download CPU model: ${e.message}")
      }
    }

    throw IllegalStateException("No model is available.")
  }

  /**
   * Returns a future that completes when the model is selected and ready to use.
   *
   * NOTE: Kotlin callers should use [selectModel] instead.
   */
  fun selectModelFuture(
    env: Environment,
    lifecycleOwner: LifecycleOwner,
  ): ListenableFuture<ModelProvider> {
    return lifecycleOwner.lifecycle.coroutineScope.future { selectModel(env) }
  }

  @Throws(IllegalStateException::class)
  private fun ensureModelFileAvailable(modelProvider: ModelProvider): ModelProvider {
    if (!modelProvider.isReady()) {
      throw IllegalStateException("Model is not ready to be used yet.")
    }
    return modelProvider
  }

  companion object {
    private const val TAG = "ModelSelector"
  }
}
