/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.examples.image_segmentation

import android.util.Log
import kotlin.math.abs

/** Utility class for tensor operations and debugging. */
object TensorUtils {
  private const val TAG = "TensorUtils"

  /** Set to true to enable tensor statistics logging */
  const val TENSOR_DEBUG = false

  /** Logs tensor statistics for debugging. */
  fun logTensorStats(tensorName: String, data: FloatArray) {
    if (!TENSOR_DEBUG || data.isEmpty()) {
      return
    }

    var min = Float.MAX_VALUE
    var max = Float.MIN_VALUE
    var sum = 0.0f
    var zeroCount = 0
    var nanCount = 0
    var infCount = 0

    // Calculate basic statistics
    for (value in data) {
      when {
        value.isNaN() -> nanCount++
        value.isInfinite() -> infCount++
        abs(value) <= 1e-8 -> zeroCount++
        else -> {
          min = minOf(min, value)
          max = maxOf(max, value)
          sum += value
        }
      }
    }

    val validCount = data.size - nanCount - infCount
    val avg = if (validCount > 0) sum / validCount else 0.0f

    // Log the results
    Log.d(TAG, "$tensorName statistics:")
    Log.d(TAG, "  Size: ${data.size}")
    Log.d(TAG, "  Min: $min, Max: $max, Avg: $avg")
    Log.d(TAG, "  Zero values: $zeroCount (${(zeroCount * 100.0f / data.size).toInt()}%)")

    if (nanCount > 0 || infCount > 0) {
      Log.w(TAG, "  NaN values: $nanCount, Infinite values: $infCount")
    }

    // Sample some values (beginning, middle, end)
    val sampleSize = 5
    val showSamples = data.size > 10

    if (showSamples) {
      val startSamples = data.slice(0 until minOf(sampleSize, data.size))
      Log.d(TAG, "  First $sampleSize values: ${startSamples.joinToString(", ")}")

      if (data.size > sampleSize * 2) {
        val middleIndex = data.size / 2
        val middleSamples = data.slice(middleIndex until minOf(middleIndex + sampleSize, data.size))
        Log.d(TAG, "  Middle $sampleSize values: ${middleSamples.joinToString(", ")}")
      }

      if (data.size > sampleSize) {
        val endSamples = data.slice(maxOf(0, data.size - sampleSize) until data.size)
        Log.d(TAG, "  Last $sampleSize values: ${endSamples.joinToString(", ")}")
      }
    }
  }
}
