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

package com.google.aiedge.examples.image_segmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.core.graphics.scale
import com.google.aiedge.examples.image_segmentation.TensorUtils.logTensorStats
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import kotlin.random.Random
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext

class ImageSegmentationHelper(private val context: Context) {
  /** As the result of image segmentation, this value emits map of probabilities */
  val segmentation: SharedFlow<SegmentationResult>
    get() = _segmentation

  private val _segmentation =
    MutableSharedFlow<SegmentationResult>(
      extraBufferCapacity = 64,
      onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )

  val error: SharedFlow<Throwable?>
    get() = _error

  private val _error = MutableSharedFlow<Throwable?>()

  private val coloredLabels: List<ColoredLabel> = coloredLabels()

  // Accessed only with singleThreadDispatcher.
  private var segmenter: Segmenter? = null
  private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")

  /** Init a CompiledModel from AI Pack. */
  suspend fun initSegmenter(acceleratorEnum: AcceleratorEnum = AcceleratorEnum.CPU) {
    // TODO: The user will implement this method in the tutorial.
  }

  /** Cleanup resources when the helper is no longer needed */
  suspend fun cleanup() {
    try {
      withContext(singleThreadDispatcher) {
        segmenter?.cleanup()
        segmenter = null
        Log.d(TAG, "Destroyed the image segmenter")
      }
    } catch (e: Exception) {
      Log.e(TAG, "Error during cleanup: ${e.message}")
    }
  }

  suspend fun segment(bitmap: Bitmap, rotationDegrees: Int) {
    // TODO: The user will implement this method in the tutorial.
  }

  private class Segmenter(
    // TODO: Add CompiledModel argument
    private val coloredLabels: List<ColoredLabel>
  ) {
    // TODO: Add input/output buffers private vals

    fun cleanup() {
      // TODO: Cleanup buffers and model
    }

    fun segment(bitmap: Bitmap, rotationDegrees: Int): SegmentationResult {
      // TODO: The user will implement this method in the tutorial.
      // For now, we'll just return an empty result.
      return SegmentationResult(Segmentation(emptyList(), emptyList()), 0)
    }

    private fun rot90Clockwise(image: Bitmap, numRotation: Int): Bitmap {
      val effectiveRotation = numRotation % 4

      if (effectiveRotation == 0) {
        return image
      }

      val w = image.width
      val h = image.height

      val matrix = Matrix()
      matrix.postTranslate(w * -0.5f, h * -0.5f)
      matrix.postRotate(-90f * effectiveRotation)
      val newW = if (effectiveRotation % 2 == 0) w else h
      val newH = if (effectiveRotation % 2 == 0) h else w
      matrix.postTranslate(newW * 0.5f, newH * 0.5f)
      return Bitmap.createBitmap(image, 0, 0, w, h, matrix, false)
    }

    private fun normalize(image: Bitmap, mean: Float, stddev: Float): FloatArray {
      val width = image.width
      val height = image.height
      val numPixels = width * height
      val pixelsIntArray = IntArray(numPixels)
      val outputFloatArray = FloatArray(numPixels * 3) // 3 channels (R, G, B)

      image.getPixels(pixelsIntArray, 0, width, 0, 0, width, height)

      for (i in 0 until numPixels) {
        val pixel = pixelsIntArray[i]

        // Extract channels (ARGB_8888 format assumed)
        val (r, g, b) =
          Triple(
            Color.red(pixel).toFloat(),
            Color.green(pixel).toFloat(),
            Color.blue(pixel).toFloat(),
          )

        // Normalize and store in interleaved format
        val outputBaseIndex = i * 3
        outputFloatArray[outputBaseIndex + 0] = (r - mean) / stddev // Red
        outputFloatArray[outputBaseIndex + 1] = (g - mean) / stddev // Green
        outputFloatArray[outputBaseIndex + 2] = (b - mean) / stddev // Blue
      }

      return outputFloatArray
    }

    private fun segment(inputFloatArray: FloatArray): Segmentation {
      // TODO: The user will implement this method in the tutorial.
      // For now, we'll just return an empty result.
      return Segmentation(emptyList(), emptyList())
    }

    private fun processImage(inferenceData: InferenceData): ByteBuffer {
      // TODO: The user will implement this method in the tutorial.
      // For now, we'll just return an empty result.
      return ByteBuffer.allocate(0)
    }
  }

  private fun coloredLabels(): List<ColoredLabel> {
    val labels =
      listOf(
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "dining table",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv",
        "------",
      )
    var hue = Random.nextDouble()

    return List(labels.size) { i ->
      val color =
        if (i == 0) {
          Color.BLACK
        } else {
          hue += goldenRatioConjugate
          hue %= 1.0
          Color.HSVToColor(floatArrayOf(hue.toFloat() * 360, 0.7f, 0.8f))
        }
      ColoredLabel(labels[i], "", color)
    }
  }

  data class Mask(val data: ByteBuffer, val width: Int, val height: Int)

  data class Segmentation(val masks: List<Mask>, val coloredLabels: List<ColoredLabel>)

  data class ColoredLabel(val label: String, val displayName: String, val argb: Int)

  enum class AcceleratorEnum {
    CPU,
    GPU,
  }

  data class SegmentationResult(val segmentation: Segmentation, val inferenceTime: Long)

  data class InferenceData(
    val width: Int,
    val height: Int,
    val channels: Int,
    val buffer: FloatBuffer,
  )

  private companion object {
    const val TAG = "ImageSegmentation"
    const val goldenRatioConjugate = 0.618033988749895
    // TODO: Add toAccelerator method in the tutorial
  }
}
