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
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.util.Random
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext

class ImageSegmentationHelper(private val context: Context) {
  companion object {
    private const val TAG = "ImageSegmentation"
  }

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

  private lateinit var model: CompiledModel
  private val coloredLabels: List<ColoredLabel> = coloredLabels()

  /** Init a CompiledModel from selfie_multiclass_256x256. */
  suspend fun initSegmenter(acceleratorEnum: AcceleratorEnum = AcceleratorEnum.CPU) {
    val options =
      when (acceleratorEnum) {
        AcceleratorEnum.CPU -> CompiledModel.Options(Accelerator.CPU)
        AcceleratorEnum.NPU -> CompiledModel.Options(Accelerator.NPU)
        AcceleratorEnum.GPU -> CompiledModel.Options(Accelerator.GPU)
      }
    try {
      val env: Environment? =
        if (acceleratorEnum == AcceleratorEnum.NPU) {
          Environment.create(BuiltinNpuAcceleratorProvider(context))
        } else {
          null
        }
      model =
        CompiledModel.create(
          context.assets,
          if (acceleratorEnum == AcceleratorEnum.NPU) "selfie_multiclass_256x256_SM8750.tflite"
          else "selfie_multiclass_256x256.tflite",
          options,
          env,
        )
    } catch (e: Exception) {
      Log.i(TAG, "Create LiteRT from deeplab_v3 is failed ${e.message}")
      _error.emit(e)
    }
  }

  suspend fun segment(bitmap: Bitmap, rotationDegrees: Int) {
    try {
      withContext(Dispatchers.IO) {
        val startTime = SystemClock.uptimeMillis()
        val rotation = -rotationDegrees / 90
        val (h, w) = Pair(256, 256)
        var image = bitmap.scale(w, h, true)
        image = rot90Clockwise(image, rotation)
        val inputFloatArray = normalize(image, 127.5f, 127.5f)
        val segmentResult = segment(inputFloatArray)
        val inferenceTime = SystemClock.uptimeMillis() - startTime
        if (isActive) {
          _segmentation.emit(SegmentationResult(segmentResult, inferenceTime))
        }
      }
    } catch (e: Exception) {
      Log.i(TAG, "Image segment error occurred: ${e.message}")
      _error.emit(e)
    }
  }

  private fun rot90Clockwise(image: Bitmap, numRotation: Int): Bitmap {
    val effectiveRotation = numRotation % 4

    if (effectiveRotation == 0) {
      return image
    }

    val (w, h) = Pair(image.width, image.height)

    val matrix = Matrix()
    matrix.postTranslate(w * -0.5f, h * -0.5f)
    matrix.postRotate(-90f * effectiveRotation)
    val newW = if (effectiveRotation % 2 == 0) w else h
    val newH = if (effectiveRotation % 2 == 0) h else w
    matrix.postTranslate(newW * 0.5f, newH * 0.5f)
    return Bitmap.createBitmap(image, 0, 0, w, h, matrix, false)
  }

  private fun normalize(image: Bitmap, mean: Float, stddev: Float): FloatArray {
    val (width, height) = Pair(image.width, image.height)
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
    if (!this::model.isInitialized) {
      return Segmentation(emptyList(), emptyList())
    }
    val (h, w, c) = Triple(256, 256, 6)
    val inputBuffers = model.createInputBuffers()
    val outputBuffers = model.createOutputBuffers()

    inputBuffers[0].writeFloat(inputFloatArray)
    model.run(inputBuffers, outputBuffers)

    val outputBuffer = FloatBuffer.wrap(outputBuffers.get(0).readFloat())

    val inferenceData = InferenceData(width = w, height = h, channels = c, buffer = outputBuffer)
    val mask = processImage(inferenceData)
    return Segmentation(
      listOf(Mask(mask, inferenceData.width, inferenceData.height)),
      coloredLabels,
    )
  }

  private fun processImage(inferenceData: InferenceData): ByteBuffer {
    val mask = ByteBuffer.allocateDirect(inferenceData.width * inferenceData.height)
    for (i in 0 until inferenceData.height) {
      for (j in 0 until inferenceData.width) {
        val offset = inferenceData.channels * (i * inferenceData.width + j)

        var maxIndex = 0
        var maxValue = inferenceData.buffer.get(offset)

        for (index in 1 until inferenceData.channels) {
          if (inferenceData.buffer.get(offset + index) > maxValue) {
            maxValue = inferenceData.buffer.get(offset + index)
            maxIndex = index
          }
        }

        mask.put(i * inferenceData.width + j, maxIndex.toByte())
      }
    }

    return mask
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
    val colors = MutableList(labels.size) { ColoredLabel(labels[0], "", Color.BLACK) }

    val random = Random()
    val goldenRatioConjugate = 0.618033988749895
    var hue = random.nextDouble()

    for (idx in 1 until labels.size) {
      hue += goldenRatioConjugate
      hue %= 1.0
      val color = Color.HSVToColor(floatArrayOf(hue.toFloat() * 360, 0.7f, 0.8f))
      colors[idx] = ColoredLabel(labels[idx], "", color)
    }

    return colors
  }

  data class Mask(val data: ByteBuffer, val width: Int, val height: Int)

  data class Segmentation(val masks: List<Mask>, val coloredLabels: List<ColoredLabel>)

  data class ColoredLabel(val label: String, val displayName: String, val argb: Int)

  enum class AcceleratorEnum {
    CPU,
    NPU,
    GPU,
  }

  data class SegmentationResult(val segmentation: Segmentation, val inferenceTime: Long)

  data class InferenceData(
    val width: Int,
    val height: Int,
    val channels: Int,
    val buffer: FloatBuffer,
  )
}
