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

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class MainViewModel(private val imageSegmentationHelper: ImageSegmentationHelper) : ViewModel() {
  companion object {
    fun getFactory(context: Context) =
      object : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
          val imageSegmentationHelper = ImageSegmentationHelper(context)
          return if (modelClass.isAssignableFrom(MainViewModel::class.java)) {
            MainViewModel(imageSegmentationHelper) as T
          } else {
            throw IllegalArgumentException("Unknown ViewModel class: ${modelClass.name}")
          }
        }
      }
  }

  init {
    viewModelScope.launch { imageSegmentationHelper.initSegmenter() }
  }

  private var segmentJob: Job? = null

  private val segmentationUiShareFlow =
    MutableStateFlow<Pair<OverlayInfo?, Long>>(Pair(null, 0L)).also { flow ->
      viewModelScope.launch {
        imageSegmentationHelper.segmentation
          .filter { it.segmentation.masks.isNotEmpty() }
          .map {
            val segmentation = it.segmentation
            val mask = segmentation.masks[0]
            val maskArray = mask.data
            val width = mask.width
            val height = mask.height
            val pixelSize = width * height
            val pixels = IntArray(pixelSize)

            val colorLabels =
              segmentation.coloredLabels.mapIndexed { index, coloredLabel ->
                ColorLabel(index, coloredLabel.label, coloredLabel.argb)
              }
            // Set color for pixels
            for (i in 0 until pixelSize) {
              val colorLabel = colorLabels[maskArray[i].toInt()]
              val color = colorLabel.getColor()
              pixels[i] = color
            }
            // Get image info
            val overlayInfo = OverlayInfo(pixels = pixels, width = width, height = height)

            val inferenceTime = it.inferenceTime
            Pair(overlayInfo, inferenceTime)
          }
          .collect { flow.emit(it) }
      }
    }

  private val mediaUri = MutableStateFlow<Uri>(Uri.EMPTY)

  private val errorMessage =
    MutableStateFlow<Throwable?>(null).also {
      viewModelScope.launch { imageSegmentationHelper.error.collect(it) }
    }

  private val lensFacing = MutableStateFlow(CameraSelector.LENS_FACING_BACK)

  val uiState: StateFlow<UiState> =
    combine(mediaUri, segmentationUiShareFlow, errorMessage, lensFacing) {
        uri,
        segmentationUiPair,
        error,
        lensFace ->
        UiState(
          mediaUri = uri,
          overlayInfo = segmentationUiPair.first,
          inferenceTime = segmentationUiPair.second,
          errorMessage = error?.message,
          lensFacing = lensFace,
        )
      }
      .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), UiState())

  fun flipCamera() {
    val newFacing =
      if (lensFacing.value == CameraSelector.LENS_FACING_BACK) {
        CameraSelector.LENS_FACING_FRONT
      } else {
        CameraSelector.LENS_FACING_BACK
      }

    lensFacing.update { newFacing }
  }

  /**
   * Start segment an image.
   *
   * @param imageProxy contain `imageBitMap` and imageInfo as `image rotation degrees`.
   */
  fun segment(imageProxy: ImageProxy) {
    segmentJob =
      viewModelScope.launch {
        imageSegmentationHelper.segment(imageProxy.toBitmap(), imageProxy.imageInfo.rotationDegrees)
        imageProxy.close()
      }
  }

  /**
   * Start segment an image.
   *
   * @param bitmap Tries to make a new bitmap based on the dimensions of this bitmap, setting the
   *   new bitmap's config to Bitmap.Config.ARGB_8888
   * @param rotationDegrees to correct the rotationDegrees during segmentation
   */
  fun segment(bitmap: Bitmap, rotationDegrees: Int) {
    segmentJob =
      viewModelScope.launch {
        val argbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        imageSegmentationHelper.segment(argbBitmap, rotationDegrees)
      }
  }

  /** Stop current segmentation */
  fun stopSegment() {
    viewModelScope.launch {
      segmentJob?.cancel()
      segmentationUiShareFlow.emit(Pair(null, 0L))
    }
  }

  /** Update display media uri */
  fun updateMediaUri(uri: Uri) {
    if (uri != mediaUri.value || uri.toString().contains("video")) {
      stopSegment()
    }
    mediaUri.update { uri }
  }

  /** Set Accelerator for ImageSegmentationHelper(CPU/NPU/GPU) */
  fun setAccelerator(accleratorEnum: ImageSegmentationHelper.AcceleratorEnum) {
    viewModelScope.launch { imageSegmentationHelper.initSegmenter(accleratorEnum) }
  }

  /** Clear error message after it has been consumed */
  fun errorMessageShown() {
    errorMessage.update { null }
  }
}
