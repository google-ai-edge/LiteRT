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

import android.graphics.Color
import android.net.Uri
import androidx.camera.core.CameraSelector
import androidx.compose.runtime.Immutable

@Immutable
class UiState(
  val mediaUri: Uri = Uri.EMPTY,
  val overlayInfo: OverlayInfo? = null,
  val inferenceTime: Long = 0L,
  val errorMessage: String? = null,
  val lensFacing: Int = CameraSelector.LENS_FACING_BACK,
)

@Immutable class OverlayInfo(val pixels: IntArray, val width: Int, val height: Int)

@Immutable
data class ColorLabel(val id: Int, val label: String, val rgbColor: Int) {

  fun getColor(): Int {
    // Use completely transparent for the background color.
    return if (id == 0) Color.TRANSPARENT
    else Color.argb(128, Color.red(rgbColor), Color.green(rgbColor), Color.blue(rgbColor))
  }
}
