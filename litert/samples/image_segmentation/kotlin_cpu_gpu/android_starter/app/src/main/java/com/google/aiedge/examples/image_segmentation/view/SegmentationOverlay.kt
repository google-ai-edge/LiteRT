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

package com.google.aiedge.examples.image_segmentation.view

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.CameraSelector
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.core.graphics.scale
import com.google.aiedge.examples.image_segmentation.OverlayInfo

@Composable
fun SegmentationOverlay(modifier: Modifier = Modifier, overlayInfo: OverlayInfo, lensFacing: Int) {
  Canvas(modifier = modifier) {
    val imageWidth: Float = size.width
    val imageHeight: Float = size.height

    val image =
      Bitmap.createBitmap(
        overlayInfo.pixels,
        overlayInfo.width,
        overlayInfo.height,
        Bitmap.Config.ARGB_8888,
      )
    // TODO: Handle the camera flip and reorient the overlay.
    val orientedBitmap = image

    val scaleBitmap = orientedBitmap.scale(imageWidth.toInt(), imageHeight.toInt(), true)
    drawImage(scaleBitmap.asImageBitmap())
  }
}
