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

import android.content.pm.PackageManager
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.widget.LinearLayout
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.aiedge.examples.image_segmentation.UiState
import com.google.common.util.concurrent.ListenableFuture
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

@Composable
fun CameraScreen(
  uiState: UiState,
  modifier: Modifier = Modifier,
  onImageAnalyzed: (ImageProxy) -> Unit,
) {
  val context = LocalContext.current
  val launcher =
    rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
      isGranted: Boolean ->
      if (isGranted) {
        // Do nothing
      } else {
        // Permission Denied
        Toast.makeText(context, "Camera permission is denied", Toast.LENGTH_SHORT).show()
      }
    }

  LaunchedEffect(key1 = uiState.errorMessage) {
    if (
      ContextCompat.checkSelfPermission(context, android.Manifest.permission.CAMERA) !=
        PackageManager.PERMISSION_GRANTED
    ) {
      launcher.launch(android.Manifest.permission.CAMERA)
    }
  }

  val width = LocalConfiguration.current.screenWidthDp
  val height = width / 3 * 4
  Box(modifier = Modifier.width(width.dp).height(height.dp)) {
    CameraPreview(onImageAnalyzed = { imageProxy -> onImageAnalyzed(imageProxy) })
    if (uiState.overlayInfo != null) {
      SegmentationOverlay(modifier = Modifier.fillMaxSize(), overlayInfo = uiState.overlayInfo)
    }
  }
}

@Composable
fun CameraPreview(modifier: Modifier = Modifier, onImageAnalyzed: (ImageProxy) -> Unit) {
  val context = LocalContext.current
  val lifecycleOwner = LocalLifecycleOwner.current
  val cameraProviderFuture by remember {
    mutableStateOf(ProcessCameraProvider.getInstance(context))
  }

  DisposableEffect(lifecycleOwner) { onDispose { cameraProviderFuture.get().unbindAll() } }

  AndroidView(
    modifier = modifier,
    factory = {
      val previewView =
        PreviewView(it).apply {
          layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT)
          scaleType = PreviewView.ScaleType.FILL_START
        }

      val executor = Executors.newSingleThreadExecutor()
      cameraProviderFuture.addListener(
        {
          bindCameraUseCases(
            lifecycleOwner = lifecycleOwner,
            cameraProviderFuture = cameraProviderFuture,
            executor = executor,
            previewView = previewView,
            onImageAnalyzed = onImageAnalyzed,
          )
        },
        ContextCompat.getMainExecutor(context),
      )

      previewView
    },
  )
}

fun bindCameraUseCases(
  lifecycleOwner: LifecycleOwner,
  cameraProviderFuture: ListenableFuture<ProcessCameraProvider>,
  executor: ExecutorService,
  previewView: PreviewView,
  onImageAnalyzed: (ImageProxy) -> Unit,
) {
  val resolutionSelector =
    ResolutionSelector.Builder()
      // Tell CameraX to prefer 4:3, and fall back to auto if 4:3 is unavailable
      .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
      .build()

  // val preview: Preview = Preview.Builder().setTargetResolution(targetSize).build()
  val preview: Preview = Preview.Builder().build()

  preview.surfaceProvider = previewView.surfaceProvider

  val cameraSelector: CameraSelector =
    CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()
  val imageAnalysis =
    ImageAnalysis.Builder()
      .setResolutionSelector(resolutionSelector)
      .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
      .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
      .build()

  imageAnalysis.setAnalyzer(executor) { imageProxy -> onImageAnalyzed(imageProxy) }
  val cameraProvider = cameraProviderFuture.get()

  cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, imageAnalysis, preview)
}
