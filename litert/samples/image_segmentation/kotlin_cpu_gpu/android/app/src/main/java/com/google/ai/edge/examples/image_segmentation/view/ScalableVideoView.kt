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

package com.google.ai.edge.examples.image_segmentation.view

import android.content.Context
import android.widget.VideoView

class ScalableVideoView(context: Context?) : VideoView(context) {
  private var mVideoWidth = 0
  private var mVideoHeight = 0
  private var displayMode = DisplayMode.ORIGINAL

  enum class DisplayMode {
    ORIGINAL,

    // original aspect ratio
    FULL_SCREEN,

    // fit to screen
    ZOOM, // zoom in
  }

  override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
    var width = getDefaultSize(0, widthMeasureSpec)
    var height = getDefaultSize(mVideoHeight, heightMeasureSpec)
    if (displayMode == DisplayMode.ORIGINAL) {
      if (mVideoWidth > 0 && mVideoHeight > 0) {
        if (mVideoWidth * height > width * mVideoHeight) {
          // video height exceeds screen, shrink it
          height = width * mVideoHeight / mVideoWidth
        } else if (mVideoWidth * height < width * mVideoHeight) {
          // video width exceeds screen, shrink it
          width = height * mVideoWidth / mVideoHeight
        }
      }
    } else if (displayMode == DisplayMode.FULL_SCREEN) {
      // just use the default screen width and screen height
    } else if (displayMode == DisplayMode.ZOOM) {
      // zoom video
      if (mVideoWidth > 0 && mVideoHeight > 0 && mVideoWidth < width) {
        height = mVideoHeight * width / mVideoWidth
      }
    }
    setMeasuredDimension(width, height)
  }

  fun changeVideoSize(width: Int, height: Int) {
    mVideoWidth = width
    mVideoHeight = height

    // not sure whether it is useful or not but safe to do so
    holder.setFixedSize(width, height)
    requestLayout()
    invalidate() // very important, so that onMeasure will be triggered
  }

  fun setDisplayMode(mode: DisplayMode) {
    displayMode = mode
  }
}
