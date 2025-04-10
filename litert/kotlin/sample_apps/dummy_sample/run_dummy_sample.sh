#!/bin/bash

blaze build --config=android_arm64 --android_ndk_min_sdk_version=26 \
  //litert/kotlin/sample_apps/dummy_sample
adb install -r \
  blaze-bin/litert/kotlin/sample_apps/dummy_sample/dummy_sample.apk

adb shell am start -a android.intent.action.MAIN \
  -n com.google.ai.edge.litert.sample.dummy/.MainActivity
