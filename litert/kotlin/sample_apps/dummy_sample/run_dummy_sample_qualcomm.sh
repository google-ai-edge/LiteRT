#!/bin/bash

blaze build --config=android_arm64 --android_ndk_min_sdk_version=26 \
  //litert/kotlin/sample_apps/dummy_sample:dummy_sample_qualcomm_v75
adb install -r \
  blaze-bin/litert/kotlin/sample_apps/dummy_sample/dummy_sample_qualcomm_v75.apk

adb shell am start -a android.intent.action.MAIN \
  -n com.google.ai.edge.litert.sample.dummy/.MainActivity \
  --ez "use_npu_accelerator" true
