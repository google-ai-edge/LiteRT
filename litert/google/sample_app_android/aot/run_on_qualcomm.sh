#!/bin/bash

blaze build --config=android_arm64 --android_ndk_min_sdk_version=26 \
  //litert/google/sample_app_android/aot:testapp_qualcomm

adb install -r \
  blaze-bin/litert/google/sample_app_android/aot/testapp_qualcomm.apk

adb shell am start -a android.intent.action.MAIN \
  -n org.odml.litert.litert.aot/.MainActivity
