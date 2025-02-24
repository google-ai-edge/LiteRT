#!/bin/bash

blaze build --config=android_arm64 --android_ndk_min_sdk_version=26 \
  //third_party/odml/litert/litert/google/sample_app_android/aot:testapp_qualcomm

adb install -r \
  blaze-bin/third_party/odml/litert/litert/google/sample_app_android/aot/testapp_qualcomm.apk

adb shell am start -a android.intent.action.MAIN \
  -n org.tensorflow.tflite.experimental.litert.aot/.MainActivity
