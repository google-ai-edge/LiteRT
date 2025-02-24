#!/bin/bash

blaze build --config=android_arm64 --android_ndk_min_sdk_version=26 \
  //third_party/odml/litert/litert/google/sample_app_android/aot:testapp_google_tensor

adb install -r \
  blaze-bin/third_party/odml/litert/litert/google/sample_app_android/aot/testapp_google_tensor.apk

adb root && adb shell setprop vendor.edgetpu.service.allow_unlisted_app true

adb shell am start -a android.intent.action.MAIN \
  -n org.tensorflow.tflite.experimental.litert.aot/.MainActivity
