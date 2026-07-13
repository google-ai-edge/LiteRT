# Google internal tips

## Manual testing of Dispatch API

This doc explains some steps needed to test Dispatch API on Android devices
manually.

To run cc_test(), you should use
//litert/google/run_test_on_android.sh script instead.

### Build the run_model for Android

```
blaze --blazerc=/dev/null build -c opt --config=android_arm64 --copt=-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1 litert/tools:run_model
```

### Dispatch API for Pixel

```
DEVICE_BIN_PATH=/data/local/tmp
blaze --blazerc=/dev/null build -c opt --config=android_arm64 litert/vendors/google_tensor/dispatch:dispatch_api_so
adb push blaze-bin/litert/vendors/google_tensor/dispatch/libLiteRtDispatch_GoogleTensor.so ${DEVICE_BIN_PATH}
```

Enable Dispatch API for 3rd party apps

```
adb ${ADB_OPTION} root
adb ${ADB_OPTION} shell setprop vendor.edgetpu.service.allow_unlisted_app true
```

### Dispatch API for Qualcomm

```
DEVICE_BIN_PATH=/data/local/tmp
blaze --blazerc=/dev/null build -c opt --config=android_arm64 litert/vendors/qualcomm/dispatch:dispatch_api_so
adb push blaze-bin/litert/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so ${DEVICE_BIN_PATH}
```

Sync QNN Libraries.

```
QNN_SDK_ROOT=third_party/qairt/latest
DEVICE_BIN_PATH=/data/local/tmp
adb ${ADB_OPTION} push --sync $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp*Stub.so ${DEVICE_BIN_PATH}
adb ${ADB_OPTION} push --sync $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so ${DEVICE_BIN_PATH}
adb ${ADB_OPTION} push --sync $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so ${DEVICE_BIN_PATH}
adb ${ADB_OPTION} push --sync $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so ${DEVICE_BIN_PATH}
adb ${ADB_OPTION} push --sync $QNN_SDK_ROOT/lib/hexagon-*/unsigned/libQnnHtp*Skel.so ${DEVICE_BIN_PATH}
```

### Dispatch API for MediaTek

```
DEVICE_BIN_PATH=/data/local/tmp
blaze --blazerc=/dev/null build -c opt --config=android_arm64 litert/vendors/mediatek/dispatch:dispatch_api_so
adb push blaze-bin/litert/vendors/mediatek/dispatch/libLiteRtDispatch_Mediatek.so ${DEVICE_BIN_PATH}
```

## GPU test on your gLinux

You need to use `--config=experimental-crosstool-cots-linux --linkopt=-Wl,
--enable-new-dtags` to build binaries.
And need to use `LD_LIBRARY_PATH=/lib/x86_64-linux-gnu` to run binaries.

Note: Even though your machine don't have GPU, you can still use WebGPU / Vulkan
via CPU emulation(llvmpipe).

Use `vulkaninfo --summary` to check installed Vulkan drivers.

Example of benchmark_model.

```
blaze build -c opt --config=experimental-crosstool-cots-linux --linkopt=-Wl,--enable-new-dtags \
  //litert/tools:benchmark_model

LD_LIBRARY_PATH=/lib/x86_64-linux-gnu blaze-bin/litert/tools/benchmark_model \
  --graph=<model_path> --use_gpu=true
```

Example of gpu_numerics_check.

```
blaze build -c opt --config=experimental-crosstool-cots-linux --linkopt=-Wl,--enable-new-dtags \
  //litert/tools:gpu_numerics_check

LD_LIBRARY_PATH=/lib/x86_64-linux-gnu  blaze-bin/litert/tools/gpu_numerics_check \
  --graph=<model_path>
```

## On-Device Benchmarking Across Backends

To benchmark on a locally connected Android device, use
`benchmark_model_device_<backend>`:

```bash
blaze run -c opt --config=android_arm64 \
  --copt=-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1 \
  //litert/tools/google:benchmark_model_device_qualcomm \
  -- --graph=<path to tflite>
```

Supported backends include `qualcomm`, `google_tensor`, `mediatek`, `gpu`, and
`cpu`.
