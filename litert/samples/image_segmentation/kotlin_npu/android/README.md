# Guide to prepare and build the app

LiteRT NPU acceleration is only available through an Early Access Program. If you are not already enrolled, [sign up](https://forms.gle/CoH4jpLwxiEYvDvF6).

See [NPU acceleration instruction](https://ai.google.dev/edge/litert/next/eap/npu) for more information about compiling NPU models and setup NPU runtime.

## Performance numbers

*   Measured on Samsung S25 Ultra
*   Synchronized execution w/o zero copy buffer interop
*   W/O pre/post processing
  *   CPU Backend: 120 - 140 ms
  *   GPU Backend: 40 - 50 ms
  *   NPU Backend: 6 - 12 ms

## Build the app bundle

WARNING: Before building the app, please follow instructions above to setup NPU
models and runtime correctly.

Please make sure your AI Pack and NPU runtime are being placed under the project
root folder (current folder for this gradle project) and copy the 
`device_targeting_configuration.xml` from your AI Pack to `./app` folder.

From the app's root directory, run:

```sh
$ ./gradlew bundle
```

And it will produce the app bundle at under thde `./app` folder
`./build/outputs/bundle/release/app-release.aab`.

## Install the app bundle to a device for local testing

Download `bundletool` from [GitHub](https://github.com/google/bundletool/releases).

```sh
$ bundletool="java -jar /path/to/the/download/bundletool-all.jar"

$ tmp_dir=$(mktemp -d)

$ $bundletool build-apks \
  --bundle=./build/outputs/bundle/release/app-release.aab \
  --output="$tmp_dir/image_segmentation.apks" \
  --local-testing \
  --overwrite \
  --ks=tools/android/debug_keystore \
  --ks-pass=pass:android \
  --ks-key-alias=androiddebugkey

$ $bundletool install-apks --apks="$tmp_dir/image_segmentation.apks" \
  --device-groups=<GROUP_FOR_YOUR_DEVICE>
```

Learn more about local testing, see [this doc](https://developer.android.com/google/play/on-device-ai#local-testing).

### Identify the group for your device

Currently, the following devices are supported:

| Vendor   | SoC Model | Android version | Group Name                 |
|----------|-----------|-----------------|----------------------------|
| Qualcomm | SM8450    |  S+             | Qualcomm_SM8450            |
| Qualcomm | SM8550    |  S+             | Qualcomm_SM8550            |
| Qualcomm | SM8650    |  S+             | Qualcomm_SM8650            |
| Qualcomm | SM8750    |  S+             | Qualcomm_SM8750            |
| Mediatek | MT6878    |  15             | Mediatek_MT6878_ANDROID_15 |
| Mediatek | MT6897    |  15             | Mediatek_MT6897_ANDROID_15 |
| Mediatek | MT6983    |  15             | Mediatek_MT6983_ANDROID_15 |
| Mediatek | MT6985    |  15             | Mediatek_MT6985_ANDROID_15 |
| Mediatek | MT6989    |  15             | Mediatek_MT6989_ANDROID_15 |
| Mediatek | MT6991    |  15             | Mediatek_MT6991_ANDROID_15 |
