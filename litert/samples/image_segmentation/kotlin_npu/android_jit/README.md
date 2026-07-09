# Guide to prepare and build the app

## Build the app bundle

From the google3/ directory, run:

```sh
$ blaze build --config=android_arm64 --android_ndk_min_sdk_version=23 \
  //litert/samples/image_segmentation/kotlin_npu/android_jit:image_segmentation_aab
```

## Install the app bundle to a device for local testing

```sh
$ bundletool="java -jar /google/bin/releases/bundletool/public/bundletool-all.jar"

$ tmp_dir=$(mktemp -d)

$ $bundletool build-apks \
  --bundle=blaze-bin/litert/samples/image_segmentation/kotlin_npu/android_jit/image_segmentation_aab_unsigned.aab \
  --output="$tmp_dir/image_segmentation.apks" \
  --ks=tools/android/debug_keystore \
  --ks-pass=pass:android \
  --ks-key-alias=androiddebugkey \
  --local-testing \
  --overwrite

$ $bundletool install-apks --apks="$tmp_dir/image_segmentation.apks" \
  --device-groups=<GROUP_FOR_YOUR_DEVICE>
```

Learn more about local testing, see
[this doc](https://developer.android.com/google/play/on-device-ai#local-testing).

### Identify the group for your device

Currently, the following devices are supported:

| Vendor   | SoC Model | Android version | Group Name                 |
|----------|-----------|-----------------|----------------------------|
| Qualcomm | SM8450    |  S+             | Qualcomm_SM8450            |
| Qualcomm | SM8550    |  S+             | Qualcomm_SM8550            |
| Qualcomm | SM8650    |  S+             | Qualcomm_SM8650            |
| Qualcomm | SM8750    |  S+             | Qualcomm_SM8750            |
| Qualcomm | SM8850    |  S+             | Qualcomm_SM8850            |
| Mediatek | MT6878    |  S+             | Mediatek_MT6878            |
| Mediatek | MT6897    |  S+             | Mediatek_MT6897            |
| Mediatek | MT6983    |  S+             | Mediatek_MT6983            |
| Mediatek | MT6985    |  S+             | Mediatek_MT6985            |
| Mediatek | MT6989    |  S+             | Mediatek_MT6989            |
| Mediatek | MT6991    |  S+             | Mediatek_MT6991            |
| Google   | Tensor G3 |  16+            | Google_LGA101              |
| Google   | Tensor G4 |  16+            | Google_LGA101              |
| Google   | Tensor G5 |  16+            | Google_LGA101              |
| Google   | Tensor G6 |  16+            | Google_LGA101              |
