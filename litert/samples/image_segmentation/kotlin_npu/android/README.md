# Guide to prepare and build the app

## Compile the segmentation model for NPU

Follow the Colab (TODO: add link) to compile the sample model, and export
compiled model files to
[Google Play AI pack](https://developer.android.com/google/play/on-device-ai)
directory structure.

## Set up AI Packs

Copy the exported models from previous step to the directory `ai_pack` under
the app's root directory. Then add a `build.gradle.kts` for each of the AI
packs:

```kotlin
// ai_pack/selfie_multiclass/build.gradle.kts

plugins { id("com.android.ai-pack") }

aiPack {
  packName = "selfie_multiclass"  // AI pack directory name
  dynamicDelivery { deliveryType = "on-demand" }
}
```

```kotlin
// ai_pack/selfie_multiclass_mtk/build.gradle.kts

plugins { id("com.android.ai-pack") }

aiPack {
  packName = "selfie_multiclass_mtk"  // AI pack directory name
  dynamicDelivery { deliveryType = "on-demand" }
}
```

Then move the file `ai_pack/device_targeting_configuration.xml` to the `app`
module's directory.

## Set up NPU runtime feature modules

Download the NPU runtime libraries from GitHub (TODO: add link) and unpack it to
the directory `litert_npu_runtime_libraries` under the app's root directory.
Then run the helper script provided to fetch Qualcomm NPU libraries from their
website.

```sh
$ ./litert_npu_runtime_libraries/fetch_qualcomm_library.sh
```

## Build the app bundle

From the app's root directory, run:

```sh
$ ./gradlew bundle
```

And it will produce the app bundle at
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
