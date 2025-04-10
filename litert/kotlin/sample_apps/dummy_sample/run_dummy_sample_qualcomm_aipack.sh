#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

tmp_dir=$(mktemp -d)

cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

blaze build --config=android_arm64 --android_ndk_min_sdk_version=26 \
  //litert/kotlin/sample_apps/dummy_sample:dummy_sample_aab

bundletool='java -jar /google/bin/releases/bundletool/public/bundletool-all.jar'

$bundletool build-apks \
  --bundle=blaze-bin/litert/kotlin/sample_apps/dummy_sample/dummy_sample_aab_unsigned.aab \
  --output="$tmp_dir/dummy_sample.apks" \
  --local-testing \
  --overwrite

$bundletool install-apks --apks="$tmp_dir/dummy_sample.apks" --device-groups=qti_v75

adb shell am start -a android.intent.action.MAIN \
  -n com.google.ai.edge.litert.sample.dummy/.MainActivity \
  --ez "use_ai_pack_model" true
