#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_DIR="$SCRIPT_DIR/gen"
VERSION="${RELEASE_VERSION:-0.0.0-nightly-SNAPSHOT}"

mkdir -p "$GEN_DIR"

build_pom_file() {
  local pom_file="$1"
  local artifact="$2"
  local title="$3"
  local version="$4"
  local depends_api="${5:-}"
  local depends_gpu="${6:-}"

  local api_dep=""
  local gpu_dep=""

  local api_version="$version"
  [[ "$api_version" == "0.0.0-nightly-debug-SNAPSHOT" ]] && api_version="0.0.0-nightly-SNAPSHOT"

  if [[ "$depends_api" == "--depends-api" ]]; then
    api_dep=$(cat <<EOF
    <dependency>
      <groupId>com.google.ai.edge.litert</groupId>
      <artifactId>litert-api</artifactId>
      <version>[${api_version}]</version>
    </dependency>
EOF
)
  fi

  if [[ "$depends_gpu" == "--depends-gpu-api" ]]; then
    gpu_dep=$(cat <<EOF
    <dependency>
      <groupId>com.google.ai.edge.litert</groupId>
      <artifactId>litert-gpu-api</artifactId>
      <version>[${api_version}]</version>
    </dependency>
EOF
)
  fi

  cat >"$pom_file" <<EOF
<project xmlns="http://maven.apache.org/POM/4.0.0"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
 xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
 http://maven.apache.org/xsd/maven-4.0.0.xsd">

  <modelVersion>4.0.0</modelVersion>
  <groupId>com.google.ai.edge.litert</groupId>
  <artifactId>${artifact}</artifactId>
  <version>${version}</version>
  <packaging>aar</packaging>

  <name>${title}</name>
  <url>https://tensorflow.org/lite/</url>

  <licenses>
    <license>
      <name>Apache License 2.0</name>
      <url>http://www.apache.org/licenses/LICENSE-2.0.txt</url>
    </license>
  </licenses>

  <dependencies>
${api_dep}
${gpu_dep}
  </dependencies>
</project>
EOF
}

make_placeholder_jar() {
  local file="$1"
  local tmp=$(mktemp)
  echo "placeholder" > "$tmp"
  jar cf "$file" "$tmp"
  rm "$tmp"
}

prepare_pom_and_artifact() {
  local package="$1"
  local title="$2"
  local artifact="$3"
  local version="$4"
  local dep1="${5:-}"
  local dep2="${6:-}"

  local name="${package}-${version}"
  local dst="$GEN_DIR/$name"

  mkdir -p "$dst"

  mv "$artifact" "$dst/$name.aar"

  build_pom_file "$dst/$name.pom" "$package" "$title" "$version" "$dep1" "$dep2"

  make_placeholder_jar "$dst/${package}-${version}-sources.jar"
  make_placeholder_jar "$dst/${package}-${version}-javadoc.jar"
}

BUILD_FLAGS=(
  -c opt
  --cxxopt=--std=c++17
  --config=android_arm64
  --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a
  --repo_env=HERMETIC_PYTHON_VERSION=3.11
  --show_timestamps
)

BUILD_FLAGS+=(${BAZEL_CONFIG_FLAGS:-})

[[ "${USE_LOCAL_TF:-}" == "true" ]] && BUILD_FLAGS+=(--config=use_local_tf)

# TODO(b/503213161): Avoid piggybacking Tensor API's bazel build test on
# LiteRT's wheel kokoro job.
bazel build "${BUILD_FLAGS[@]}" //tensor/...

bazel build "${BUILD_FLAGS[@]}" \
  //tflite/java:tensorflow-lite-api \
  //tflite/java:tensorflow-lite \
  //tflite/java:tensorflow-lite-gpu-api \
  //tflite/java:tensorflow-lite-gpu \
  //tflite/acceleration/configuration:gpu_plugin \
  //tflite/acceleration/configuration:nnapi_plugin

prepare_pom_and_artifact litert-api "LiteRT API" \
  bazel-bin/tflite/java/tensorflow-lite-api.aar "$VERSION"

prepare_pom_and_artifact litert "LiteRT implementation" \
  bazel-bin/tflite/java/tensorflow-lite.aar "$VERSION" \
  --depends-api

prepare_pom_and_artifact litert-gpu-api "LiteRT GPU API" \
  bazel-bin/tflite/java/tensorflow-lite-gpu-api.aar "$VERSION"

prepare_pom_and_artifact litert-gpu "LiteRT GPU implementation" \
  bazel-bin/tflite/java/tensorflow-lite-gpu.aar "$VERSION" \
  --depends-api --depends-gpu-api
