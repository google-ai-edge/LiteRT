#!/usr/bin/env bash
# Copyright 2024 The AI Edge LiteRT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
set -ex

# Run this script under the root directory.

# Expected env variables:
#  - (Optional) RELEASE_VERSION (default=0.0.0-nightly-SNAPSHOT)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_DIR="${SCRIPT_DIR}/gen"

# Builds pom file for TFLite artifacts.
#
# Args:
# 1. POM_FILE destination path of generated pom.
# 2. TFLITE_ARTIFACT which artifact is to be generated, e.g. "litert-api".
# 3. TITLE brief human-readable title of the Maven package, e.g. "LiteRT API".
# 4. TFLITE_VERSION the version of the artifact.
# 5. --depends-api to add com.google.ai.edge.litert:litert-api into
#    POM dependency.
# 6. --depends-gpu-api to add com.google.ai.edge.litert:litert-gpu-api into
#    POM dependency.
build_pom_file() {
  local POM_FILE="$1"
  local TFLITE_ARTIFACT="$2"
  local TITLE="$3"
  local TFLITE_VERSION="$4"

  API_DEPENDENCY=""
  GPU_API_DEPENDENCY=""

  TFLITE_API_VERSION="$TFLITE_VERSION"
  if [[ "$TFLITE_API_VERSION" == "0.0.0-nightly-debug-SNAPSHOT" ]]; then
    # API doesn't have debug version.
    TFLITE_API_VERSION="0.0.0-nightly-SNAPSHOT"
  fi

  # Please note that TFLite runtime libraries depend on the API library with
  # exact same version, so that an old runtime never pulls in a new API (which
  # may has new methods / classes which are not implemented in the runtime).
  if [[ "$5" == "--depends-api" ]]; then
    API_DEPENDENCY=$(cat <<-END
    <dependency>
      <groupId>com.google.ai.edge.litert</groupId>
      <artifactId>litert-api</artifactId>
      <version>[${TFLITE_API_VERSION}]</version>
    </dependency>
END
)
  fi
  if [[ "$6" == "--depends-gpu-api" ]]; then
    GPU_API_DEPENDENCY=$(cat <<-END
    <dependency>
      <groupId>com.google.ai.edge.litert</groupId>
      <artifactId>litert-gpu-api</artifactId>
      <version>[${TFLITE_API_VERSION}]</version>
    </dependency>
END
)
  fi

  cat >"${POM_FILE}" <<EOF
<project
    xmlns="http://maven.apache.org/POM/4.0.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>

  <groupId>com.google.ai.edge.litert</groupId>
  <artifactId>${TFLITE_ARTIFACT}</artifactId>
  <version>${TFLITE_VERSION}</version>
  <packaging>aar</packaging>

  <name>${TITLE}</name>
  <url>https://tensorflow.org/lite/</url>
  <description>A library that helps deploy machine learning models on mobile devices</description>

  <licenses>
    <license>
      <name>The Apache Software License, Version 2.0</name>
      <url>http://www.apache.org/licenses/LICENSE-2.0.txt</url>
    </license>
  </licenses>

  <developers>
    <developer>
      <name>Google AI Edge Authors</name>
      <organization>TensorFlow</organization>
      <organizationUrl>https://tensorflow.org</organizationUrl>
    </developer>
  </developers>

  <scm>
    <connection>scm:git:git://github.com/tensorflow/tensorflow.git</connection>
    <developerConnection>scm:git:ssh://github.com:tensorflow/tensorflow.git</developerConnection>
    <url>https://github.com/tensorflow/tensorflow/tree/master/</url>
  </scm>

  <build>
    <plugins>
      <plugin>
        <groupId>com.simpligility.maven.plugins</groupId>
        <artifactId>android-maven-plugin</artifactId>
        <version>4.1.0</version>
        <extensions>true</extensions>
        <configuration>
            <sign>
                <debug>false</debug>
            </sign>
        </configuration>
      </plugin>
    </plugins>
  </build>

  <dependencies>
${API_DEPENDENCY}
${GPU_API_DEPENDENCY}
  </dependencies>
</project>
EOF
}

# Makes a placeholder jar on the specified path.
function make_placeholder_jar() {
  echo "This is a placeholder JAR." > "/tmp/readme.md"
  jar cf "$1" "/tmp/readme.md"
}

# Prepares POM and jar/aar artifacts in ${GEN_DIR}.
#
# Args:
# - 1. Package name, e.g. `litert-api`.
# - 2. Package title, e.g. `LiteRT API`.
# - 3. Artifact path
# - 4. Version
# - 5. --depends-api to add litert-api into dependencies
# - 6. --depends-gpu-api to add litert-gpu-api into dependencies
prepare_pom_and_artifact() {
  local PACKAGE="$1"
  local TITLE="$2"
  local ARTIFACT_PATH="$3"
  local VERSION="$4"
  NAME="${PACKAGE}-${VERSION}"
  DST_DIR="${GEN_DIR}/${NAME}"

  mkdir -p "${DST_DIR}"

  mv "${ARTIFACT_PATH}" "${DST_DIR}/${NAME}.aar"

  POM_FILE="${DST_DIR}/${NAME}.pom"
  build_pom_file "${POM_FILE}" "${PACKAGE}" "${TITLE}" "${VERSION}" "$5" "$6"

  # Source JAR, javadoc JAR and pgp signs are required to publish to OSSRH.
  # https://central.sonatype.org/publish/requirements/
  SOURCES_JAR="${DST_DIR}/${PACKAGE}-${VERSION}-sources.jar"
  JAVADOC_JAR="${DST_DIR}/${PACKAGE}-${VERSION}-javadoc.jar"
  make_placeholder_jar "${SOURCES_JAR}"
  make_placeholder_jar "${JAVADOC_JAR}"
}

# To configure Android via TF's 'configure' script.
export TF_SET_ANDROID_WORKSPACE=1

BUILD_FLAGS=("-c" "opt" \
    "--cxxopt=--std=c++17" \
    "--config=android_arm64" \
    "--fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a" \
    "--define=android_dexmerger_tool=d8_dexmerger" \
    "--define=android_incremental_dexing_tool=d8_dexbuilder" \
    "--repo_env=HERMETIC_PYTHON_VERSION=3.11" \
    "--show_timestamps")

# Merge extra config flags from the environment
BUILD_FLAGS+=(${BAZEL_CONFIG_FLAGS})

# Conditionally use local submodules vs http_archve tf
if [[ "${USE_LOCAL_TF}" == "true" ]]; then
  BUILD_FLAGS+=("--config=use_local_tf")
fi

if [[ "$BUILD_LITERT_KOTLIN_API" == "true" ]]; then
  echo "Building Litert Kotlin API."
  bazel build "${BUILD_FLAGS[@]}" --action_env ANDROID_NDK_API_LEVEL=26 \
      //litert/kotlin:litert_kotlin_api
else
  echo "Skipping building Litert Kotlin API."
fi

bazel build "${BUILD_FLAGS[@]}" \
    //tflite/java:tensorflow-lite-api \
    //tflite/java:tensorflow-lite \
    //tflite/java:tensorflow-lite-gpu-api \
    //tflite/java:tensorflow-lite-gpu \
    //tflite/acceleration/configuration:gpu_plugin \
    //tflite/acceleration/configuration:nnapi_plugin
    # //tflite/delegates/hexagon/java:tensorflow-lite-hexagon

export VERSION="${RELEASE_VERSION:-0.0.0-nightly-SNAPSHOT}"

prepare_pom_and_artifact "litert-api" "LiteRT API" \
    "bazel-bin/tflite/java/tensorflow-lite-api.aar" "${VERSION}"
prepare_pom_and_artifact "litert" "LiteRT implementation" \
    "bazel-bin/tflite/java/tensorflow-lite.aar" "${VERSION}" \
    --depends-api
prepare_pom_and_artifact "litert-gpu-api" "LiteRT GPU API" \
    "bazel-bin/tflite/java/tensorflow-lite-gpu-api.aar" "${VERSION}"
prepare_pom_and_artifact "litert-gpu" "LiteRT GPU implementation" \
    "bazel-bin/tflite/java/tensorflow-lite-gpu.aar" "${VERSION}" \
    --depends-api --depends-gpu-api
# prepare_pom_and_artifact "litert-hexagon" "LiteRT Hexagon" \
#     "bazel-bin/tflite/delegates/hexagon/java/tensorflow-lite-hexagon.aar" \
#     "${VERSION}"

if [[ "$VERSION" == "0.0.0-nightly-SNAPSHOT" ]]; then
  # Build debug version of litert, litert-gpu
  bazel build "${BUILD_FLAGS[@]}" \
      --define=tflite_keep_symbols=true \
      //tflite/java:tensorflow-lite \
      //tflite/java:tensorflow-lite-gpu
  prepare_pom_and_artifact "litert" "LiteRT implementation" \
      "bazel-bin/tflite/java/tensorflow-lite.aar" \
      "0.0.0-nightly-debug-SNAPSHOT" --depends-api
  prepare_pom_and_artifact "litert-gpu" "LiteRT GPU implementation" \
      "bazel-bin/tflite/java/tensorflow-lite-gpu.aar" \
      "0.0.0-nightly-debug-SNAPSHOT" --depends-api
fi

# No need to build select-tf-ops for now.
# bazel build "${BUILD_FLAGS[@]}" \
#     --config=monolithic --define=TENSORFLOW_PROTOS=lite \
#     --copt=-mno-sse4 --copt=-mno-sse4a --copt=-mno-sse4.1 --copt=-mno-sse4.2 \
#     //tflite/java:tensorflow-lite-select-tf-ops

# prepare_pom_and_artifact "litert-select-tf-ops" "LiteRT with selected TF Ops" \
#     "bazel-bin/tflite/java/tensorflow-lite-select-tf-ops.aar" \
#     "${VERSION}"
