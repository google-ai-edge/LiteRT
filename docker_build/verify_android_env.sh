# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

# This script is designed to be executed inside the container to verify Android SDK/NDK setup
# You can run it outside of the container too if you want ot check your local environment setup

echo "Checking Android SDK/NDK environment..."
echo "----------------------------------------"

echo "ANDROID_SDK_HOME: $ANDROID_SDK_HOME"
echo "ANDROID_NDK_HOME: $ANDROID_NDK_HOME"
echo "ANDROID_NDK_VERSION: $ANDROID_NDK_VERSION"
echo "ANDROID_BUILD_TOOLS_VERSION: $ANDROID_BUILD_TOOLS_VERSION"
echo "ANDROID_SDK_API_LEVEL: $ANDROID_SDK_API_LEVEL"
echo "ANDROID_NDK_API_LEVEL: $ANDROID_NDK_API_LEVEL"
echo "----------------------------------------"

echo "Checking if Android SDK directories exist:"
if [ -d "$ANDROID_SDK_HOME" ]; then
  echo "SDK directory exists: $ANDROID_SDK_HOME"
  echo "SDK Contents:"
  ls -la $ANDROID_SDK_HOME
else
  echo "SDK directory NOT FOUND: $ANDROID_SDK_HOME"
fi

echo "----------------------------------------"
echo "Checking if Android NDK directories exist:"
if [ -d "$ANDROID_NDK_HOME" ]; then
  echo "NDK directory exists: $ANDROID_NDK_HOME"
  echo "NDK Contents:"
  ls -la $ANDROID_NDK_HOME
else
  echo "NDK directory NOT FOUND: $ANDROID_NDK_HOME"
fi

echo "----------------------------------------"
echo "Checking build tools:"
if [ -d "$ANDROID_SDK_HOME/build-tools/$ANDROID_BUILD_TOOLS_VERSION" ]; then
  echo "Build tools exist: $ANDROID_SDK_HOME/build-tools/$ANDROID_BUILD_TOOLS_VERSION"
else
  echo "Build tools NOT FOUND: $ANDROID_SDK_HOME/build-tools/$ANDROID_BUILD_TOOLS_VERSION"
fi

echo "----------------------------------------"
echo "Checking platforms:"
if [ -d "$ANDROID_SDK_HOME/platforms/android-$ANDROID_SDK_API_LEVEL" ]; then
  echo "Platform exists: $ANDROID_SDK_HOME/platforms/android-$ANDROID_SDK_API_LEVEL"
else
  echo "Platform NOT FOUND: $ANDROID_SDK_HOME/platforms/android-$ANDROID_SDK_API_LEVEL"
fi

echo "----------------------------------------"
echo "Checking .tf_configure.bazelrc:"
if [ -f ".tf_configure.bazelrc" ]; then
  echo ".tf_configure.bazelrc exists"
  echo "Contents:"
  cat .tf_configure.bazelrc
else
  echo ".tf_configure.bazelrc NOT FOUND"
fi

echo "----------------------------------------"
echo "Verification complete"
