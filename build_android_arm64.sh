#!/bin/bash

# Build script for LiteRT with Android ARM64 configuration
# This replicates the Bazel command: bazel build --config=android_arm64 //litert/tools:run_model

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}LiteRT Android ARM64 Build Script${NC}"
echo "=================================================="

# Check for Android NDK
if [ -z "$ANDROID_NDK_ROOT" ] && [ -z "$ANDROID_NDK" ]; then
    echo -e "${RED}ERROR: Android NDK not found.${NC}"
    echo "Please set ANDROID_NDK_ROOT or ANDROID_NDK environment variable."
    echo "Example: export ANDROID_NDK_ROOT=/path/to/android-ndk"
    exit 1
fi

# Set NDK path
if [ -n "$ANDROID_NDK_ROOT" ]; then
    export ANDROID_NDK="$ANDROID_NDK_ROOT"
elif [ -n "$ANDROID_NDK" ]; then
    export ANDROID_NDK_ROOT="$ANDROID_NDK"
fi

echo -e "${YELLOW}Using Android NDK:${NC} $ANDROID_NDK"

# Build directory
BUILD_DIR="build_android_arm64"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${YELLOW}Source directory:${NC} $SOURCE_DIR"
echo -e "${YELLOW}Build directory:${NC} $BUILD_DIR"

# Clean and create build directory
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Cleaning existing build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake for Android ARM64
echo -e "${GREEN}Configuring CMake for Android ARM64...${NC}"
cmake \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-21 \
    -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release \
    -DLITERT_BUILD_TOOLS=ON \
    -DLITERT_BUILD_TESTS=OFF \
    -DLITERT_ENABLE_GPU=ON \
    -DLITERT_ENABLE_QUALCOMM=OFF \
    -DLITERT_ENABLE_MEDIATEK=OFF \
    -DLITERT_ENABLE_GOOGLE_TENSOR=OFF \
    "$SOURCE_DIR"

# Build the run_model target
echo -e "${GREEN}Building run_model target...${NC}"
cmake --build . --target run_model --parallel $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Check if build was successful
if [ -f "litert/tools/run_model" ]; then
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo -e "${GREEN}✓ run_model executable created at:${NC} $BUILD_DIR/litert/tools/run_model"
    
    # Get file info
    echo -e "${YELLOW}File information:${NC}"
    ls -la litert/tools/run_model
    file litert/tools/run_model || echo "file command not available"
    
    echo ""
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo "The equivalent of 'bazel build --config=android_arm64 //litert/tools:run_model' has been built."
else
    echo -e "${RED}✗ Build failed - run_model executable not found${NC}"
    exit 1
fi