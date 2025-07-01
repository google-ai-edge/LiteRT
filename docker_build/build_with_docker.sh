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

# Change to the directory of this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running"
    exit 1
fi

echo "Building Docker image..."
docker build -t litert_build_env -f ./hermetic_build.Dockerfile .

if [ $? -ne 0 ]; then
  echo "Error: Docker build failed."
  exit 1
fi

CONTAINER_NAME="litert_build_container"

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Using existing container: ${CONTAINER_NAME}"
  echo "To remove it and start fresh, run: docker rm -f ${CONTAINER_NAME}"
  docker start -ai ${CONTAINER_NAME}
else
  echo "Running build in new Docker container..."
  docker run --name ${CONTAINER_NAME} --user $(id -u):$(id -g) -e HOME=/litert_build -e USER=$(id -un) -v $(pwd)/..:/litert_build litert_build_env
fi

if [ $? -ne 0 ]; then
  echo "Error: Build failed inside Docker container."
  exit 1
fi

echo "Build completed successfully!"
echo ""
echo "Container '${CONTAINER_NAME}' is preserved with all build outputs."
echo "You can:"
echo "  - Access build outputs: docker exec -it ${CONTAINER_NAME} bash"
echo "  - Copy files out: docker cp ${CONTAINER_NAME}:/litert_build/bazel-bin/<path> ."
echo "  - Or directly access the artifact from bazel-bin/ (or bazel-out)."
echo "  - Remove container: docker rm -f ${CONTAINER_NAME}"
