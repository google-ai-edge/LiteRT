# Building with Docker

This repository provides a Docker-based hermetic build environment that
automatically configures all necessary dependencies for building the project
without requiring manual configuration or setup. It handles both git submodule
initialization and project configuration in a single step.

## Prerequisites

- Docker installed on your machine
- Docker Compose (optional, for using docker-compose.yml)

## Building with the Docker Script

1. Clone this repository
2. Run the build script:
   ```
   ./build_with_docker.sh
   ```

This will:

- Build a Docker image with all necessary dependencies
- Create a persistent container named `litert_build_container`
- Mount the current litert checkout directory
- Generate the configuration file (.tf_configure.bazelrc)
- Build a target. We use `//litert/runtime:metrics` as an example
- Keep the container running so you can access build outputs

## Building with Docker Compose

Alternatively, you can use Docker Compose:

```bash
# Set user IDs for proper permissions
export UID=$(id -u)
export GID=$(id -g)

# Run the build
docker-compose up

# Access the container afterwards
docker exec -it litert_build_container bash
```

## Working with Build Outputs

The Docker container is persistent (not removed after builds), so all build outputs and symlinks remain valid. You can access them in several ways:

### Access the container shell
```bash
# Enter the container to browse outputs
docker exec -it litert_build_container bash

# Navigate to build outputs
cd bazel-bin/
```

### Copy files from the container
```bash
# Copy a specific file
docker cp litert_build_container:/litert_build/bazel-bin/litert/tools/run_model ./my_output

# Copy an entire directory
docker cp litert_build_container:/litert_build/bazel-bin/litert/tools/ ./tools_output/
```

### Container Management
```bash
# Check if container exists
docker ps -a | grep litert_build_container

# Remove the container to start fresh
docker rm -f litert_build_container

# Re-run build_with_docker.sh to create a new container
./build_with_docker.sh
```

## Customizing the Build

To build different targets, you can either:

1. Execute commands in the existing container:
   ```bash
   docker exec -it litert_build_container bazel build //litert/your_custom:target
   ```

2. Modify the `hermetic_build.Dockerfile` and change the CMD line

3. Modify the command in `docker-compose.yml`

4. Run Docker directly with a custom command:
   ```bash
   docker run --name my_custom_container \
     --user $(id -u):$(id -g) \
     -e HOME=/litert_build \
     -e USER=$(id -un) \
     -v $(pwd)/..:/litert_build \
     litert_build_env \
     bash -c "bazel build //litert/your_custom:target"
   ```

## How It Works

The Docker environment:
1. Sets up a Ubuntu 24.04 build environment (with newer libc/libc++)
2. Installs Bazel 7.4.1 and necessary build tools
3. Configures Android SDK and NDK with the correct versions
4. Automatically initializes and updates git submodules
5. Automatically generates the .tf_configure.bazelrc file
6. Provides a hermetic build environment independent of your local setup

## Troubleshooting

If you encounter build errors:

1. Check that your Docker daemon has sufficient RAM and CPU allocated
2. Ensure you have proper permissions to mount the current directory
3. Check the Docker logs for any specific error messages

You can access the container for debugging:
```bash
# If container exists from a previous build
docker exec -it litert_build_container bash

# Or create a new interactive container
docker run -it --name debug_container \
  --user $(id -u):$(id -g) \
  -e HOME=/litert_build \
  -e USER=$(id -un) \
  -v $(pwd)/..:/litert_build \
  litert_build_env bash
```
