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
- Run the container, mounting the current litert checkout directory
- Generate the configuration file (.tf_configure.bazelrc)
- Build a target. We use `//litert/runtime:metrics` as an example

## Building with Docker Compose

Alternatively, you can use Docker Compose:

```
docker-compose up
```

## Customizing the Build

To build different targets, you can either:

1. Modify the `hermetic_build.Dockerfile` and change the CMD line
2. Modify the command in `docker-compose.yml`
3. Pass a custom command when running Docker:
   ```
   docker run --rm --user $(id -u):$(id -g) -v $(pwd)/..:/litert_build litert_build_env bash -c "bazel build //litert/your_custom:target"
   ```

## How It Works

The Docker environment:
1. Sets up a Ubuntu 22.04 build environment (with newer libc/libc++)
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

You can run a shell in the container for debugging:
```
docker run --rm -it --user $(id -u):$(id -g) -e HOME=/litert_build -e USER=$(id -un) -v $(pwd)/..:/litert_build litert_build_env bash

```
