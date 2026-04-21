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
- Generate the configuration file (.litert_configure.bazelrc)
- Build a target. We use `//litert/runtime:compiled_model` as an example

## Building the Python Wheel

To build the `ai-edge-litert` pip wheel and all vendor SDK packages:

```
cd docker_build
./build_wheel_with_docker.sh
```

This builds the main wheel (with Intel OpenVINO, Qualcomm, and MediaTek compiler
plugins bundled) plus vendor SDK sdist packages. Outputs are collected in
`dist/` at the repository root.

Options: - `--use_existing_image` — skip `docker build` and reuse the existing
image - `--dbg` — build in debug mode instead of optimized - `--python=3.12` —
set `HERMETIC_PYTHON_VERSION`

To verify the Intel OpenVINO plugin is bundled: `unzip -l dist/*.whl | grep
openvino`

## Building with Docker Compose

Alternatively, you can use Docker Compose:

```
docker-compose up
```

## Customizing the Build

The default `build_with_docker.sh` runs `run_build.sh` inside the container,
which builds `//litert/runtime:compiled_model`. To build different targets, edit
`run_build.sh` — it is copied into the Docker image at build time.

### Editing `run_build.sh`

`run_build.sh` sources `/setup_bazel_env.sh` which sets `EXTRA_STARTUP` (proxy
forwarding for Bazel's JVM downloader, SVE workaround for Apple Silicon). Use
`${EXTRA_STARTUP}` in every `bazel` command so proxy and platform flags are
applied automatically.

Default `run_build.sh`: `bash source /setup_bazel_env.sh bazel ${EXTRA_STARTUP}
build //litert/runtime:compiled_model`

#### Example: Intel OpenVINO compiler plugin, dispatch library, and benchmark tool

```bash
source /setup_bazel_env.sh
bazel ${EXTRA_STARTUP} build //litert/vendors/intel_openvino/compiler:libLiteRtCompilerPlugin_IntelOpenvino.so
bazel ${EXTRA_STARTUP} build //litert/vendors/intel_openvino/dispatch:dispatch_api_so
bazel ${EXTRA_STARTUP} build //litert/tools:benchmark_model
```

#### Example: All vendor compiler plugins

```bash
source /setup_bazel_env.sh
bazel ${EXTRA_STARTUP} build //litert/vendors/qualcomm/compiler:qualcomm_compiler_plugin_so
bazel ${EXTRA_STARTUP} build //litert/vendors/mediatek/compiler:mediatek_compiler_plugin_so
bazel ${EXTRA_STARTUP} build //litert/vendors/intel_openvino/compiler:libLiteRtCompilerPlugin_IntelOpenvino.so
```

After editing, rebuild the Docker image to pick up the changes (the file is
`COPY`'d during `docker build`): `bash ./build_with_docker.sh # rebuilds image +
runs build ./build_with_docker.sh --use_existing_image # skip image rebuild
(won't pick up run_build.sh changes)`

### Alternative: Override CMD without editing files

You can also pass a one-off command directly via `docker run`, without modifying
`run_build.sh`: ```bash

# From the repository root:

docker run --rm \
--security-opt seccomp=unconfined \
--user $(id -u):$(id -g) \
-e HOME=/litert_build -e USER=$(id -un) \
-e http_proxy="${http_proxy:-}" \
-e https_proxy="${https_proxy:-}" \
-v $(pwd):/litert_build \
litert_build_env \
bash -c 'source /setup_bazel_env.sh && bazel ${EXTRA_STARTUP} build
//litert/your_custom:target' ```

## Accessing Build Artifacts

Copy artifacts out of the container:
```
docker cp <container>:/litert_build/bazel-bin/<path> .
```
(`litert_build_container` is the name used by `build_with_docker.sh`. Use
`docker ps -a` to find the name for Docker Compose.)

To browse outputs from inside a container shell, run (from the repo root):
```
docker run --rm -it --user $(id -u):$(id -g) -e HOME=/litert_build -e USER=$(id -un) -v $(pwd):/litert_build litert_build_env bash
```

## How It Works

The Docker environment:
1. Sets up a Ubuntu 24.04 build environment (with newer libc/libc++)
2. Installs Bazel 7.4.1 and necessary build tools
3. Configures Android SDK and NDK with the correct versions
4. Automatically initializes and updates git submodules
5. Automatically generates the .litert_configure.bazelrc file
6. Provides a hermetic build environment independent of your local setup

## Proxy Configuration

If you are behind a corporate proxy, export the standard proxy environment
variables **before** running any Docker build script:

```bash
export http_proxy="http://proxy.example.com:123"
export https_proxy="http://proxy.example.com:123"
export no_proxy="localhost,127.0.0.1,.example.com"
```

The build scripts automatically forward these variables through three layers:

| Layer        | Mechanism                               | Tools affected     |
| ------------ | --------------------------------------- | ------------------ |
| Docker image | `--build-arg` in `docker build`         | `apt-get`, `wget`, |
: build        :                                         : `pip` during image :
:              :                                         : creation           :
| Container    | `-e` in `docker run`                    | `curl`, `pip`,     |
: runtime      :                                         : general network    :
:              :                                         : access             :
| Bazel JVM    | `--host_jvm_args=-Dhttps.proxyHost=...` | Bazel repository   |
: downloader   :                                         : fetches            :

Bazel's JVM-based downloader does **not** read `http_proxy`/`https_proxy`
natively. The scripts parse the proxy URL, extract host and port, and pass them
as Java system properties via `--host_jvm_args`.

When no proxy variables are set (or they are empty), all proxy code is a
complete no-op — no flags are added and connections go direct.

## Troubleshooting

If you encounter build errors:

1. Check that your Docker daemon has sufficient RAM and CPU allocated
2. Ensure you have proper permissions to mount the current directory
3. Check the Docker logs for any specific error messages

You can run a shell in the container for debugging (from the repo root):
```
docker run --rm -it --user $(id -u):$(id -g) -e HOME=/litert_build -e USER=$(id -un) -v $(pwd):/litert_build litert_build_env bash
```
