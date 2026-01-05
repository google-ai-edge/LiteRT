# TFLite Python Wheel Package

This directory contains scripts and Bazel targets for building the TFLite Python
wheel package.

## Scripts

### `build_pip_package.sh`

This is the main script for building the TFLite Python wheel package. It uses
Bazel to build the required dependencies and then packages them into a `.whl`
file.

**Usage:**

```bash
./build_pip_package.sh
```

**Optional Environment Variables:**

*   `TFLITE_TARGET`: Specifies the target architecture for the build. Supported
    values are `armhf`, `aarch64`, `rpi0`, and `native`.
*   `NIGHTLY_RELEASE_DATE`: If set, the wheel will be built as a nightly release
    with the specified date in the version string (e.g., `YYYYMMDD`).
*   `USE_LOCAL_TF`: If set to `true`, the script will use the local TensorFlow
    source code from `third_party/tensorflow` instead of fetching it from the
    internet.
*   `TEST_MANYLINUX_COMPLIANCE`: If set to `true`, the script will run a test to
    ensure that the generated wheel is manylinux compliant.

### `build_pip_package_with_docker.sh`

This script builds the TFLite Python wheel package inside a Docker container.
This is useful for ensuring a consistent build environment and for building
manylinux compliant wheels.

**Usage:**

```bash
./build_pip_package_with_docker.sh
```

**Optional Environment Variables:**

*   `DOCKER_PYTHON_VERSION`: The Python version to use in the Docker container.
    Defaults to `3.10`.
*   All environment variables supported by `build_pip_package.sh` are also
    supported by this script.

## Bazel Targets

*   `//litert/tools/pip_package:litert_wheel`: The main Bazel target for
    building the TFLite wheel. This target is responsible for collecting all the
    necessary files and packaging them into a wheel.
*   `//litert/tools/pip_package:manylinux_compliance_test`: A test target
    that checks if the generated wheel is manylinux compliant.

For more details on the build process, please refer to the Bazel BUILD file in
this directory.
