# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for generating device scripts with starlark."""

load("@rules_platform//platform_data:defs.bzl", "platform_data")
load("//litert/build_common:litert_build_defs.bzl", "absolute_label")

# DEVICE PATHS #####################################################################################

DEVICE_RLOCATION_ROOT = "/data/local/tmp/runfiles"

def device_rlocation(label = None, get_parent = False):
    """Get the path on device for a given label.

    Args:
        label: The label to get the path for. If None, returns the root path.
        get_parent: If true, get the parent directory of the resolved path.

    Returns:
        The path on device for the given label.
    """
    if not label:
        return DEVICE_RLOCATION_ROOT
    abs_label = absolute_label(label)
    res = DEVICE_RLOCATION_ROOT + "/" + abs_label.replace("@", "external/").replace("//", "").replace(":", "/")
    if get_parent:
        return res[:res.rfind("/")]
    return res

def make_path_args(spec):
    """Formats shell path-like variable assignment exprs from common directories in given labels

    Useful for making things like LD_LIBRARY_PATH=... for paths on device.

    An entry of the spec contains a key, and a list of labels. Unique leaf directories paths are
    extracted from the labels and joined into a colon-separated string.

    Example:
    ```
    make_path_args({
        "LD_LIBRARY_PATH": [
            "// foo : bar",
        ],
        "ADSP_LIBRARY_PATH": [
            "// foo : baz",
            "// foo : bat"
        ],
    })
    ```
    will return:
    ```
    LD_LIBRARY_PATH=/data/local/tmp/runfiles/foo/bar
    ADSP_LIBRARY_PATH=/data/local/tmp/runfiles/foo/baz:/data/local/tmp/runfiles/foo/bat
    ```

    Args:
        spec: A dict of path variable names to lists of labels.

    Returns:
        A list of shell variable assignment expressions.
    """

    res = []
    for path_var, values in spec.items():
        # TODO: Figure out why OSS doesn't have `set` core datatype.
        dirs = []
        for v in values:
            parent = device_rlocation(v, True)
            if parent not in dirs:
                dirs.append(parent)
        res.append("{path_var}={paths}".format(
            path_var = path_var,
            paths = ":".join(dirs),
        ))
    return res

# DYNAMIC LIBRARY DEPENDENCIES #####################################################################

# COMMON

def BackendSpec(id, libs = [], mh_devices = [], dispatch = None, plugin = None, mh_user = "odml-device-lab", host_libs = []):
    """
    Defines a backend specification.

    Args:
        id: The backend id.
        libs: A list of tuples of (library target, environment variable). Path to the target
            will be added to the environment variable.
        mh_devices: A list of mobile harness device specifications.
        dispatch: The dispatch library target name.
        plugin: The compiler plugin library target name.
        mh_user: The "run_as" arg to use in device cloud if it is enabled.
        host_libs: A list of pre-built libraries for the host platform.

    Returns:
        A struct representing the backend specification.
    """

    libs = libs + [
        ("//litert/c:libLiteRtRuntimeCApi.so", "LD_LIBRARY_PATH"),
    ]
    libs_agg = []
    env_paths = {}
    for lib in libs:
        lib_targ = lib[0]
        libs_agg.append(lib_targ)
        if dispatch and lib_targ.endswith(dispatch):
            dispatch = lib_targ
        if plugin and lib_targ.endswith(plugin):
            plugin = lib_targ
        paths = []
        if len(lib) > 1:
            paths = lib[1]
        if "append" not in dir(paths):
            paths = [paths]
        for p in paths:
            if p not in env_paths:
                env_paths[p] = []
            env_paths[p].append(lib_targ)
    if not mh_devices:
        mh_devices = [{}]
    return struct(
        id = id,
        libs = libs_agg,
        env_paths = make_path_args(env_paths),
        mh_devices = mh_devices,
        default_mh_device = mh_devices[0],
        dispatch = dispatch,
        plugin = plugin,
        mh_user = mh_user,
        host_libs = host_libs,
    )

# QUALCOMM

def _QualcommSpec():
    return {
        "qualcomm": BackendSpec(
            id = "qualcomm",
            libs = [
                ("@qairt//:lib/aarch64-android/libQnnHtp.so", "LD_LIBRARY_PATH"),
                ("@qairt//:lib/aarch64-android/libQnnHtpV75Stub.so", "LD_LIBRARY_PATH"),
                ("@qairt//:lib/aarch64-android/libQnnSystem.so", "LD_LIBRARY_PATH"),
                ("@qairt//:lib/aarch64-android/libQnnHtpPrepare.so", "LD_LIBRARY_PATH"),
                ("@qairt//:lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so", "ADSP_LIBRARY_PATH"),
                ("//litert/vendors/qualcomm/dispatch:libLiteRtDispatch_Qualcomm.so", "LD_LIBRARY_PATH"),
                ("//litert/vendors/qualcomm/compiler:libLiteRtCompilerPlugin_Qualcomm.so", "LD_LIBRARY_PATH"),
            ],
            mh_devices = [{
                "model": "regex:sm-s928b|sm-s928u1",
                "pool": "shared",
            }],
            plugin = "libLiteRtCompilerPlugin_Qualcomm.so",
            dispatch = "libLiteRtDispatch_Qualcomm.so",
            host_libs = [
                "@qairt//:lib/x86_64-linux-clang/libQnnHtp.so",
                "@qairt//:lib/x86_64-linux-clang/libQnnSystem.so",
            ],
        ),
    }

# MEDIATEK

def _MediatekSpec():
    return {
        "mediatek": BackendSpec(
            id = "mediatek",
            libs = [
                ("//litert/vendors/mediatek/dispatch:libLiteRtDispatch_MediaTek.so", "LD_LIBRARY_PATH"),
                ("//litert/vendors/mediatek/compiler:libLiteRtCompilerPlugin_MediaTek.so", "LD_LIBRARY_PATH"),
            ],
            mh_devices = [{
                "hardware": "mt6989",
                "label": "odml-test",
            }],
            dispatch = "libLiteRtDispatch_MediaTek.so",
            plugin = "libLiteRtCompilerPlugin_MediaTek.so",
            host_libs = ["@neuro_pilot//:v8_latest/host/lib/libneuron_adapter.so"],
        ),
    }

# GOOGLE TENSOR

def _GoogleTensorSpec():
    return {
        "google_tensor": BackendSpec(
            id = "google_tensor",
            libs = [
                ("//litert/vendors/google_tensor/dispatch:libLiteRtDispatch_GoogleTensor.so", "LD_LIBRARY_PATH"),
            ],
            mh_devices = [{
                "label": "odml-test",
                "model": "pixel 9",
            }],
            mh_user = "odml-team",
            dispatch = "libLiteRtDispatch_GoogleTensor.so",
        ),
    }

# EXAMPLE

def _ExampleSpec():
    return {
        "example": BackendSpec(
            id = "example",
            libs = [
                ("//litert/vendors/examples:libLiteRtDispatch_Example.so", "LD_LIBRARY_PATH"),
                ("//litert/vendors/examples:libLiteRtCompilerPlugin_Example.so", "LD_LIBRARY_PATH"),
            ],
            mh_devices = [{
                "pool": "shared",
            }],
            plugin = "libLiteRtCompilerPlugin_Example.so",
            dispatch = "libLiteRtDispatch_Example.so",
        ),
    }

# CPU

def _CpuSpec():
    return {
        "cpu": BackendSpec(
            id = "cpu",
        ),
    }

# GPU

def _GpuSpec():
    return {
        "gpu": BackendSpec(
            id = "gpu",
        ),
    }

# COMMON

def _Specs(name):
    return (_QualcommSpec() | _GoogleTensorSpec() | _MediatekSpec() | _CpuSpec() | _GpuSpec() | _ExampleSpec())[name]

# Check if the backend maps to an NPU backend.
def is_npu_backend(name):
    return name in ["qualcomm", "mediatek", "google_tensor", "example"]

# Get the libs for the given backend.
def get_libs(name):
    return _Specs(name).libs

# Get the spec for the given backend.
def get_spec(name):
    return _Specs(name)

# Get the host libs for the given backend.
def get_host_libs(name):
    return _Specs(name).host_libs

# Public facing functions to get lib locations from a backend id. Can be used in flag creation.
def dispatch_device_rlocation(backend_id):
    spec = _Specs(backend_id)
    return device_rlocation(spec.dispatch, True)

# Public facing functions to get lib locations from a backend id. Can be used in flag creation.
def plugin_device_rlocation(backend_id):
    spec = _Specs(backend_id)
    return device_rlocation(spec.plugin, True)

# buildifier: disable=unnamed-macro
def split_dep_platform(
        prefix,
        targets,
        testonly,
        host_suffix = "_for_host",
        device_suffix = "_for_device",
        no_host = False,
        no_device = True):
    """Forks the configuration of "targets" into builds for both the host and device.

    This function creates `platform_data` targets for both host (linux) and device (android)
    platforms, agnostic to the top-level config. For example, it can be used to build a
    `.so` library for both android & linux.

    Args:
        prefix: A prefix for the generated target names.
        targets: A single label or a list of labels to build for both platforms.
        testonly: Whether the generated targets are testonly.
        host_suffix: Suffix to append to host target names.
        device_suffix: Suffix to append to device target names.
        no_host: If True, do not generate host targets.
        no_device: If True, do not generate device targets.

    Returns:
        A struct with two fields:
        - host: A list of labels for the generated host targets.
        - device: A list of labels for the generated device targets.
    """
    if "append" not in dir(targets):
        targets = [targets]

    host = []
    device = []

    for targ in targets:
        lab = Label(targ)
        host_name = prefix + "_" + lab.name + host_suffix
        device_name = prefix + "_" + lab.name + device_suffix

        if not no_host:
            platform_data(
                name = host_name,
                target = targ,
                platform = "//litert/integration_test:host",
                testonly = testonly,
            )
            host.append(":" + host_name)

        if not no_device:
            platform_data(
                name = device_name,
                target = targ,
                platform = "//litert/integration_test:device",
                testonly = testonly,
            )
            device.append(":" + device_name)

    return struct(host = host, device = device)
