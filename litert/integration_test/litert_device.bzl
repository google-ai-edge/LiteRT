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

"""
This module defines the `run_on_device` macro, which helps to execute a binary target on a device.
"""

# copybara:uncomment_begin(google-only)
# load("//research/aimatter/testing:xeno_lab.bzl", "xeno_mobile_test")
# 
# copybara:uncomment_end(google-only)
load("//litert/build_common:litert_build_defs.bzl", "absolute_label")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "if_oss")

# MISCELLANEOUS ####################################################################################

def hidden_test_tags():
    """
    Get tags to disable a test that is not expected to work on Forge.

    Returns:
        A list of tags to hide a test from various tools.
    """
    return [
        "no-remote-exec",
        "manual",
        "notap",
        "nobuilder",
        "no_oss",
    ]

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
    res = DEVICE_RLOCATION_ROOT + "/" + abs_label.replace("//", "").replace(":", "/")
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

# QUALCOMM

def BackendSpec(id, libs = [], mh_devices = [], dispatch = None, plugin = None):
    """
    Defines a backend specification.

    Args:
        id: The backend id.
        libs: A list of tuples of (library target, environment variable). Path to the target
            will be added to the environment variable.
        mh_devices: A list of mobile harness device specifications.
        dispatch: The dispatch library target name.
        plugin: The compiler plugin library target name.

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
    )

QUALCOMM_SPEC = BackendSpec(
    id = "qualcomm",
    libs = [
        ("//third_party/qairt/latest:lib/aarch64-android/libQnnHtp.so", "LD_LIBRARY_PATH"),
        ("//third_party/qairt/latest:lib/aarch64-android/libQnnHtpV75Stub.so", "LD_LIBRARY_PATH"),
        ("//third_party/qairt/latest:lib/aarch64-android/libQnnSystem.so", "LD_LIBRARY_PATH"),
        ("//third_party/qairt/latest:lib/aarch64-android/libQnnHtpPrepare.so", "LD_LIBRARY_PATH"),
        ("//third_party/qairt/latest:lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so", "ADSP_LIBRARY_PATH"),
        ("//litert/vendors/qualcomm/dispatch:libLiteRtDispatch_Qualcomm.so", "LD_LIBRARY_PATH"),
        ("//litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so", "LD_LIBRARY_PATH"),
    ],
    mh_devices = [{
        "model": "sm-s928b",
    }],
    plugin = "qnn_compiler_plugin_so",
    dispatch = "libLiteRtDispatch_Qualcomm.so",
)

# MEDIATEK

MEDIATEK_SPEC = BackendSpec(
    id = "mediatek",
    libs = [
        ("//litert/vendors/mediatek/dispatch:libLiteRtDispatch_Mtk.so", "LD_LIBRARY_PATH"),
    ],
    mh_devices = [{
        "hardware": "mt6989",
    }],
    dispatch = "libLiteRtDispatch_Mtk.so",
)

# GOOGLE TENSOR

GOOGLE_TENSOR_SPEC = BackendSpec(
    id = "google_tensor",
    libs = [
        ("//litert/vendors/google_tensor/dispatch:libLiteRtDispatch_GoogleTensor.so", "LD_LIBRARY_PATH"),
    ],
    mh_devices = [{
        "label": "odml-test",
        "model": "pixel 9",
    }],
    dispatch = "libLiteRtDispatch_GoogleTensor.so",
)

# CPU

CPU_SPEC = BackendSpec(
    id = "cpu",
)

# GPU

GPU_SPEC = BackendSpec(
    id = "gpu",
)

# COMMON

_SPECS = {
    QUALCOMM_SPEC.id: QUALCOMM_SPEC,
    GOOGLE_TENSOR_SPEC.id: GOOGLE_TENSOR_SPEC,
    MEDIATEK_SPEC.id: MEDIATEK_SPEC,
    CPU_SPEC.id: CPU_SPEC,
    GPU_SPEC.id: GPU_SPEC,
}

# RUN ON DEVICE MACRO ##############################################################################

def get_driver():
    return if_oss(
        "//litert/integration_test:run_on_device_driver_OSS",
        "//litert/integration_test/google:run_on_device_driver",
    )

def litert_device_exec(
        name,
        target,
        backend_id = "cpu",
        driver = get_driver(),
        data = [],
        exec_args = [],
        exec_env_vars = []):
    """
    Macro to execute a binary target on a device.

    #copybara:comment_begin(google-only)
    Note: Allows running the target on a locally connected device through ADB or via Mobile Harness
    (see xeno_mobile_test target below).
    #copybara:comment_end(google-only)

    The output of this macro is an executable shell script that pushes all the necessary files to
    the device and executes the target with the given arguments and environment variables.

    Args:
        name: Name of the target.
        backend_id: The backend id to use for the test (e.g. QUALCOMM_ID, GOOGLE_TENSOR_ID).
        target: The binary target to execute on device.
        driver: The driver script to use for execution.
        data: List of data files to push to the device.
        exec_args: List of arguments to pass to the executable.
        exec_env_vars: List of environment variables to set before executing the target.
    """
    data = data + []
    exec_env_vars = exec_env_vars + []

    backend = _SPECS[backend_id]

    data.extend(backend.libs)
    exec_env_vars.extend(backend.env_paths)

    call_mobile_install = """
    echo '$(location {driver}) \
        --bin=$(rlocationpath {target}) \
        --data={data} \
        --do_exec=true \
        --exec_args={exec_args} \
        --exec_env_vars={exec_env_vars} \
        '\
        > $@
    """

    concat_targ_data = "$$(echo \"$(rlocationpaths {})\" | sed \"s/ /,/g\")"
    data_str = ",".join([concat_targ_data.format(d) for d in data])

    # NOTE: Tilde delimiter here (also see driver script) to allow passing list args to underlying
    # binary.
    exec_args_str = "~".join(["{}".format(a) for a in exec_args])
    exec_env_vars_str = ",".join(["{}".format(a) for a in exec_env_vars])

    driver_targ = driver.removesuffix(".sh")
    driver_sh = driver_targ + ".sh"

    cmd = call_mobile_install.format(
        driver = driver_sh,
        target = target,
        data = data_str,
        exec_args = exec_args_str,
        exec_env_vars = exec_env_vars_str,
    )

    exec_script = name + "_exec.sh"

    native.genrule(
        name = name + "_gen_script",
        srcs = [driver_sh] + [target] + data,
        outs = [exec_script],
        tags = ["manual", "notap"],
        cmd = cmd,
        testonly = True,
    )

    native.sh_binary(
        testonly = True,
        tags = ["manual", "notap"],
        name = name,
        deps = [driver_targ],
        srcs = [exec_script],
        data = [target] + data,
    )

    # copybara:uncomment_begin(google-only)
    # xeno_mobile_test(
        # name = name + "_lab_test",
        # test_target = target,
        # args = [
            # "--run_as=odml-device-lab",
        # ],
        # dimensions = backend.default_mh_device,
        # tags = [
            # "android",
            # "external",
            # "guitar",
            # "manual",
            # "notap",
        # ],
        # test_data = data,
        # run_as_top_app = False,
        # env_vars = exec_env_vars,
    # )
    # copybara:uncomment_end(google-only)

def litert_device_test(
        name,
        srcs,
        deps,
        rule = native.cc_test,
        backend_id = "",
        driver = get_driver(),
        data = [],
        exec_args = [],
        exec_env_vars = [],
        tags = [],
        linkopts = []):
    """
    Syntactic sugar for the litert_device_exec macro.

    Creates a target to run internally given the srcs and deps (default cc_test).

    Args:
        name: Name of the target.
        srcs: The source files for the target to be generated.
        deps: The dependencies for the target to be generated.
        rule: The rule to use for the target to be generated.
        backend_id: The backend id to use for the test (e.g. QUALCOMM_ID, GOOGLE_TENSOR_ID).
        driver: The driver script to use for execution.
        data: List of data files to push to the device and for the target to be generated.
        exec_args: List of arguments to pass to the executable.
        exec_env_vars: List of environment variables to set before executing the target.
        tags: List of tags to apply to the target to be generated.
        linkopts: List of linkopts to apply to the target to be generated.
    """

    target = name + "_{}".format(name)

    rule(
        name = target,
        srcs = srcs,
        deps = deps,
        data = data,
        linkopts = select({
            "@org_tensorflow//tensorflow:android": ["-landroid"],
            "//conditions:default": [],
        }) + linkopts,
        tags = hidden_test_tags() + tags,
    )

    litert_device_exec(
        name = name,
        target = absolute_label(":{}".format(target)),
        backend_id = backend_id,
        driver = driver,
        data = data,
        exec_args = exec_args,
        exec_env_vars = exec_env_vars,
    )

def litert_integration_test(
        name,
        models,
        backend_id = "cpu",
        skips = []):
    """
    Higher level macro that configures run_on_device or a mobile test to run with gen_device_test.

    Args:
        name: Name of the target.
        models: A single target that may contain model or many models in the same directory.
        backend_id: The backend to test against (see gen_device_test).
        skips: List of substrings of models to skip.
    """

    backend = _SPECS[backend_id]

    req_hardware = backend.id != "cpu" and backend.id != "gpu"

    # Accelerator option to pass to the compiled model api on device.
    hw_cfg = "npu" if req_hardware else backend.id

    skips_str = ",".join(skips)

    # Create CLI args for the gen_device_test binary on device.
    cli_args = [
        "--model_path={}".format(device_rlocation(models)),
        "--hw={}".format(hw_cfg),
        "--skips={}".format(skips_str),
    ]
    if backend.dispatch:
        cli_args.append("--dispatch_library_dir={}".format(device_rlocation(backend.dispatch, True)))
    if backend.plugin:
        cli_args.append("--compiler_library_dir={}".format(device_rlocation(backend.plugin, True)))

    data = [models]
    driver = get_driver()
    target = "//litert/integration_test:gen_device_test"

    litert_device_exec(
        name = name,
        target = target,
        driver = driver,
        backend_id = backend_id,
        data = data,
        exec_args = cli_args,
    )
