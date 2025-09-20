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
# load("//devtools/build_cleaner/skylark:build_defs.bzl", "register_extension_info")
# load("//devtools/deviceinfra/api/builddefs/test:mobile_test.bzl", "mobile_test")
# load(INTERNAL_PHYSICAL_MOBILE_TESTING_INFRA, "guitar")
#
# copybara:uncomment_end
load("//litert/build_common:litert_build_defs.bzl", "absolute_label", "if_oss")

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
        "local",
    ]

# DEVICE PATHS #####################################################################################

def _strip_double_prefix():
    return "perl -ne \"s/(^|\\s)litert\\/litert/\\1litert/g; print;\""

def _add_external_prefix():
    return "perl -ne \"s/(^|\\s)(?!litert)([^\\s]+)/\\1external\\/\\2/g; print;\""

def _change_delim(before, after):
    return "sed \"s/{before}/{after}/g\"".format(before = before, after = after)

def _transform_str(s, *cmds):
    if not cmds:
        return s
    cmds_ = ["echo {}".format(s)]
    cmds_.extend(*cmds)
    cmds_str = " | ".join(cmds_)
    return "$$({})".format(cmds_str)

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

# QUALCOMM

def BackendSpec(id, libs = [], mh_devices = [], dispatch = None, plugin = None, mh_user = "odml-device-lab"):
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
    )

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
        ),
    }

# MEDIATEK

def _MediatekSpec():
    return {
        "mediatek": BackendSpec(
            id = "mediatek",
            libs = [
                ("//litert/vendors/mediatek/dispatch:libLiteRtDispatch_Mediatek.so", "LD_LIBRARY_PATH"),
                ("//litert/vendors/mediatek/compiler:libLiteRtCompilerPlugin_MediaTek.so", "LD_LIBRARY_PATH"),
            ],
            mh_devices = [{
                "hardware": "mt6989",
                "label": "odml-test",
            }],
            dispatch = "libLiteRtDispatch_Mediatek.so",
            plugin = "libLiteRtCompilerPlugin_MediaTek.so",
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

# copybara:uncomment_begin(google-only)
# # MOBILE HARNESS WRAPPER ###########################################################################
#
# def _litert_mh_exec(
#         name,
#         target,
#         run_as = "odml-device-lab",
#         data = [],
#         exec_args = [],
#         exec_env_vars = [],
#         dimensions = {}):
#     """Wraps the mobile harness "mobile_test" macro"""
#
#     files = {
#         "bin": [target],
#     }
#
#     push_files_list = []
#
#     for data_target in data:
#         data_target_split = data_target.split(":")
#         if len(data_target_split) != 2:
#             # This is required by `mobile_test` for value in `files`.
#             fail("Data inputs must include a colon even if relative label")
#         data_id = data_target_split[-1]
#         files[data_id] = [data_target]
#
#         # NOTE: Any non raw file data targets must output a directory with the same name as the
#         # target (e.g. filegroup "foo" that globs "foo/*"). This may need to be updated later
#         # but it works for everything in LiteRt for now.
#         push_files_list.append("{id}:{loc}".format(
#             id = data_id,
#             loc = device_rlocation(data_target, get_parent = False),
#         ))
#
#     params = {
#         "run_dir": device_rlocation(),
#         "options": " ".join(exec_args),
#         "prepare_des_dir_when_src_is_file": "true",  # Allows dest to be a dir (parent) when src is a file.
#         "run_env": " ".join(exec_env_vars),
#         "remove_files_before_push": "true",
#     }
#
#     if push_files_list:
#         params["push_files"] = ",".join(push_files_list)
#
#     args = [
#         "--run_as={}".format(run_as),
#     ]
#
#     mobile_test(
#         tags = hidden_test_tags(),
#         name = name,
#         dimensions = dimensions,
#         files = files,
#         params = params,
#         args = args,
#         device = "AndroidRealDevice",
#         driver = "AndroidNativeBin",
#         decorators = [
#             "AndroidFilePusherDecorator",
#         ],
#         visibility = [
#             "//litert/integration_test:__subpackages__",
#             "//litert/google:__subpackages__",
#         ],
#     )
#
# copybara:uncomment_end

# RUN ON DEVICE MACRO ##############################################################################

# Public facing functions to get lib locations from a backend id. Can be used in flag creation.
def dispatch_device_rlocation(backend_id):
    spec = _Specs(backend_id)
    return device_rlocation(spec.dispatch, True)

# Public facing functions to get lib locations from a backend id. Can be used in flag creation.
def plugin_device_rlocation(backend_id):
    spec = _Specs(backend_id)
    return device_rlocation(spec.plugin, True)

def get_driver():
    return if_oss(
        "//litert/integration_test:run_on_device_driver",
        "//litert/integration_test:run_on_device_driver",
    )

def litert_device_exec(
        name,
        target,
        backend_id = "cpu",
        driver = get_driver(),
        data = [],
        exec_args = [],
        exec_env_vars = [],
        remote_suffix = "",
        local_suffix = "_adb"):
    """
    Macro to execute a binary target on a device.

    #copybara:comment_begin(google-only)
    Note: Allows running the target on a locally connected device through ADB or via Mobile Harness
    (see litert_mh_exec target).
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
        remote_suffix: Suffix for the target runnin on device cloud if enabled.
        local_suffix: Suffix for the target that runs locally on physical device through adb.
    """
    data = data + []
    exec_env_vars = exec_env_vars + []

    backend = _Specs(backend_id)

    data.extend(backend.libs)
    exec_env_vars.extend(backend.env_paths)

    path_transforms = if_oss([_strip_double_prefix(), _add_external_prefix()])

    bin_path_str = _transform_str("$(rlocationpath {})".format(target), path_transforms)

    call_mobile_install = """
    echo '$(location {driver}) \
        --bin={bin_path_str} \
        --data={data} \
        --do_exec=true \
        --exec_args=\"{exec_args}\" \
        --exec_env_vars={exec_env_vars} \
        -- "$$@" \
        '\
        > $@
    """

    data_str = ",".join([_transform_str("$(rlocationpaths {})".format(d), path_transforms + [_change_delim(" ", ",")]) for d in data])

    # NOTE: Tilde delimiter here (also see driver script) to allow passing list args to underlying
    # binary.
    exec_args_str = "~".join(["{}".format(a) for a in exec_args])
    exec_env_vars_str = ",".join(["{}".format(a) for a in exec_env_vars])

    driver_targ = driver.removesuffix(".sh")
    driver_sh = driver_targ + ".sh"

    cmd = call_mobile_install.format(
        driver = driver_sh,
        bin_path_str = bin_path_str,
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
        name = name + local_suffix,
        deps = [driver_targ],
        srcs = [exec_script],
        data = [target] + data,
    )

    # copybara:uncomment_begin(google-only)
    # _litert_mh_exec(
    # name = name + remote_suffix,
    # target = target,
    # run_as = backend.mh_user,
    # data = data,
    # exec_args = exec_args,
    # exec_env_vars = exec_env_vars,
    # dimensions = backend.default_mh_device,
    # )
    # copybara:uncomment_end(google-only)

def litert_device_test(
        name,
        srcs,
        deps,
        features = [],
        rule = native.cc_test,
        backend_id = "",
        driver = get_driver(),
        data = [],
        exec_args = [],
        exec_env_vars = [],
        tags = [],
        linkopts = [],
        copts = [],
        **kwargs):
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
        copts: List of copts to apply to the target to be generated.
    """

    target = "_{}".format(name)

    rule(
        name = target,
        srcs = srcs,
        deps = deps,
        features = features,
        data = data,
        linkopts = select({
            "@org_tensorflow//tensorflow:android": ["-landroid"],
            "//conditions:default": [],
        }) + linkopts,
        copts = copts,
        tags = hidden_test_tags() + tags,
        **kwargs
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

# copybara:uncomment_begin(google-only)
# register_extension_info(
#     extension = litert_device_test,
#     label_regex_map = {
#         "deps": "deps:_{extension_name}",
#     },
# )
# copybara:uncomment_end

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

    backend = _Specs(backend_id)

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

# copybara:uncomment_begin(google-only)
# # GUITAR UTIL ######################################################################################
#
# def litert_pixel_9_mh_guitar_test(targets, dimension_model = "\"pixel 9\""):
#     return guitar.Tests(
#         args = [
#             "--allocation_exit_strategy=FAIL_FAST_NO_MATCH",
#             "--dimension_model={}".format(dimension_model),
#             "--dimension_build_type=userdebug",
#             "--dimension_label=odml-test",
#             "--run_as=xeno-mh-guitar",
#         ],
#         bazel_flags = [
#             "--config=android_arm64",
#             "--copt=-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1",
#             "--android_ndk_min_sdk_version=26",
#         ],
#         execution_method = "DISTRIBUTED_ON_BORG",
#         targets = targets,
#     )
#
# def litert_qualcomm_mh_guitar_test(targets):
#     return guitar.Tests(
#         args = [
#             "--allocation_exit_strategy=FAIL_FAST_NO_MATCH",
#             "--dimension_pool=shared",
#             "--dimension_model=sm-s928u1",
#             "--run_as=xeno-mh-guitar",
#         ],
#         bazel_flags = [
#             "--config=android_arm64",
#             "--copt=-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1",
#             "--android_ndk_min_sdk_version=26",
#         ],
#         execution_method = "DISTRIBUTED_ON_BORG",
#         targets = targets,
#     )
#
# def litert_mediatek_mh_guitar_test(targets):
#     return guitar.Tests(
#         args = [
#             "--allocation_exit_strategy=FAIL_FAST_NO_MATCH",
#             "--dimension_label=odml-test",
#             "--dimension_hardware=mt6989",
#             "--run_as=xeno-mh-guitar",
#         ],
#         bazel_flags = [
#             "--config=android_arm64",
#             "--copt=-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1",
#             "--android_ndk_min_sdk_version=26",
#         ],
#         execution_method = "DISTRIBUTED_ON_BORG",
#         targets = targets,
#     )
#
# def litert_cpu_mh_guitar_test(targets):
#     return guitar.Tests(
#         args = [
#             "--allocation_exit_strategy=FAIL_FAST_NO_MATCH",
#             "--dimension_label=odml-test",
#             "--run_as=xeno-mh-guitar",
#         ],
#         bazel_flags = [
#             "--config=android_arm64",
#             "--copt=-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1",
#             "--android_ndk_min_sdk_version=26",
#         ],
#         execution_method = "DISTRIBUTED_ON_BORG",
#         targets = targets,
#     )
#
# copybara:uncomment_end
