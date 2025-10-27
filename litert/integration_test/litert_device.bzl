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
This module defines various macros which helps to execute a binary target on a device.
"""

# copybara:uncomment_begin(google-only)
# load("//devtools/build_cleaner/skylark:build_defs.bzl", "register_extension_info")
# load("//devtools/deviceinfra/api/builddefs/test:mobile_test.bzl", "mobile_test")
#
# copybara:uncomment_end
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//litert/build_common:litert_build_defs.bzl", "absolute_label")
load("//litert/build_common:special_rule.bzl", "litert_android_linkopts")
load("//litert/integration_test:litert_device_common.bzl", "device_rlocation", "get_spec")
load("//litert/integration_test:litert_device_script.bzl", "litert_device_script")

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

# MOBILE HARNESS WRAPPER ###########################################################################

# copybara:uncomment_begin(google-only)
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

# copybara:comment_begin(oss-only)

def _litert_mh_exec(**unused_kwargs):
    pass

# copybara:comment_end

# RUN ON DEVICE MACRO ##############################################################################

def litert_device_exec(
        name,
        target,
        backend_id = "cpu",
        data = [],
        exec_args = [],
        remote_suffix = "",
        local_suffix = "_adb",
        testonly = True):
    """
    Macro to execute a binary target on a device through adb.

    The output of this macro is an executable shell script that pushes all the necessary files to
    the device and executes the target with the given arguments and environment variables.

    Args:
        name: Name of the target.
        backend_id: The backend id to use for the test (e.g. QUALCOMM_ID, GOOGLE_TENSOR_ID).
        target: The binary target to execute on device.
        data: List of data files to push to the device.
        exec_args: List of arguments to pass to the executable.
        exec_env_vars: List of environment variables to set before executing the target.
        remote_suffix: Suffix for the target runnin on device cloud if enabled.
        local_suffix: Suffix for the target that runs locally on physical device through adb.
        testonly: Whether the target is testonly.
    """
    backend = get_spec(backend_id)

    litert_device_script(
        name = name + local_suffix,
        data = data,
        bin = target,
        script = "//litert/integration_test:mobile_install.sh",
        exec_args = exec_args,
        testonly = testonly,
        backend_id = backend_id,
    )

    # Copybara comment doesn't work right if it is inside an if statement (breaks formatting).
    if remote_suffix != None:
        _litert_mh_exec(
            name = name + remote_suffix,
            target = target,
            run_as = backend.mh_user,
            data = data,
            exec_args = exec_args,
            exec_env_vars = backend.env_paths,
            dimensions = backend.default_mh_device,
        )

def litert_device_test(
        name,
        srcs,
        deps,
        features = [],
        rule = cc_test,
        backend_id = "",
        data = [],
        exec_args = [],
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
        linkopts = litert_android_linkopts() + linkopts,
        copts = copts,
        tags = hidden_test_tags() + tags,
        **kwargs
    )

    litert_device_exec(
        name = name,
        target = absolute_label(":{}".format(target)),
        backend_id = backend_id,
        data = data,
        exec_args = exec_args,
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

    backend = get_spec(backend_id)

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
    target = "//litert/integration_test:gen_device_test"

    litert_device_exec(
        name = name,
        target = target,
        backend_id = backend_id,
        data = data,
        exec_args = cli_args,
    )
