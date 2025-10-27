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

"""Template expansion that supports bazel variable resolution.."""

load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
load("@rules_shell//shell:sh_library.bzl", "sh_library")
load("//litert/integration_test:litert_device_common.bzl", "get_spec", "split_dep_platform")

def _extract_binary(target):
    # cc_test case
    if RunEnvironmentInfo in target:
        environment = target[RunEnvironmentInfo].environment
        cc_test_binary = environment.get("CC_TEST_BINARY", None)
        if cc_test_binary:
            return cc_test_binary

    # cc_binary case
    file_to_run = target.files_to_run.executable
    if file_to_run:
        return file_to_run.short_path

    fail("Could not locate an executable file provided by the target, excpected cc_test or cc_binary.")

def _fmt_location(files):
    return "\"{}\"".format(" ".join([f.short_path for f in files]))

def _device_script_lib_impl(ctx):
    # TODO: The "for_{device/host}" ones are just a shell script that calls the actual thing with
    # "run_under", make the resolution here more robust. We have to grab the actual file
    # for the device case since that indirection won't work.

    forwarded_runfiles = []
    subs = ctx.actions.template_dict()

    if ctx.attr.host_bin:
        files = [f for f in ctx.attr.host_bin.files.to_list() if "for_host" in f.short_path]
        if len(files) != 1:
            fail("Should be only one output for single host_bin target built with platform data.")
        binary = _extract_binary(ctx.attr.host_bin)
        forwarded_runfiles.append(files[0])
        subs.add("@@host_bin@@", "\"{}\"".format(binary))

    if ctx.attr.device_bin:
        files = [f for f in ctx.attr.device_bin.files.to_list() if "for_device" not in f.short_path]
        if len(files) != 1:
            fail("Should be only one output for single device_bin target built with platform data.")
        binary = files[0].short_path
        forwarded_runfiles.append(files[0])
        subs.add("@@device_bin@@", "\"{}\"".format(binary))

    host_libs = []
    for v in ctx.attr.host_libs:
        files = [f for f in v.files.to_list() if "for_host" in f.short_path]
        if len(files) != 1:
            fail("Should be only one output for single host_lib target built with platform data.")
        forwarded_runfiles.extend(files)
        host_libs.extend(files)
    for v in ctx.attr.extra_host_libs:
        files = v.files.to_list()
        forwarded_runfiles.extend(files)
        host_libs.extend(files)
    subs.add("@@host_libs@@", _fmt_location(host_libs))

    device_libs = []
    for v in ctx.attr.device_libs:
        files = [f for f in v.files.to_list() if "for_device" not in f.short_path]
        if len(files) != 1:
            fail("Should be only one output for single device_lib target built with platform data.")
        forwarded_runfiles.extend(files)
        device_libs.extend(files)
    for v in ctx.attr.extra_device_libs:
        files = v.files.to_list()
        forwarded_runfiles.extend(files)
        device_libs.extend(files)
    subs.add("@@device_libs@@", _fmt_location(device_libs))

    data_files = []
    for v in ctx.attr.data:
        files = v.files.to_list()
        forwarded_runfiles.extend(files)
        data_files.extend(files)
    subs.add("@@data@@", _fmt_location(data_files))

    subs.add("@@exec_env_vars@@", "\"{}\"".format(" ".join(ctx.attr.exec_env_vars)))

    runfiles = ctx.runfiles(files = forwarded_runfiles)

    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.out,
        computed_substitutions = subs,
    )

    return [DefaultInfo(runfiles = runfiles)]

_device_script_lib = rule(
    implementation = _device_script_lib_impl,
    doc = """
    Expands a template shell script, substituting placeholders for host/device binaries,
    libraries, data files, and execution environment variables.

    This rule collects all necessary files (host/device binaries, libraries, and data)
    and makes them available in the runfiles of the generated target. The template
    can use placeholders like `@@host_bin@@`, `@@device_bin@@`, `@@host_libs@@`,
    `@@device_libs@@`, `@@data@@`, and `@@exec_env_vars@@`, which will be replaced
    with the appropriate paths or values.
    """,
    attrs = {
        "template": attr.label(
            mandatory = True,
            allow_single_file = True,
            doc = "The template shell file to expand.",
        ),
        "out": attr.output(
            mandatory = True,
            doc = "The destination of the expanded file.",
        ),
        "host_bin": attr.label(
            mandatory = False,
            default = None,
            executable = True,
            cfg = "exec",
            doc = "The binary for host to package with the script.",
        ),
        "device_bin": attr.label(
            mandatory = False,
            default = None,
            executable = True,
            cfg = "target",
            doc = "The binary for device to package with the script.",
        ),
        "data": attr.label_list(
            mandatory = False,
            default = [],
            allow_files = True,
            doc = "The data files to package with the script.",
        ),
        "host_libs": attr.label_list(
            mandatory = False,
            default = [],
            allow_files = True,
            doc = "The buildable host shared libraries to package with the script (e.g. compiler plugins, dispatch libraries, etc.).",
        ),
        "extra_host_libs": attr.label_list(
            mandatory = False,
            default = [],
            allow_files = True,
            doc = "The pre-compiled host shared libraries to package with the script (e.g. vendor sdk).",
        ),
        "device_libs": attr.label_list(
            mandatory = False,
            default = [],
            allow_files = True,
            doc = "The buildable device shared libraries to package with the script (e.g. compiler plugins, dispatch libraries, etc.).",
        ),
        "extra_device_libs": attr.label_list(
            mandatory = False,
            default = [],
            allow_files = True,
            doc = "The pre-compiled device shared libraries to package with the script (e.g. vendor sdk).",
        ),
        "exec_env_vars": attr.string_list(
            mandatory = False,
            default = [],
            doc = "The environment variables to set before executing the target.",
        ),
    },
)

def litert_device_script(
        name,
        script = None,
        bin = None,
        data = [],
        testonly = True,
        exec_args = [],
        backend_id = "cpu",
        build_for_host = False,
        build_for_device = True):
    """Generates a shell script and runfiles for executing a binary on a device.

    This macro sets up a build environment to run a specified binary (`bin`) on a target device,
    potentially requiring different versions of the binary and libraries for the host and device.
    It packages necessary data, host-specific libraries, and device-specific libraries based on
    the provided `backend_id`.

    Args:
      name: The name of the generated sh_binary target.
      script: The main shell script file to be executed.
      bin: The label of the binary target to be run. This binary will be built for both the host
           and the device platforms.
      data: A list of additional data dependencies required by the script or binary.
      testonly: If True, the generated targets are marked as testonly.
      exec_args: A list of arguments to be passed to the final sh_binary.
      backend_id: The identifier for the backend configuration (e.g., "cpu", "gpu").
                  Used to fetch backend-specific libraries and environment variables.
      build_for_host: If True, build the packaged deps for the host platform.
      build_for_device: If True, build the packaged deps for the device platform.
    """

    if build_for_host and build_for_device:
        fail("Double split config not yet supported.")

    split_dep_kwargs = {"prefix": name, "testonly": testonly, "no_host": not build_for_host, "no_device": not build_for_device}

    backend = get_spec(backend_id)
    bins = split_dep_platform(targets = bin, **split_dep_kwargs)

    buildable_libs = []
    extra_device_libs = []
    extra_host_libs = []

    for lib in backend.libs:
        if backend.plugin and backend.plugin in lib:
            buildable_libs.append(lib)
        elif backend.dispatch and backend.dispatch in lib:
            buildable_libs.append(lib)
        elif "libLiteRtRuntimeCApi.so" in lib:
            buildable_libs.append(lib)
        else:
            extra_device_libs.append(lib)

    for lib in backend.host_libs:
        extra_host_libs.append(lib)

    libs = split_dep_platform(targets = buildable_libs, **split_dep_kwargs)

    _device_script_lib(
        name = name + "_lib_expanded",
        template = "//litert/integration_test:device_script_template.sh",
        out = name + "_lib.sh",
        host_bin = bins.host[0] if bins.host else None,
        device_bin = bins.device[0] if bins.device else None,
        data = data,
        host_libs = libs.host,
        device_libs = libs.device,
        extra_host_libs = extra_host_libs,
        extra_device_libs = extra_device_libs,
        testonly = testonly,
        exec_env_vars = backend.env_paths,
    )

    sh_library(
        name = name + "_lib",
        srcs = [":" + name + "_lib_expanded"],
        testonly = testonly,
    )

    sh_binary(
        name = name,
        srcs = [script],
        deps = [":" + name + "_lib"],
        testonly = testonly,
        args = exec_args,
    )
