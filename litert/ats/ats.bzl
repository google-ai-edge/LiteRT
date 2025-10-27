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
Macros to define pre-configured ATS test suites and run through the litert_device* macros.
"""

load("//litert/integration_test:litert_device.bzl", "litert_device_exec")
load("//litert/integration_test:litert_device_common.bzl", "dispatch_device_rlocation", "is_npu_backend", "plugin_device_rlocation")
load("//litert/integration_test:litert_device_script.bzl", "litert_device_script")

def _make_ats_args(quote_re, init = [], **kwargs):
    def _fmt_re(re):
        if len(re) == 1:
            return re[0]
        if quote_re:
            return "\\'({})\\'".format("|".join(re))
        return "({})".format("|".join(re))

    extra_flags = kwargs.get("extra_flags", [])
    exec_args = [
        "--quiet=false",
    ] + extra_flags + init

    backend = kwargs.get("backend", "cpu")
    if is_npu_backend(backend):
        exec_args += [
            "--backend=npu",
        ]
    else:
        exec_args.append(
            "--backend=\"{}\"".format(backend),
        )

    dont_register = kwargs.get("dont_register", [])
    if dont_register:
        exec_args.append(
            "--dont_register={}".format(_fmt_re(dont_register)),
        )

    do_register = kwargs.get("do_register", [])
    if do_register:
        exec_args.append(
            "--do_register={}".format(_fmt_re(do_register)),
        )

    param_seeds = kwargs.get("param_seeds", {})
    if param_seeds:
        exec_args.append(
            "--seeds=\"{}\"".format(",".join(["{}:{}".format(k, v) for k, v in param_seeds.items()])),
        )
    return exec_args

def litert_define_ats(
        backend,
        name,
        jit_suffix,
        compile_only_suffix,
        compile_aot_and_run_suffix = None,
        dont_register = [],
        do_register = [],
        param_seeds = {},
        extra_flags = []):
    """Defines a pre-configured ATS test suite.

    Args:
      name: The name of the test suite.
      backend: The backend to use for the test suite.
      jit_suffix: Suffix for the Just-In-Time execution target.
      compile_only_suffix: Suffix for the Compile-Only target.
      compile_aot_and_run_suffix: Suffix for the Compile AOT and Run target.
      dont_register: A list of regular expressions for tests that should not be registered.
      do_register: A list of regular expressions for tests that should be registered.
      param_seeds: A dictionary of parameter seeds for the test suite.
      extra_flags: A list of extra flags to pass to the test suite.
    """

    if compile_aot_and_run_suffix:
        fail("Compile aot and run on device is not supported yet.")

    init_run_args = []
    if is_npu_backend(backend):
        init_run_args += [
            "--dispatch_dir=\"{}\"".format(dispatch_device_rlocation(backend)),
            "--plugin_dir=\"{}\"".format(plugin_device_rlocation(backend)),
        ]

    run_args = _make_ats_args(
        init = init_run_args,
        backend = backend,
        dont_register = dont_register,
        do_register = do_register,
        param_seeds = param_seeds,
        extra_flags = extra_flags,
        quote_re = True,
    )

    if jit_suffix != None:
        litert_device_exec(
            name = name + jit_suffix,
            target = "//litert/ats:ats",
            remote_suffix = "_remote",
            local_suffix = "",
            exec_args = run_args,
            backend_id = backend,
        )

    compile_args = _make_ats_args(
        init = ["--compile_mode=true"],
        backend = backend,
        dont_register = dont_register,
        do_register = do_register,
        param_seeds = param_seeds,
        extra_flags = extra_flags,
        quote_re = True,
    )

    if compile_only_suffix != None:
        litert_device_script(
            name = name + compile_only_suffix,
            script = "//litert/ats:ats_aot.sh",
            bin = "//litert/ats:ats",
            backend_id = backend,
            exec_args = compile_args,
            build_for_host = True,
            build_for_device = False,
        )
