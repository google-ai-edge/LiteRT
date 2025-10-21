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

def _make_ats_args(quote_re, **kwargs):
    def _fmt_re(re):
        if len(re) == 1:
            return re[0]
        if quote_re:
            return "\\'({})\\'".format("|".join(re))
        return "({})".format("|".join(re))

    extra_flags = kwargs.get("extra_flags", [])
    exec_args = [
        "--quiet=false",
    ] + extra_flags

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
        name,
        backend,
        dont_register = [],
        do_register = [],
        param_seeds = {},
        extra_flags = []):
    """Defines a pre-configured ATS test suite.

    Args:
      name: The name of the test suite.
      backend: The backend to use for the test suite.
      dont_register: A list of regular expressions for tests that should not be registered.
      do_register: A list of regular expressions for tests that should be registered.
      param_seeds: A dictionary of parameter seeds for the test suite.
      extra_flags: A list of extra flags to pass to the test suite.
    """

    run_args = _make_ats_args(
        backend = backend,
        dont_register = dont_register,
        do_register = do_register,
        param_seeds = param_seeds,
        extra_flags = extra_flags,
        quote_re = True,
    )
    if is_npu_backend(backend):
        run_args += [
            "--dispatch_dir=\"{}\"".format(dispatch_device_rlocation(backend)),
            "--plugin_dir=\"{}\"".format(plugin_device_rlocation(backend)),
        ]

    litert_device_exec(
        name = name,
        target = "//litert/ats:ats",
        backend_id = backend,
        exec_args = run_args,
    )
