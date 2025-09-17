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

load("//litert/integration_test:litert_device.bzl", "dispatch_device_rlocation", "is_npu_backend", "litert_device_exec", "plugin_device_rlocation")

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
    exec_args = [
        "--quiet=true",
    ] + extra_flags

    if is_npu_backend(backend):
        exec_args += [
            "--dispatch_dir=\"{}\"".format(dispatch_device_rlocation(backend)),
            "--plugin_dir=\"{}\"".format(plugin_device_rlocation(backend)),
            "--backend=npu",
        ]
    else:
        exec_args.append(
            "--backend=\"{}\"".format(backend),
        )

    if dont_register:
        if len(dont_register) == 1:
            dont_register_str = dont_register[0]
        else:
            dont_register_str = "({})".format("|".join(dont_register))
        exec_args.append(
            "--dont_register='{}'".format(dont_register_str),
        )

    if do_register:
        if len(do_register) == 1:
            do_register_str = do_register[0]
        else:
            do_register_str = "({})".format("|".join(do_register))

        exec_args.append(
            "--do_register='{}'".format(do_register_str),
        )

    if param_seeds:
        exec_args.append(
            "--seeds=\"{}\"".format(",".join(["{}:{}".format(k, v) for k, v in param_seeds.items()])),
        )

    litert_device_exec(
        name = name,
        target = "//litert/ats:ats",
        backend_id = backend,
        exec_args = exec_args,
    )
