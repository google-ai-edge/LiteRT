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
Macros to define pre-configured CTS test suites and run through the litert_device* macros.
"""

load("//litert/integration_test:litert_device.bzl", "litert_device_exec")

def litert_define_cts(
        name,
        backend,
        dont_register = [],
        param_seeds = {}):
    """Defines a pre-configured CTS test suite.

    Args:
      name: The name of the test suite.
      backend: The backend to use for the test suite.
      dont_register: A list of regular expressions for tests that should not be registered.
      param_seeds: A dictionary of parameter seeds for the test suite.
    """
    dont_reg_str = "({})".format("|".join(dont_register))
    seeds_str = ",".join(["{}:{}".format(k, v) for k, v in param_seeds.items()])
    litert_device_exec(
        name = name,
        target = "//litert/cts:cts",
        backend_id = backend,
        exec_args = [
            "--backend={}".format(backend),
            "--dont_register=\"{}\"".format(dont_reg_str),
            "--seeds=\"{}\"".format(seeds_str),
            "--quiet=false",
        ],
    )
