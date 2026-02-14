# Copyright 2018 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Support functions for common --define operations."""

def _bool_value(define_name, default, *, config_vars = None):
    """Looks up a define on ctx for a boolean value.

    Will also report an error if the value is not a supported value.

    Args:
      define_name: The name of the define to look up.
      default: The value to return if the define isn't found.
      config_vars: A dictionary (String to String) of configuration variables. Can be from ctx.var.

    Returns:
      True/False or the default value if the define wasn't found.
    """
    value = config_vars.get(define_name, None)
    if value != None:
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        fail("Valid values for --define={} are: true|yes|1 or false|no|0.".format(
            define_name,
        ))
    return default

defines = struct(
    bool_value = _bool_value,
)
