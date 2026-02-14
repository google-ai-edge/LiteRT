# Copyright 2020 The Bazel Authors. All rights reserved.
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
#

from tools.wrapper_common import execute


def invoke_lipo(binary_path, binary_slices, output_path):
  """Wraps lipo with given arguments for inputs and outputs."""
  cmd = ["xcrun", "lipo", binary_path]
  # Create a thin binary if there's only one needed slice, otherwise create a
  # universal binary
  if len(binary_slices) == 1:
    cmd.extend(["-thin", next(iter(binary_slices))])
  else:
    for binary_slice in binary_slices:
      cmd.extend(["-extract", binary_slice])
  cmd.extend(["-output", output_path])
  _, stdout, stderr = execute.execute_and_filter_output(cmd,
                                                        raise_on_failure=True)
  if stdout:
    print(stdout)
  if stderr:
    print(stderr)


def find_archs_for_binaries(binary_list):
  """Queries lipo to identify binary archs from each of the binaries.

  Args:
    binary_list: A list of strings, each of which is the path to a binary whose
      architectures should be retrieved.

  Returns:
    A tuple containing two values:

    1.  A set containing the union of all architectures found in every binary.
    2.  A dictionary where each key is one of the elements in `binary_list` and
        the corresponding value is the set of architectures found in that
        binary.

    If there was an error invoking `lipo` or the output was something
    unexpected, `None` will be returned for both tuple elements.
  """
  found_architectures = set()
  archs_by_binary = dict()

  for binary in binary_list:
    cmd = ["xcrun", "lipo", "-info", binary]
    _, stdout, stderr = execute.execute_and_filter_output(cmd,
                                                          raise_on_failure=True)
    if stderr:
      print(stderr)
    if not stdout:
      print("Internal Error: Did not receive output from lipo for inputs: " +
            " ".join(cmd))
      return (None, None)

    cut_output = stdout.split(":")
    if len(cut_output) < 3:
      print("Internal Error: Unexpected output from lipo, received: " + stdout)
      return (None, None)

    archs_found = cut_output[2].strip().split(" ")
    if not archs_found:
      print("Internal Error: Could not find architecture for binary: " + binary)
      return (None, None)

    archs_by_binary[binary] = set(archs_found)

    for arch_found in archs_found:
      found_architectures.add(arch_found)

  return (found_architectures, archs_by_binary)
