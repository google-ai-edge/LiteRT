# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Checks that every native binary in a Windows archive targets one architecture.

Cross-compiled binaries (e.g. win_arm64 built on an x86_64 host) cannot run on
the build host, so this inspects the PE/COFF header of every .dll/.pyd/.exe in
the given wheels (.whl, whose platform tag is also checked) or zip archives.

Usage:
  python check_windows_wheel_arch.py --arch arm64 dist/*.whl
  python check_windows_wheel_arch.py --arch arm64 dist/*.zip
"""

import argparse
import struct
import sys
import zipfile

_PE_MACHINES = {
    "x86_64": 0x8664,
    "arm64": 0xAA64,
}
_WHEEL_PLATFORM_TAGS = {
    "x86_64": "win_amd64",
    "arm64": "win_arm64",
}
_NATIVE_EXTENSIONS = (".dll", ".pyd", ".exe")


def _pe_machine(data: bytes) -> int | None:
  """Returns the COFF machine type of a PE image, or None if not a PE file."""
  if len(data) < 0x40 or data[:2] != b"MZ":
    return None
  (pe_offset,) = struct.unpack_from("<I", data, 0x3C)
  if len(data) < pe_offset + 6 or data[pe_offset : pe_offset + 4] != b"PE\0\0":
    return None
  (machine,) = struct.unpack_from("<H", data, pe_offset + 4)
  return machine


def check_wheel(wheel_path: str, arch: str) -> list[str]:
  """Returns a list of errors found in the wheel or zip archive."""
  errors = []
  expected_tag = _WHEEL_PLATFORM_TAGS[arch]
  if wheel_path.endswith(".whl") and not wheel_path.endswith(
      f"-{expected_tag}.whl"
  ):
    errors.append(f"{wheel_path}: platform tag is not {expected_tag}")

  expected_machine = _PE_MACHINES[arch]
  num_binaries = 0
  with zipfile.ZipFile(wheel_path) as wheel:
    for name in wheel.namelist():
      if "\\" in name:
        errors.append(f"{wheel_path}: {name} contains backslash path separator")
      if not name.lower().endswith(_NATIVE_EXTENSIONS):
        continue
      num_binaries += 1
      machine = _pe_machine(wheel.read(name))
      if machine is None:
        errors.append(f"{wheel_path}: {name} is not a PE image")
      elif machine != expected_machine:
        errors.append(
            f"{wheel_path}: {name} has machine 0x{machine:04X}, expected"
            f" 0x{expected_machine:04X} ({arch})"
        )
      else:
        print(f"OK  0x{machine:04X}  {name}")
  if num_binaries == 0:
    errors.append(f"{wheel_path}: no native binaries found")
  return errors


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--arch", choices=sorted(_PE_MACHINES), required=True)
  parser.add_argument("archives", nargs="+")
  args = parser.parse_args()

  errors = []
  for wheel_path in args.archives:
    errors.extend(check_wheel(wheel_path, args.arch))
  for error in errors:
    print(f"ERROR: {error}", file=sys.stderr)
  return 1 if errors else 0


if __name__ == "__main__":
  sys.exit(main())
