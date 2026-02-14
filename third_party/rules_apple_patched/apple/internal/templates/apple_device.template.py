#!/usr/bin/env python3

# Copyright 2024 The Bazel Authors. All rights reserved.
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

"""Invoked by `bazel run` to launch *_application targets on a physical device."""

# This script works in one of two modes.
#
# If no device identifier is provided:
#
# 1. Discovers a compatible device with the minimum_os of the
#    *_application target. If not found, fail.
# 2. Installs and launches the application
# 3. Displays the application's output on the console
#
# If a device identifier is provided:
#
# 1. Installs and launches the application on a device corresponding to `device_identifier`.
# 2. Displays the application's output on the console

import collections.abc
import contextlib
import json
import logging
import os
import os.path
import pathlib
import platform
import plistlib
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional
from uuid import uuid4
import zipfile


logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

if platform.system() != "Darwin":
  raise Exception(
      "Cannot run Apple platform application targets on a non-mac machine."
  )


class Device(collections.abc.Mapping):
  """Wraps the `device` dictionary from `devicectl list devices -j`.

  Provides an ordering so
  booted devices > shutdown devices,
  paired devices > unpaired devices,
  iPhones > iPads.
  In addition, maintains the original order from `devicectl list devices`
  as `list_index` to ensure newer device types are sorted after older device types
  """

  def __init__(self, device, list_index):
    self.device = device
    self.list_index = list_index

  @property
  def device_properties(self) -> Dict[str, Any]:
    return self["deviceProperties"]

  @property
  def hardware_properties(self) -> Dict[str, Any]:
    return self["hardwareProperties"]

  @property
  def name(self) -> str:
    return self.device_properties["name"]

  @property
  def identifier(self) -> str:
    return self["identifier"]

  @property
  def udid(self) -> str:
    return self.hardware_properties.get("udid", "unknown")

  @property
  def device_type(self) -> str:
    return self.hardware_properties.get("deviceType", "unknown")

  @property
  def os_version_number(self) -> str:
    return self.device_properties.get("osVersionNumber", "unknown")

  @property
  def is_apple_tv(self) -> bool:
    return self.device_type.lower() == "appletv"

  @property
  def is_apple_watch(self) -> bool:
    return self.device_type.lower() == "applewatch"

  @property
  def is_apple_vision(self) -> bool:
    return self.device_type.lower() == "applevision"

  @property
  def is_iphone(self) -> bool:
    return self.device_type.lower() == "iphone"

  @property
  def is_ipad(self) -> bool:
    return self.device_type.lower() == "ipad"

  @property
  def is_shutdown(self):
    return not self.is_booted

  @property
  def is_booted(self):
    return self.device_properties.get("bootState", "unknown") == "Booted"

  @property
  def is_paired(self):
    return self.device_properties.get("pairingState", "unknown") == "paired"

  def __getitem__(self, name):
    return self.device[name]

  def __iter__(self):
    return iter(self.device)

  def __len__(self):
    return len(self.device)

  def __repr__(self):
    return self.name + "(" + self.udid + ")"

  def __lt__(self, other):
    if self.is_shutdown and other.is_booted:
      return True
    elif self.is_booted and other.is_shutdown:
      return False
    elif not self.is_paired and other.is_paired:
      return True
    elif self.is_paired and not other.is_paired:
      return False
    elif self.is_ipad and other.is_iphone:
      return True
    elif self.is_iphone and other.is_ipad:
      return False
    return self.list_index < other.list_index

  def supports_platform_type(self, platform_type: str) -> bool:
    """Returns boolean to indicate if device supports given Apple platform type."""
    if platform_type == "ios":
      return self.is_iphone or self.is_ipad
    elif platform_type == "tvos":
      return self.is_apple_tv
    elif platform_type == "watchos":
      return self.is_apple_watch
    elif platform_type == "visionos":
      return self.is_apple_vision
    else:
      raise ValueError(
          f"Apple platform type not supported for running on device: {platform_type}."
      )


def os_version_number_to_int(version: str) -> int:
  """Converts a OS version number string to an integer.
  Args:
    minimum_os: A string in the form '12.2' or '13.2.3'.

  Returns:
    An integer in the form 0xAABBCC, where AA is the major version, BB is
    the minor version, and CC is the micro version.
  """
  # Pad the version to major.minor.micro.
  version_components = (version.split(".") + ["0"] * 3)[:3]
  result = 0
  for component in version_components:
    result = (result << 8) | int(component)
  return result


def discover_best_compatible_device(
    *,
    platform_type: str,
    devicectl_path: str,
    minimum_os: str,
) -> Optional[Device]:
  """Discovers the most suitable compatible device by prioritizing booted devices
  over shutdown ones, paired devices over unpaired ones, and iPhones over iPads
  and also maintains the original order from devicectl list devices.

  Args:
    platform_type: The Apple platform type for the given *_application() target.
    devicectl_path: The path to the `devicectl` binary.
    minimum_os: The minimum OS version required by the *_application() target.

  Returns:
    A tuple (device_type, device) containing the DeviceType and Device
    of the best compatible simulator (might be None if no match was found).

  Raises:
    subprocess.SubprocessError: if `devicectl list devices` fails or times out.
  """
  json_fp = tempfile.NamedTemporaryFile(delete=False)
  subprocess.run(
      [devicectl_path, "list", "devices", "--json-output", json_fp.name],
      capture_output=True,
      check=True,
  )
  with open(json_fp.name, 'r') as json_file:
    devicectl_data = json.load(json_file)
  json_fp.close()
  os.unlink(json_fp.name)
  compatible_devices = []
  minimum_version_int = os_version_number_to_int(minimum_os)
  # Remember the index of each device type to preserve that ordering when
  # sorting device types.
  for list_index, device_data in enumerate(devicectl_data["result"]["devices"]):
    device = Device(device_data, list_index)
    if not device.supports_platform_type(platform_type):
      continue
    version_int = os_version_number_to_int(device.os_version_number)
    if version_int < minimum_version_int:
      continue
    compatible_devices.append(device)
  compatible_devices.sort()
  logger.debug(
      "Found %d compatible devices.", len(compatible_devices)
  )
  return  compatible_devices[-1] if compatible_devices else None


def register_dsyms(dsyms_dir: str):
  """Adds all dSYMs in `dsyms_dir` to the symbolscache.

  Args:
    dsyms_dir: Path to directory potentially containing dSYMs
  """
  symbolscache_command = [
      "/usr/bin/symbolscache",
      "delete",
      "--tag",
      "Bazel",
      "compact",
      "add",
      "--tag",
      "Bazel",
  ] + [
      a
      for a in pathlib.Path(dsyms_dir).glob(
          "**/*.dSYM/Contents/Resources/DWARF/*"
      )
  ]
  logger.debug("Running command: %s", symbolscache_command)
  result = subprocess.run(
      symbolscache_command,
      capture_output=True,
      check=True,
      encoding="utf-8",
      text=True,
  )
  logger.debug("symbolscache output: %s", result.stdout)


@contextlib.contextmanager
def extracted_app(
    application_output_path: str, app_name: str
) -> collections.abc.Generator[str, None, None]:
  """Extracts Foo.app from *_application() output and makes it writable.

  Args:
    application_output_path: Path to the output of an `*_application()`. If the
      path is a directory, copies it to a temporary directory and makes the
      contents writable, as `simctl install` fails to install an `.app` that is
      read-only. If the path is an .ipa archive, unzips it to a temporary
      directory.
    app_name: The name of the application (e.g. "Foo" for "Foo.app").

  Yields:
    Path to Foo.app in temporary directory (re-used if already present).
  """
  if os.path.isdir(application_output_path):
    # Re-use the same path for each run and rsync to it (reducing
    # copies). Ensure the result is writable, or `simctl install` will
    # fail with `Unhandled error domain NSPOSIXErrorDomain, code 13`.
    dst_dir = os.path.join(tempfile.gettempdir(), "bazel_temp_" + app_name)
    os.makedirs(dst_dir, exist_ok=True)

    # NOTE: use `which` to find the path to `rsync`.
    # In macOS 15.4, the system `rsync` is using `openrsync` which contains some permission issues.
    # This allows users to workaround the issue by overriding the system `rsync` with a working version.
    # Remove this once we no longer support macOS versions with broken `rsync`.
    rsync_path = shutil.which("rsync")

    rsync_command = [
        rsync_path,
        "--archive",
        "--delete",
        "--checksum",
        "--chmod=u+w",
        "--verbose",
        # The output path might itself be a symlink; resolve to the
        # real path so rsync doesn't just copy the symlink.
        os.path.realpath(application_output_path),
        dst_dir,
    ]
    logger.debug(
        "Found app directory: %s, running command: %s",
        application_output_path,
        rsync_command,
    )
    result = subprocess.run(
        rsync_command,
        capture_output=True,
        check=True,
        encoding="utf-8",
        text=True,
    )
    logger.debug("rsync output: %s", result.stdout)
    yield os.path.join(dst_dir, app_name + ".app")
  else:
    # Create a new temporary directory for each run, deleting it
    # afterwards (there's no efficient way to "sync" an unzip, so this
    # can't re-use the output directory).
    with tempfile.TemporaryDirectory(prefix="bazel_temp") as temp_dir:
      logger.debug(
          "Unzipping IPA from %s to %s", application_output_path, temp_dir
      )
      with zipfile.ZipFile(application_output_path) as ipa_zipfile:
        ipa_zipfile.extractall(temp_dir)
        yield os.path.join(temp_dir, "Payload", app_name + ".app")


def bundle_id(bundle_path: str) -> str:
  """Returns the bundle ID given a bundle directory path."""
  info_plist_path = os.path.join(bundle_path, "Info.plist")
  with open(info_plist_path, mode="rb") as plist_file:
    plist = plistlib.load(plist_file)
    return plist["CFBundleIdentifier"]


def devicectl_launch_environ(device_identifier: str) -> Dict[str, str]:
  """Calculates an environment dictionary for running `devicectl device process launch`."""
  # Pass environment variables prefixed with "IOS_" to the device, replace
  # the prefix with "DEVICECTL_CHILD_". bazel adds "IOS_" to the env vars which
  # will be passed to the app as prefix to differentiate from other env vars. We
  # replace the prefix "IOS_" with "DEVICECTL_CHILD_" here, because "devicectl" only
  # pass the env vars prefixed with "DEVICECTL_CHILD_" to the app.
  result = {
    "DEVICECTL_CHILD_BAZEL_DEVICE_UDID": device_identifier,
  }
  for k, v in os.environ.items():
    if not k.startswith("IOS_"):
      continue
    new_key = k.replace("IOS_", "DEVICECTL_CHILD_", 1)
    result[new_key] = v
  if "IDE_DISABLED_OS_ACTIVITY_DT_MODE" not in os.environ:
    # Ensure os_log() mirrors writes to stderr. (lldb and Xcode set this
    # environment variable as well.)
    result["DEVICECTL_CHILD_OS_ACTIVITY_DT_MODE"] = "enable"
  return result


def run_app(
    *,
    device_identifier: str,
    devicectl_path: str,
    application_output_path: str,
    app_name: str,
) -> None:
  """Installs and runs an app on the specified device.

  Args:
    device_identifier: The identifier of the device.
    devicectl_path: The path to the `devicectl` binary.
    application_output_path: Path to the output of an `*_application()`.
    app_name: The name of the application (e.g. "Foo" for "Foo.app").
  """
  root_dir = os.path.dirname(application_output_path)
  register_dsyms(root_dir)
  with extracted_app(application_output_path, app_name) as app_path:
    logger.info("Installing app %s to device %s", app_path, device_identifier)
    subprocess.run(
        [
          devicectl_path,
          "device",
          "install",
          "app",
          "--device",
          device_identifier,
          app_path
        ],
        check=True,
    )
    app_bundle_id = bundle_id(app_path)
    launch_args = shlex.split(
      os.environ.get(
        "BAZEL_DEVICECTL_LAUNCH_FLAGS",
        # Attaches the application to the console and waits for it to exit.
        "--console",
      ),
    )
    logger.info(
        "Launching app %s on %s", app_bundle_id, device_identifier
    )
    launch_args = [
        devicectl_path,
        "device",
        "process",
        "launch",
        *launch_args,
        "--device",
        device_identifier,
    ]
    # Append optional launch arguments.
    app_args = [app_bundle_id] + sys.argv[1:]
    launch_app(
        launch_args=launch_args,
        app_args=app_args,
        env=devicectl_launch_environ(device_identifier=device_identifier),
        device_identifier=device_identifier,
    )


def launch_app(
    *,
    launch_args: list[str],
    app_args: list[str],
    env: Dict[str, str],
    device_identifier: str,
) -> None:
  """Launches an app in a simulator.

  Args:
    launch_args: The arguments to pass to simctl to launch the app, excluding
      the bundle id and app arguments.
    app_args: The bundle id and app arguments to pass to simctl to launch the
      app.
    env: The environment variables to pass to simctl.
    device_identifier: The identifier of the device.
  """
  launch_info_path = os.environ.get("BAZEL_APPLE_LAUNCH_INFO_PATH")
  if not launch_info_path:
    subprocess.run(launch_args + app_args, env=env, check=True)
    return

  if "--json-output" in launch_args:
    idx = launch_args.index("--json-output")
    json_output_path = launch_args[idx + 1]
    delete_json_output = False
  else:
    json_output_path = os.path.join(
        tempfile.gettempdir(), f"devicectl_launch_{uuid4().hex}.json",
    )
    launch_args = launch_args + ["--json-output", json_output_path]
    delete_json_output = True
  args = launch_args + app_args

  proc = subprocess.Popen(
      args,
      env=env,
  )

  exit_code = proc.wait()

  # `devicectl` only writes to `--json-output` after the process has exited. We
  # use `subprocess.Popen()` instead of `subprocess.run()` to allow us to write
  # the launch info before reporting process exit.
  try:
    with open(json_output_path, "r", encoding="utf-8") as f:
      data = json.load(f)
      pid = (
          data.get("result", {}).get("process", {}).get("processIdentifier")
      )
      if pid:
        os.makedirs(os.path.dirname(launch_info_path), exist_ok=True)
        with open(launch_info_path, "w", encoding="utf-8") as f:
          f.write(json.dumps(
              {
                  "platform": "device",
                  "udid": device_identifier,
                  "pid": pid,
              },
              indent=2,
          ))
      else:
          logger.error("Failed to find PID in JSON output")
  except Exception as e:
    logger.error("Failed to write launch info to file: %s", e)

  if delete_json_output:
    try:
      os.remove(json_output_path)
    except Exception:
      pass

  if exit_code != 0:
    raise subprocess.CalledProcessError(exit_code, args)


def main(
    *,
    app_name: str,
    application_output_path: str,
    minimum_os: str,
    platform_type: str,
    device_identifier: str,
):
  """Main entry point to `bazel run` for *_application() targets.

  Args:
    app_name: The name of the application (e.g. "Foo" for "Foo.app").
    application_output_path: Path to the output of an *_application().
    minimum_os: The minimum OS version required by the *_application() target.
    platform_type: The Apple platform type for the given *_application() target.
    device: The identifier of the device (<uuid|ecid|serial_number|udid|name|dns_name>).
  """
  xcode_select_result = subprocess.run(
      ["xcode-select", "-p"],
      encoding="utf-8",
      check=True,
      stdout=subprocess.PIPE,
  )
  developer_path = xcode_select_result.stdout.rstrip()
  devicectl_path = os.path.join(developer_path, "usr", "bin", "devicectl")

  if not device_identifier:
    logger.info(
        f"Searching for a compatible device for platform type %s with minimum OS %s",
        platform_type,
        minimum_os,
    )
    device = discover_best_compatible_device(
      platform_type=platform_type,
      devicectl_path=devicectl_path,
      minimum_os=minimum_os,
    )
    if not device:
      raise Exception(
          f"No compatible device found for platform type % with minimum OS %s",
          platform_type,
          minimum_os,
      )
    else:
      logger.info(
        "Found device %s with identifier %s",
        device.name,
        device.identifier,
      )
    device_identifier = device.identifier
  else:
    logger.info("Using device %s", device_identifier)

  run_app(
      device_identifier=device_identifier,
      devicectl_path=devicectl_path,
      application_output_path=application_output_path,
      app_name=app_name,
  )


if __name__ == "__main__":
  try:
    # Template values filled in by rules_apple/apple/internal/run_support.bzl.
    main(
        app_name="%app_name%",
        application_output_path="%ipa_file%",
        minimum_os="%minimum_os%",
        platform_type="%platform_type%",
        device_identifier="%device%",
    )
  except subprocess.CalledProcessError as e:
    logger.error("%s exited with error code %d", e.cmd, e.returncode)
    sys.exit(e.returncode)
  except KeyboardInterrupt:
    sys.exit(1)
