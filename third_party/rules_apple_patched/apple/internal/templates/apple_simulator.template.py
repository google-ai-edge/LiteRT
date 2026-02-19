#!/usr/bin/env python3

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

"""Invoked by `bazel run` to launch *_application targets in the simulator."""

# This script works in one of two modes.
#
# If either --ios_simulator_version or --ios_simulator_device were not
# passed to bazel:
#
# 1. Discovers a simulator compatible with the minimum_os of the
#    *_application target, preferring already-booted simulators
#    if possible
# 2. Boots the simulator if needed
# 3. Installs and launches the application
# 4. Displays the application's output on the console
#
# This mode does not kill running simulators or shutdown or delete the simulator
# after it completes.
#
# If --ios_simulator_version and --ios_simulator_device were both passed
# to bazel:
#
# 1. Creates a new temporary simulator by running "simctl create ..."
# 2. Boots the new temporary simulator
# 3. Installs and launches the application
# 4. Displays the application's output on the console
# 5. When done, shuts down and deletes the newly-created simulator
#
# All environment variables with names starting with "IOS_" are passed to the
# application, after stripping the prefix "IOS_".

import collections.abc
import contextlib
import json
import logging
import os
import os.path
import pathlib
import platform
import plistlib
import pty
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import IO, Dict, Optional, Sequence
import zipfile


# Custom type for methods yielding an Apple simulator UDID.
AppleSimulatorUDID = collections.abc.Generator[str, None, None]


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


class BufferFlusher:
  """Flushes a buffer to a file descriptor.

  This is used to ensure that the buffer is flushed to the file descriptor
  as soon as possible.
  """

  def __init__(self, raw: IO[bytes]):
    self.raw = raw

  def write(self, b: bytes) -> int:
    n = self.raw.write(b)
    self.raw.flush()
    return n


class DeviceType(collections.abc.Mapping):
  """Wraps the `devicetype` dictionary from `simctl list -j`.

  Provides an ordering so iPhones > iPads. In addition, maintains the
  original order from `simctl list` as `simctl_list_index` to ensure
  newer device types are sorted after older device types.
  """

  def __init__(self, device_type, simctl_list_index):
    self.device_type = device_type
    self.simctl_list_index = simctl_list_index

  def __getitem__(self, name):
    return self.device_type[name]

  def __iter__(self):
    return iter(self.device_type)

  def __len__(self):
    return len(self.device_type)

  def __repr__(self):
    return self["name"] + " (" + self["identifier"] + ")"

  def __lt__(self, other):
    # Order iPhones ahead of (later in the list than) iPads.
    if self.is_ipad() and other.is_iphone():
      return True
    elif self.is_iphone() and other.is_ipad():
      return False
    # Order device types from the same product family in the same order
    # as `simctl list`.
    return self.simctl_list_index < other.simctl_list_index

  def supports_platform_type(self, platform_type: str) -> bool:
    """Returns boolean to indicate if device supports given Apple platform type."""
    if platform_type == "ios":
      return self.is_iphone() or self.is_ipad()
    elif platform_type == "tvos":
      return self.is_apple_tv()
    elif platform_type == "watchos":
      return self.is_apple_watch()
    elif platform_type == "visionos":
      return self.is_apple_vision()
    else:
      raise ValueError(
          f"Apple platform type not supported for simulator: {platform_type}."
      )

  def is_apple_tv(self) -> bool:
    return self.has_product_family_or_identifier("Apple TV")

  def is_apple_watch(self) -> bool:
    return self.has_product_family_or_identifier("Apple Watch")

  def is_apple_vision(self) -> bool:
    return self.has_product_family_or_identifier("Apple Vision")

  def is_iphone(self) -> bool:
    return self.has_product_family_or_identifier("iPhone")

  def is_ipad(self) -> bool:
    return self.has_product_family_or_identifier("iPad")

  def has_product_family_or_identifier(self, device_type: str) -> bool:
    product_family = self.get("productFamily")
    if product_family:
      return product_family == device_type
    # Some older simulators are missing `productFamily`. Try to guess from the
    # identifier.
    return device_type in self["identifier"]


class Device(collections.abc.Mapping):
  """Wraps the `device` dictionary from `simctl list -j`.

  Provides an ordering so booted devices > shutdown devices, delegating
  to `DeviceType` order when both devices have the same state.
  """

  def __init__(self, device, device_type):
    self.device = device
    self.device_type = device_type

  def is_shutdown(self):
    return self["state"] == "Shutdown"

  def is_booted(self):
    return self["state"] == "Booted"

  def __getitem__(self, name):
    return self.device[name]

  def __iter__(self):
    return iter(self.device)

  def __len__(self):
    return len(self.device)

  def __repr__(self):
    return self["name"] + "(" + self["udid"] + ")"

  def __lt__(self, other):
    if self.is_shutdown() and other.is_booted():
      return True
    elif self.is_booted() and other.is_shutdown():
      return False
    else:
      return self.device_type < other.device_type


def minimum_os_to_simctl_runtime_version(minimum_os: str) -> int:
  """Converts a minimum OS string to a simctl RuntimeVersion integer.

  Args:
    minimum_os: A string in the form '12.2' or '13.2.3'.

  Returns:
    An integer in the form 0xAABBCC, where AA is the major version, BB is
    the minor version, and CC is the micro version.
  """
  # Pad the minimum OS version to major.minor.micro.
  minimum_os_components = (minimum_os.split(".") + ["0"] * 3)[:3]
  result = 0
  for component in minimum_os_components:
    result = (result << 8) | int(component)
  return result


def discover_best_compatible_simulator(
    *,
    platform_type: str,
    simctl_path: str,
    minimum_os: str,
    sim_device: str,
    sim_identifier: str,
    sim_os_version: str,
) -> tuple[Optional[DeviceType], Optional[Device]]:
  """Discovers the best compatible simulator device type and device.

  Args:
    platform_type: The Apple platform type for the given *_application() target.
    simctl_path: The path to the `simctl` binary.
    minimum_os: The minimum OS version required by the *_application() target.
    sim_device: Optional name of the device type (e.g. "iPhone 8 Plus").
    sim_identifier: The identifier of the simulator (<uuid>).
    sim_os_version: Optional version of the Apple platform runtime (e.g.
      "13.2").

  Returns:
    A tuple (device_type, device) containing the DeviceType and Device
    of the best compatible simulator (might be None if no match was found).

  Raises:
    subprocess.SubprocessError: if `simctl list` fails or times out.
  """
  # The `simctl list` CLI provides only very basic case-insensitive description
  # matching search term functionality.
  #
  # This code needs to enforce a numeric floor on `minimum_os`, so it directly
  # parses the JSON output by `simctl list` instead of repeatedly invoking
  # `simctl list` with search terms.
  cmd = [simctl_path, "list", "-j"]
  with subprocess.Popen(cmd, stdout=subprocess.PIPE) as process:
    simctl_data = json.load(process.stdout)
    if process.wait() != os.EX_OK:
      raise subprocess.CalledProcessError(process.returncode, cmd)
  compatible_device_types = []
  minimum_runtime_version = minimum_os_to_simctl_runtime_version(minimum_os)
  # Prepare the device name for case-insensitive matching.
  sim_device = sim_device and sim_device.casefold()
  # `simctl list` orders device types from oldest to newest. Remember
  # the index of each device type to preserve that ordering when
  # sorting device types.
  for simctl_list_index, device_type in enumerate(simctl_data["devicetypes"]):
    device_type = DeviceType(device_type, simctl_list_index)
    if not device_type.supports_platform_type(platform_type):
      continue
    # Some older simulators are missing `maxRuntimeVersion`. Assume those
    # simulators support all OSes (even though it's not true).
    max_runtime_version = device_type.get("maxRuntimeVersion")
    if max_runtime_version and max_runtime_version < minimum_runtime_version:
      continue
    if sim_device and device_type["name"].casefold() != sim_device:
      continue
    compatible_device_types.append(device_type)
  compatible_device_types.sort()
  logger.debug(
      "Found %d compatible device types.", len(compatible_device_types)
  )
  compatible_runtime_identifiers = set()
  for runtime in simctl_data["runtimes"]:
    if not runtime["isAvailable"]:
      continue
    if sim_os_version and runtime["version"] != sim_os_version:
      continue
    compatible_runtime_identifiers.add(runtime["identifier"])
  compatible_devices = []
  for runtime_identifier, devices in simctl_data["devices"].items():
    if runtime_identifier not in compatible_runtime_identifiers:
      continue
    for device in devices:
      if not device["isAvailable"]:
        continue
      if sim_identifier:
        if device["udid"] != sim_identifier:
          continue
        compatible_device = Device(device, None)
        compatible_devices.append(compatible_device)
        break
      compatible_device = None
      for device_type in compatible_device_types:
        if device["deviceTypeIdentifier"] == device_type["identifier"]:
          compatible_device = Device(device, device_type)
          break
      if not compatible_device:
        continue
      compatible_devices.append(compatible_device)
  compatible_devices.sort()
  logger.debug("Found %d compatible devices.", len(compatible_devices))
  if not sim_identifier and compatible_device_types:
    best_compatible_device_type = compatible_device_types[-1]
  else:
    best_compatible_device_type = None
  if compatible_devices:
    best_compatible_device = compatible_devices[-1]
  else:
    best_compatible_device = None
  return (best_compatible_device_type, best_compatible_device)


def persistent_simulator(
    *,
    platform_type: str,
    simctl_path: str,
    minimum_os: str,
    sim_device: str,
    sim_identifier: str,
    sim_os_version: str,
) -> str:
  """Finds or creates a persistent compatible Apple simulator.

  Boots the simulator if needed. Does not shut down or delete the simulator when
  done.

  Args:
    platform_type: The Apple platform type for the given *_application() target.
    simctl_path: The path to the `simctl` binary.
    minimum_os: The minimum OS version required by the *_application() target.
    sim_device: Optional name of the device type (e.g. "iPhone 8 Plus").
    sim_identifier: The identifier of the simulator (<uuid>).
    sim_os_version: Optional version of the Apple platform runtime (e.g.
      "13.2").

  Returns:
    The UDID of the compatible Apple simulator.

  Raises:
    Exception: if a compatible simulator was not found.
  """
  (best_compatible_device_type, best_compatible_device) = (
      discover_best_compatible_simulator(
          platform_type=platform_type,
          simctl_path=simctl_path,
          minimum_os=minimum_os,
          sim_device=sim_device,
          sim_identifier=sim_identifier,
          sim_os_version=sim_os_version,
      )
  )
  if best_compatible_device:
    udid = best_compatible_device["udid"]
    if best_compatible_device.is_shutdown():
      logger.debug("Booting compatible device: %s", best_compatible_device)
      subprocess.run([simctl_path, "boot", udid], check=True)
    else:
      logger.debug("Using compatible device: %s", best_compatible_device)
    return udid
  if best_compatible_device_type:
    device_name = best_compatible_device_type["name"]
    device_id = best_compatible_device_type["identifier"]
    logger.info("Creating new %s simulator", device_name)
    create_result = subprocess.run(
        [simctl_path, "create", device_name, device_id],
        encoding="utf-8",
        stdout=subprocess.PIPE,
        check=True,
    )
    udid = create_result.stdout.rstrip()
    logger.debug("Created new simulator: %s", udid)
    return udid
  raise Exception(
      f"Could not find or create a simulator for the {platform_type} platform "
      f"compatible with minimum OS version {minimum_os} (uuid "
      f"'{sim_identifier}', device name '{sim_device}', OS version "
      f"'{sim_os_version}')"
  )


def wait_for_sim_to_boot(simctl_path: str, udid: str) -> bool:
  """Blocks until the given simulator is booted.

  Args:
    simctl_path: The path to the `simctl` binary.
    udid: The identifier of the simulator to wait for.

  Returns:
    True if the simulator boots within 60 seconds, False otherwise.
  """
  logger.info("Waiting for simulator to boot...")
  subprocess.run(
      [simctl_path, "bootstatus", udid, "-b"],
      encoding="utf-8",
      check=True,
  )
  return True


def boot_simulator(*, developer_path: str, simctl_path: str, udid: str) -> None:
  """Launches the Apple simulator for the given identifier.

  Ensures the Simulator process is in the foreground.

  Args:
    developer_path: The path to /Applications/Xcode.app/Contents/Developer.
    simctl_path: The path to the `simctl` binary.
    udid: The identifier of the simulator to wait for.

  Raises:
    Exception: if the simulator did not launch within 60 seconds.
  """
  logger.info("Launching simulator with udid: %s", udid)
  # Using subprocess.Popen() to launch Simulator.app and then
  # `osascript -e "tell application \"Simulator\" to activate" is racy
  # and can fail with:
  #
  #   Simulator got an error: Connection is invalid. (-609)
  #
  # This is likely because the newly-spawned Simulator.app process
  # hasn't had time to connect to the Apple Events system which
  # `osascript` relies on.
  simulator_path = os.path.join(developer_path, "Applications/Simulator.app")
  subprocess.run(
      ["open", "-a", simulator_path, "--args", "-CurrentDeviceUDID", udid],
      check=True,
  )
  logger.debug("Simulator launched.")
  if not wait_for_sim_to_boot(simctl_path, udid):
    raise Exception("Failed to launch simulator with UDID: " + udid)


@contextlib.contextmanager
def temporary_simulator(
    *, platform_type: str, simctl_path: str, device: str, version: str
) -> AppleSimulatorUDID:
  """Creates a temporary Apple simulator, cleaned up automatically upon close.

  Args:
    platform_type: The Apple platform type for the given *_application() target.
    simctl_path: The path to the `simctl` binary.
    device: The name of the device (e.g. "iPhone 8 Plus").
    version: The version of the Apple platform runtime (e.g. "13.2").

  Yields:
    The UDID of the newly-created Apple simulator.
  """
  runtime_version_name = version.replace(".", "-")
  # capitalizes 'os' from Apple platform type string (e.g. watchos -> watchOS)
  runtime_platform = platform_type[0:-2].lower() + platform_type[-2:].upper()
  logger.info("Creating simulator, device=%s, version=%s", device, version)
  simctl_create_result = subprocess.run(
      [
          simctl_path,
          "create",
          "TestDevice",
          device,
          "{prefix}.{runtime_platform}-{runtime_version_name}".format(
              prefix="com.apple.CoreSimulator.SimRuntime",
              runtime_platform=runtime_platform,
              runtime_version_name=runtime_version_name,
          ),
      ],
      encoding="utf-8",
      check=True,
      stdout=subprocess.PIPE,
  )
  udid = simctl_create_result.stdout.rstrip()
  try:
    logger.info("Killing all running simulators...")
    subprocess.run(
        ["pkill", "Simulator"], stderr=subprocess.DEVNULL, check=False
    )
    yield udid
  finally:
    logger.info("Shutting down simulator with udid: %s", udid)
    subprocess.run(
        [simctl_path, "shutdown", udid], stderr=subprocess.DEVNULL, check=False
    )
    logger.info("Deleting simulator with udid: %s", udid)
    subprocess.run([simctl_path, "delete", udid], check=True)


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
) -> AppleSimulatorUDID:
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


def simctl_launch_environ() -> Dict[str, str]:
  """Calculates an environment dictionary for running `simctl launch`."""
  # Pass environment variables prefixed with "IOS_" to the simulator, replace
  # the prefix with "SIMCTL_CHILD_". bazel adds "IOS_" to the env vars which
  # will be passed to the app as prefix to differentiate from other env vars. We
  # replace the prefix "IOS_" with "SIMCTL_CHILD_" here, because "simctl" only
  # pass the env vars prefixed with "SIMCTL_CHILD_" to the app.
  result = {}
  for k, v in os.environ.items():
    if not k.startswith("IOS_"):
      continue
    new_key = k.replace("IOS_", "SIMCTL_CHILD_", 1)
    result[new_key] = v
  if "IDE_DISABLED_OS_ACTIVITY_DT_MODE" not in os.environ:
    # Ensure os_log() mirrors writes to stderr. (lldb and Xcode set this
    # environment variable as well.)
    result["SIMCTL_CHILD_OS_ACTIVITY_DT_MODE"] = "enable"
  return result


@contextlib.contextmanager
def apple_simulator(
    *,
    platform_type: str,
    simctl_path: str,
    minimum_os: str,
    sim_device: str,
    sim_identifier: str,
    sim_os_version: str,
) -> AppleSimulatorUDID:
  """Finds or creates a persistent compatible Apple simulator.

  Args:
    platform_type: The Apple platform type for the given *_application() target.
    simctl_path: The path to the `simctl` binary.
    minimum_os: The minimum OS version required by the *_application() target.
    sim_device: Optional name of the device type (e.g. "iPhone 8 Plus").
    sim_identifier: The identifier of the simulator (<uuid>).
    sim_os_version: Optional version of the Apple platform runtime (e.g.
      "13.2").

  Yields:
    The UDID of the simulator.
  """
  prefer_persistent = os.environ.get("BAZEL_APPLE_PREFER_PERSISTENT_SIMS", "0") == "1"
  if not prefer_persistent and sim_device and sim_os_version:
    with temporary_simulator(
        platform_type=platform_type,
        simctl_path=simctl_path,
        device=sim_device,
        version=sim_os_version,
    ) as udid:
      yield udid
  else:
    yield persistent_simulator(
        platform_type=platform_type,
        simctl_path=simctl_path,
        minimum_os=minimum_os,
        sim_device=sim_device,
        sim_identifier=sim_identifier,
        sim_os_version=sim_os_version,
    )


def run_app_in_simulator(
    *,
    simulator_udid: str,
    developer_path: str,
    simctl_path: str,
    application_output_path: str,
    app_name: str,
) -> None:
  """Installs and runs an app in the specified simulator.

  Args:
    simulator_udid: The UDID of the simulator in which to run the app.
    developer_path: The path to /Applications/Xcode.app/Contents/Developer.
    simctl_path: The path to the `simctl` binary.
    application_output_path: Path to the output of an `*_application()`.
    app_name: The name of the application (e.g. "Foo" for "Foo.app").
  """
  boot_simulator(
      developer_path=developer_path,
      simctl_path=simctl_path,
      udid=simulator_udid,
  )
  root_dir = os.path.dirname(application_output_path)
  register_dsyms(root_dir)
  with extracted_app(application_output_path, app_name) as app_path:
    app_bundle_id = bundle_id(app_path)
    logger.info("Will install app %s to simulator %s", app_path, simulator_udid)
    # First, quietly kill any existing instances of the app to match Xcode's behavior.
    # Otherwise we've observed that the simulator gets confused when trying to re-install the app.
    logger.debug(
        "Terminating existing instances of %s in %s", app_bundle_id, simulator_udid
    )
    subprocess.run(
        [simctl_path, "terminate", simulator_udid, app_bundle_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # We should now be able to install and run it.
    logger.debug("Installing...")
    subprocess.run(
        [simctl_path, "install", simulator_udid, app_path],
        check=True,
    )
    launch_args = shlex.split(
      os.environ.get(
        "BAZEL_SIMCTL_LAUNCH_FLAGS",
        # Attaches the application to the console and waits for it to exit.
        "--console-pty",
      ),
    )
    logger.info(
        "Launching app %s in simulator %s", app_bundle_id, simulator_udid
    )
    args = [
        simctl_path,
        "launch",
        *launch_args,
        simulator_udid,
        app_bundle_id,
    ]
    # Append optional launch arguments.
    args.extend(sys.argv[1:])
    launch_app(args, env=simctl_launch_environ(), simulator_udid=simulator_udid)


def launch_app(
    args: Sequence[str],
    *,
    env: Dict[str, str],
    simulator_udid: str,
) -> None:
  """Launches an app in a simulator.

  Args:
    args: The arguments to pass to simctl.
    env: The environment variables to pass to simctl.
    simulator_udid: The UDID of the simulator in which to run the app.
  """
  launch_info_path = os.environ.get("BAZEL_APPLE_LAUNCH_INFO_PATH")
  if not launch_info_path:
    subprocess.run(args, env=env, check=True)
    return

  # Open a PTY to capture the output of simctl. We need a PTY to ensure that
  # the PID is written to stdout before the rest of the app output.
  primary_fd, secondary_fd = pty.openpty()

  proc = subprocess.Popen(
      args,
      env=env,
      stdout=secondary_fd,
      close_fds=True,
  )

  # simctl has the fd dup; close ours.
  os.close(secondary_fd)

  with os.fdopen(primary_fd, "rb", buffering=0) as r:
    # Grab PID from the first line of output.
    first_line = r.readline()
    pid_match = re.search(rb":\s*(\d+)\s*$", first_line)
    if pid_match:
      pid = int(pid_match.group(1))
      try:
        os.makedirs(os.path.dirname(launch_info_path), exist_ok=True)
        with open(launch_info_path, "w", encoding="utf-8") as f:
          f.write(json.dumps(
              {
                  "platform": "ios-simulator",
                  "udid": simulator_udid,
                  "pid": pid,
              },
              indent=2,
          ))
      except Exception as e:
        logger.error("Failed to write launch info to file: %s", e)
    else:
      logger.error("Failed to parse PID from output")

    # Stream the rest until simctl exits.
    sys.stdout.buffer.write(first_line)
    sys.stdout.flush()
    shutil.copyfileobj(r, BufferFlusher(sys.stdout.buffer))

  exit_code = proc.wait()
  if exit_code != 0:
    raise subprocess.CalledProcessError(exit_code, args)


def main(
    *,
    app_name: str,
    application_output_path: str,
    minimum_os: str,
    platform_type: str,
    sim_device: str,
    sim_identifier: str,
    sim_os_version: str,
):
  """Main entry point to `bazel run` for *_application() targets.

  Args:
    app_name: The name of the application (e.g. "Foo" for "Foo.app").
    application_output_path: Path to the output of an *_application().
    minimum_os: The minimum OS version required by the *_application() target.
    platform_type: The Apple platform type for the given *_application() target.
    sim_device: The name of the device type (e.g. "iPhone 8 Plus").
    sim_identifier: The identifier of the simulator (<uuid>).
    sim_os_version: The version of the Apple platform runtime (e.g. "13.2").
  """
  xcode_select_result = subprocess.run(
      ["xcode-select", "-p"],
      encoding="utf-8",
      check=True,
      stdout=subprocess.PIPE,
  )
  developer_path = xcode_select_result.stdout.rstrip()
  simctl_path = os.path.join(developer_path, "usr", "bin", "simctl")

  with apple_simulator(
      platform_type=platform_type,
      simctl_path=simctl_path,
      minimum_os=minimum_os,
      sim_device=sim_device,
      sim_identifier=sim_identifier,
      sim_os_version=sim_os_version,
  ) as simulator_udid:
    run_app_in_simulator(
        simulator_udid=simulator_udid,
        developer_path=developer_path,
        simctl_path=simctl_path,
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
        sim_device="%sim_device%",
        sim_identifier="%sim_identifier%",
        sim_os_version="%sim_os_version%",
    )
  except subprocess.CalledProcessError as e:
    logger.error("%s exited with error code %d", e.cmd, e.returncode)
    sys.exit(e.returncode)
  except KeyboardInterrupt:
    sys.exit(1)
