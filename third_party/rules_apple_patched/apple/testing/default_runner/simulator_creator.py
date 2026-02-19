#!/usr/bin/python3
# Copyright 2022 The Bazel Authors. All rights reserved.
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

import argparse
import json
import random
import string
import subprocess
import sys
import time
from typing import List, Optional


def _simctl(extra_args: List[str]) -> str:
    return subprocess.check_output(["xcrun", "simctl"] + extra_args).decode()


def _boot_simulator(simulator_id: str) -> None:
    # This private command boots the simulator if it isn't already, and waits
    # for the appropriate amount of time until we can actually run tests
    try:
        output = _simctl(["bootstatus", simulator_id, "-b"])
        print(output, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode

        # When reusing simulators we may encounter the error:
        # 'Unable to boot device in current state: Booted'.
        #
        # This is because the simulator is already booted, and we can ignore it
        # if we check and the simulator is in fact booted.
        if exit_code == 149:
            devices = json.loads(
                _simctl(["list", "devices", "-j", simulator_id]),
            )["devices"]
            device = next(
                (
                    blob
                    for devices_for_os in devices.values()
                    for blob in devices_for_os
                    if blob["udid"] == simulator_id
                ),
                None
            )
            if device and device["state"].lower() == "booted":
                print(
                    f"Simulator '{device['name']}' ({simulator_id}) is already booted",
                    file=sys.stderr,
                )
                exit_code = 0

        # Both of these errors translate to strange simulator states that may
        # end up causing issues, but attempting to actually use the simulator
        # instead of failing at this point might still succeed
        #
        # 164: EBADDEVICE
        # 165: EBADDEVICESTATE
        if exit_code in (164, 165):
            print(
                f"Ignoring 'simctl bootstatus' exit code {exit_code}",
                file=sys.stderr,
            )
        elif exit_code != 0:
            print(f"'simctl bootstatus' exit code {exit_code}", file=sys.stderr)
            raise

    # Add more arbitrary delay before tests run. Even bootstatus doesn't wait
    # long enough and tests can still fail because the simulator isn't ready
    time.sleep(3)


def _device_name(device_type: str, os_version: str) -> str:
    return f"BAZEL_TEST_{device_type}_{os_version}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "os_version", help="The iOS version to run the tests on, ex: 12.1"
    )
    parser.add_argument(
        "device_type", help="The iOS device to run the tests on, ex: iPhone X"
    )
    parser.add_argument(
        "--name",
        required=False,
        default=None,
        help="The name to use for the device; default is 'BAZEL_TEST_[device_type]_[os_version]'",
    )
    parser.add_argument(
        "--reuse-simulator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle simulator reuse; default is True",
    )
    return parser


def _main(os_version: str, device_type: str, name: Optional[str], reuse_simulator: bool) -> None:
    devices = json.loads(_simctl(["list", "devices", "-j"]))["devices"]
    device_name = name or _device_name(device_type, os_version)
    runtime_identifier = "com.apple.CoreSimulator.SimRuntime.iOS-{}".format(
        os_version.replace(".", "-")
    )

    existing_device=None

    if reuse_simulator:
        devices_for_os = devices.get(runtime_identifier) or []
        existing_device = next(
            (blob for blob in devices_for_os if blob["name"] == device_name), None
        )

    if existing_device:
        simulator_id = existing_device["udid"]
        name = existing_device["name"]
        # If the device is already booted assume that it was created with this
        # script and bootstatus has already waited for it to be in a good state
        # once
        state = existing_device["state"].lower()
        print(f"Existing simulator '{name}' ({simulator_id}) state is: {state}", file=sys.stderr)
        if state != "booted":
            _boot_simulator(simulator_id)
    else:
        if not reuse_simulator:
            # Simulator reuse is based on device name, therefore we must generate a unique name to
            # prevent unintended reuse.
            device_name_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            device_name += f"_{device_name_suffix}"

        simulator_id = _simctl(
            ["create", device_name, device_type, runtime_identifier]
        ).strip()
        print(f"Created new simulator '{device_name}' ({simulator_id})", file=sys.stderr)
        _boot_simulator(simulator_id)

    print(simulator_id.strip())


if __name__ == "__main__":
    args = _build_parser().parse_args()
    _main(args.os_version, args.device_type, args.name, args.reuse_simulator)
