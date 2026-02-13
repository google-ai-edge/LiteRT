# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The exit codes of test runner."""


def enum(**enums):
  return type('Enum', (), enums)

EXITCODE = enum(
    SUCCEEDED=0,
    ERROR=1,
    UNKNOWN=10,
    FAILED=11,
    TEST_NOT_START=12,
    NEED_REBOOT_DEVICE=13,
    NEED_RECREATE_SIM=14,
    SIM_ERROR=15)

EXITCODE_INFOS = {
    EXITCODE.SUCCEEDED: 'Test succeed',
    EXITCODE.UNKNOWN: 'Unknown test result',
    EXITCODE.ERROR: 'General error',
    EXITCODE.FAILED: 'Test failure',
    EXITCODE.TEST_NOT_START: 'Test has not started',
    EXITCODE.NEED_REBOOT_DEVICE: 'Need reboot the device to recover it',
    EXITCODE.NEED_RECREATE_SIM: 'Need recreate a new simulator to run test',
    EXITCODE.SIM_ERROR: 'The simulator has error'}
