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

load("//research/aimatter/testing:xeno_lab.bzl", "xeno_mobile_test")
load("//testing/integration/guitar/build_defs:guitar_workflow.bzl", "guitar", "guitar_workflow_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/tensorflow:license"],
    default_visibility = ["//litert:__subpackages__"],
)

# @unused
DOC = """
Instructions to run mobile test locally (for testing via Mobile Harness):

Command:
blaze test --config=android_arm64 --copt="-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1" --android_ndk_min_sdk_version=26 \
--cache_test_results=no --test_output=streamed --notest_loasd --test_arg=--allocation_exit_strategy=FAIL_FAST_NO_MATCH \
--test_arg=--run_as=mobileiq-user //litert/google/mobile_tests:litert_dispatch_delegate_google_tensor_test

To run on locally connected device add these flags:
--test_arg=--adb_dont_kill_server --test_arg=--exec_mode=local --test_arg=--dimensions=""
"""

xeno_mobile_test(
    name = "litert_dispatch_delegate_google_tensor_test",
    args = [
        "--run_as=xeno-mh-guitar",
    ],
    dimensions = {
        "label": "odml-test",
        "model": "pixel 9",
    },
    env_vars = "LD_LIBRARY_PATH=/data/local/tmp/runfiles/third_party/odml/litert/litert/vendors/google_tensor/dispatch",
    run_as_top_app = False,
    tags = [
        "android",
        "notap",
    ],
    test_data = [
        "//litert/test:testdata/simple_model_google_tensor.bin",
        "//litert/test:testdata/simple_model_npu.tflite",
        "//litert/vendors/google_tensor/dispatch:libLiteRtDispatch_GoogleTensor.so",
    ],
    test_target = "//litert/runtime/dispatch:dispatch_delegate_google_tensor_test",
)

xeno_mobile_test(
    name = "litert_dispatch_delegate_qualcomm_test",
    args = [
        "--run_as=xeno-mh-guitar",
    ],
    dimensions = {
        "label": "odml-test",
        "model": "sm-s928b",
    },
    env_vars = "LD_LIBRARY_PATH=/data/local/tmp/runfiles/third_party/odml/litert/litert/vendors/qualcomm/dispatch:/data/local/tmp/runfiles/third_party/qairt/latest",
    run_as_top_app = False,
    tags = [
        "android",
        "notap",
    ],
    test_data = [
        "//litert/test:testdata/simple_model_npu.tflite",
        "//litert/test:testdata/simple_model_qualcomm.bin",
        "//litert/vendors/qualcomm/dispatch:libLiteRtDispatch_Qualcomm.so",
        "//third_party/qairt/latest:qnn_android_arm64_prebuilt_libs",
    ],
    test_target = "//litert/runtime/dispatch:dispatch_delegate_qualcomm_test",
)

# guitar run -w //litert/google/mobile_tests:litert_guitar_workflow -v citc:LOCAL
guitar_workflow_test(
    name = "litert_guitar_workflow",
    auto_deps = [":litert_dispatch_delegate_google_tensor_test"],
    integration_test = guitar.IntegrationTest(tests = [
        guitar.Tests(
            args = [
                "--allocation_exit_strategy=FAIL_FAST_NO_MATCH",
                "--dimension_label=odml-test",  # To run on ODML test lab.
                "--dimension_model=\"pixel 9\"",
            ],
            blaze_flags = [
                "--config=android_arm64",
                "--copt=-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1",
                "--android_ndk_min_sdk_version=26",
            ],
            execution_method = "DISTRIBUTED_ON_BORG",
            targets = [
                ":litert_dispatch_delegate_google_tensor_test",
            ],
        ),
    ]),
)
