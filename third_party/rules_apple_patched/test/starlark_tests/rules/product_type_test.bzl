# Copyright 2019 The Bazel Authors. All rights reserved.
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

"Starlark test for testing the product type."

load(
    "@bazel_skylib//lib:unittest.bzl",
    "analysistest",
    "unittest",
)
load(
    "//apple:providers.bzl",
    "AppleBundleInfo",
)

def _product_type_test_impl(ctx):
    "Test that the value of a product_type matches the expected one."
    env = analysistest.begin(ctx)
    target = analysistest.target_under_test(env)
    if not AppleBundleInfo in target:
        unittest.fail(env, "Could not read AppleBundleInfo.")
        return analysistest.end(env)
    bundle_info = target[AppleBundleInfo]

    if bundle_info.product_type != ctx.attr.expected_product_type:
        msg = "product_type ({}) did not match the expected value ({})."
        unittest.fail(env, msg.format(
            bundle_info.product_type,
            ctx.attr.expected_product_type,
        ))
    return analysistest.end(env)

product_type_test = analysistest.make(
    _product_type_test_impl,
    attrs = {
        "expected_product_type": attr.string(
            mandatory = True,
            doc = "The `apple_product_type` value that is expected.",
        ),
    },
)
