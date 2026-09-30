# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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
# ==============================================================================

"""Repository rule to create a platform for a docker image to be used with RBE."""

def _rbe_platform_impl(repository_ctx):
    exec_properties = repository_ctx.attr.platform_exec_properties
    serialized_exec_properties = "{\n"
    for k, v in exec_properties.items():
        serialized_exec_properties += '        "%s": "%s",\n' % (k, v)
    serialized_exec_properties += "    }"

    cpu = "x86_64"
    platform = repository_ctx.attr.platform

    build_content = """# Platform definition for RBE container
package(default_visibility = ["//visibility:public"])

platform(
    name = "platform",
    constraint_values = [
        "@platforms//cpu:{cpu}",
        "@platforms//os:{platform}",
    ],
    exec_properties = {exec_properties},
)
""".format(
        cpu = cpu,
        platform = platform,
        exec_properties = serialized_exec_properties,
    )

    repository_ctx.file("BUILD.bazel", build_content)

rbe_platform = repository_rule(
    implementation = _rbe_platform_impl,
    attrs = {
        "platform_exec_properties": attr.string_dict(mandatory = True),
        "platform": attr.string(default = "linux"),
    },
)

def ml_build_rbe_platform(
        name = "ml_build_config_platform",
        container_image = "docker://us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build@sha256:ea67e8453d8b09c2ba48853da5e79efef4b65804b4a48dfae4b4da89ffd38405"):
    rbe_platform(
        name = name,
        platform = "linux",
        platform_exec_properties = {
            "container-image": container_image,
            "Pool": "default",
        },
    )
