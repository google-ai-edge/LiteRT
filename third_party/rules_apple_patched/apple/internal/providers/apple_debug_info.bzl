# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""AppleDebugInfo provider implementation for resource aspect and debug symbols partial support."""

visibility("//apple/internal/...")

AppleDebugInfo = provider(
    doc = """
Private provider to propagate transitive dSYM and link maps information used by the debug symbols
partial and the resource aspect.
""",
    fields = {
        "dsyms": """
Depset of `File` references to dSYM files if requested in the build with --apple_generate_dsym.
""",
        "linkmaps": """
Depset of `File` references to linkmap files if requested in the build with --objc_generate_linkmap.
""",
    },
)
