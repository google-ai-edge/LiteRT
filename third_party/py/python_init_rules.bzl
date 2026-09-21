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

"""Hermetic Python initialization. Consult the WORKSPACE on how to use it."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def python_init_rules(extra_patches = []):
    """Defines (doesn't setup) the rules_python repository."""

    tf_http_archive(
        name = "rules_python",
        sha256 = "e11d2e1efce1589e5bdfa93986712c74fc7467a0f093143d489d2ef5ebb1ed2a",
        strip_prefix = "rules_python-2.2.0",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_python/releases/download/2.2.0/rules_python-2.2.0.tar.gz"),
        patch_file = [
            "//third_party/py:rules_python_scope.patch",
            "//third_party/py:rules_python_freethreaded.patch",
        ] + [p for p in extra_patches if "site_init_retry" not in str(p)],
    )
