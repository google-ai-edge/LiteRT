# Copyright 2026 Google LLC.
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

"""Stand-in for @llvm-project//mlir:tblgen.bzl.

LiteRT does not fetch LLVM/MLIR. These macros only let BUILD files that mix
runtime and MLIR targets load; the MLIR targets fail if they are analyzed.
"""

def _unavailable_impl(ctx):
    fail("%s requires @llvm-project, which the LiteRT build does not provide." % ctx.label)

_unavailable = rule(implementation = _unavailable_impl)

def _stub(name, **kwargs):
    _unavailable(
        name = name,
        tags = ["manual"],
        testonly = kwargs.get("testonly", False),
        visibility = kwargs.get("visibility"),
    )

def gentbl_cc_library(name, **kwargs):
    _stub(name, **kwargs)

def gentbl_filegroup(name, **kwargs):
    _stub(name, **kwargs)

def td_library(name, **kwargs):
    _stub(name, **kwargs)
