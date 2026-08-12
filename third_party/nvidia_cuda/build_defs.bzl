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

"""Open-source CUDA build rules backed by a local CUDA toolkit."""

def cuda_library(name, srcs, hdrs = [], deps = [], linkstatic = True, **kwargs):
    """Builds CUDA sources into one static archive.

    Args:
      name: Bazel target name.
      srcs: CUDA source labels.
      hdrs: Header labels needed while compiling the sources.
      deps: Additional tool dependencies for the CUDA action.
      linkstatic: Must be true because the output is a static archive.
      **kwargs: Additional attributes forwarded to the generated rule.
    """
    if not linkstatic:
        fail("cuda_library only supports linkstatic = True")
    archive = "lib%s.a" % name
    source_locations = " ".join(["$(location %s)" % src for src in srcs])
    native.genrule(
        name = name,
        srcs = srcs + hdrs,
        outs = [archive],
        cmd = " ".join([
            "NVCC=\"$$(readlink -f $(location @local_cuda//:bin/nvcc))\";",
            "ARCH_FLAGS=\"-gencode=arch=compute_80,code=sm_87 -gencode=arch=compute_80,code=compute_80\";",
            "case \"$$($$NVCC --list-gpu-code)\" in *sm_120*) ARCH_FLAGS=\"$$ARCH_FLAGS -gencode=arch=compute_120,code=sm_120\" ;; esac;",
            "\"$$NVCC\" --lib -std=c++17 -O3 -Xcompiler=-fPIC $$ARCH_FLAGS",
            "-I. -Iexternal/local_cuda/include",
            source_locations,
            "-o $@",
        ]),
        tools = deps + [
            "@local_cuda//:headers",
            "@local_cuda//:nvcc_tools",
            "@local_cuda//:bin/nvcc",
        ],
        **kwargs
    )
