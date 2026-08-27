<!-- Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# Intel OpenVINO NPU Support for LiteRT

See the [LiteRT Intel dev page](https://ai.google.dev/edge/litert/next/intel)
for details.

## Changing the OpenVINO package

The OpenVINO package used to build LiteRT (Bazel) and the one bundled by the
`ai_edge_litert_sdk_intel` pip package (setup.py, at install time) are both
pinned from a single source of truth:
`third_party/intel_openvino/openvino_version.bzl`. This file is generated;
do not hand-edit it.

To bump the OpenVINO build, or switch between the `release` and `nightly`
channel, copy the archive URL for the **Windows** package from
[storage.openvinotoolkit.org](https://storage.openvinotoolkit.org) and run:

```sh
python ci/tools/update_openvino_version.py \
    --windows-url '<windows package archive url>' \
    --validate
```

This derives the matching ubuntu24 / ubuntu22 / android URLs and the PEP 440
version, checks that all four archives are reachable (`--validate`), and
regenerates `openvino_version.bzl`. Commit the regenerated file.

`python ci/tools/update_openvino_version.py --check` re-derives the file from
its own committed Windows URL and fails if it doesn't match — this is the CI
guardrail against hand edits or partial updates.

Notes:

- To build against a local OpenVINO SDK instead of downloading one, set
  `OPENVINO_NATIVE_DIR` to an absolute path; it takes precedence over
  `openvino_version.bzl` (see `litert/sdk_util/repo.bzl`).
- If the pinned channel is `nightly`, installing the pip SDK also requires
  `--extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly`
  so pip can locate the nightly `openvino` wheel; `release` versions are on
  PyPI directly.

