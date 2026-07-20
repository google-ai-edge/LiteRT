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

"""Generates an identifier from a subset of files in the MLDrift project.

This generates a fingerprint of the MLDrift library source files. This is done
before the files are compiled. So any change to the source files even if it
does not affect the compiled binary will cause the fingerprint to change.
"""

import argparse
import hashlib
import sys
import textwrap

parser = argparse.ArgumentParser(
    prog="MLDriftExternalTensorWeightRearrangementFingerprint",
    description=(
        "Generates a C source file that defines a function that returns a"
        " fingerprint of the given MLDrift source files and writes it to the"
        " output."
    ),
)
parser.add_argument(
    "--output", required=True, action="store", help="Set the output"
)
parser.add_argument(
    "inputs",
    nargs="+",
    help="The source files to use to generate the fingerprint.",
)

FILE_TEMPLATE = """// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Auto-generated file. Do not edit!
//   Generator: third_party/odml/litert/ml_drift/delegate/serialization_weight_cache/generate_build_identifier.py

#include "ml_drift_delegate/delegate/serialization_weight_cache/build_identifier.h"

#include <cstdint>
#include <cstring>

#include "absl/types/span.h"

namespace ml_drift {{

namespace {{

static const uint8_t build_identifier[] = {{
{}
}};

}}  // namespace

absl::Span<const uint8_t> GetBuildIdentifier() {{
  return absl::MakeSpan(build_identifier, sizeof(build_identifier));
}}

bool CheckBuildIdentifier(absl::Span<const uint8_t> identifier) {{
  if (identifier.size() != sizeof(build_identifier)) {{
    return false;
  }}
  return !memcmp(identifier.data(),build_identifier, identifier.size());
}}

}}  // namespace ml_drift

"""


def main(args) -> None:
  m = hashlib.sha256()
  for path in args.inputs:
    if any(path.endswith(ext) for ext in [".c", ".S", ".cc", ".h"]):
      with open(path, "rb") as f:
        m.update(f.read())
    else:
      print(
          "generate_external_tensor_build_identifier.py: Unknown file"
          f" extension for {path}. File was ignored.",
          file=sys.stderr,
      )
  byte_list = ", ".join(str(b).rjust(3, "x") for b in m.digest())
  byte_list = textwrap.indent(textwrap.fill(byte_list, width=40), "  ").replace(
      "x", " "
  )
  with open(args.output, "w") as out:
    out.write(FILE_TEMPLATE.format(byte_list))


if __name__ == "__main__":
  main(parser.parse_args())
