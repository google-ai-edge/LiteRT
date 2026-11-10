// Copyright 2026 Google LLC.
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

#ifndef ODML_LITERT_LITERT_TOOLS_PARTITION_REPORT_H_
#define ODML_LITERT_LITERT_TOOLS_PARTITION_REPORT_H_

#include <string>

#include "tflite/interpreter.h"

namespace litert {
namespace tools {

// Returns a human-readable description of how the primary subgraph of
// `interpreter` is split between delegate kernels and nodes that fell back to
// the CPU interpreter. The report contains:
//   1. One line per entry of the execution plan: either
//      DELEGATE(<delegate name>) with the number of original nodes it replaced,
//      or the builtin/custom op name of a fallback node.
//   2. A summary: number of delegate partitions, number of fallback nodes, and
//      the fallback nodes grouped into contiguous "islands" with an opcode
//      histogram per island.
//   3. Boundary traffic: for every delegate partition, the bytes of its input
//      tensors that are produced by fallback nodes (or are graph inputs) and
//      of its output tensors consumed by fallback nodes (or graph outputs).
//      Constant tensors are not counted. Tensors handed directly from one
//      delegate partition to another are reported separately as
//      delegate->delegate traffic. Dynamic tensors are counted separately
//      since their size is only known after Invoke().
//   4. A single machine-friendly line, e.g.
//      PARTITION_REPORT partitions=3 delegated_nodes=61 fallback_nodes=7
//      boundary_bytes=1234567 inter_partition_bytes=0
//      dynamic_boundary_tensors=0
//
// Call this after all delegates have been applied. Byte counts use the static
// tensor shapes from the model (or the allocated sizes if AllocateTensors()
// has already run); dynamic tensors are only counted, not sized.
std::string GeneratePartitionReport(const tflite::Interpreter& interpreter);

}  // namespace tools
}  // namespace litert

#endif  // ODML_LITERT_LITERT_TOOLS_PARTITION_REPORT_H_
