/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Local diagnostic for the staged Gemma attention runtime. This deliberately
// depends on the same XNNPACK internal headers as the linked local archives.
#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_ACTIVE_RUNTIME_AUDIT_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_ACTIVE_RUNTIME_AUDIT_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xnnpack.h"
#include "xnnpack/operator.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/subgraph.h"

namespace litert::tensor::examples::gemma4::native {

// Call after a successful attention Run(), when propagated shapes are current.
// query_rows is the folded row count (num_query_heads * token_rows), NOT the
// number of token rows. This audits exactly the following two BMMs:
//   QK: [1,1,query_rows,dim] x [1,1,depth,dim]^T
//   PV: [1,1,query_rows,depth] x [1,1,depth,dim]
// Both cache inputs remain signed INT8; Q, probabilities, and outputs are FP32.
//
// QS8->QC8 adapters are permitted. Current XNNPACK implements them with byte
// copies plus channelwise scale metadata, so this does not promise copy-free
// attention. It does rule out full-capacity operands, activation quantization,
// and materialized FP32 K/V on any executed edge of this attention runtime.
// Workspace intermediates may be reused after their last consumer. We inspect
// their types/shapes, never stale intermediate bytes after inference.
inline absl::Status AuditActiveAttention(xnn_runtime_t runtime, int query_rows,
                                         int depth, int dim) {
  if (!runtime || query_rows <= 0 || depth <= 0 || dim <= 0) {
    return absl::InvalidArgumentError(
        "Active attention audit requires a runtime and positive query_rows, "
        "depth, and dim");
  }
  if (!runtime->has_been_setup || !runtime->values || !runtime->opdata) {
    return absl::FailedPreconditionError(
        "Active attention audit must follow a successful runtime invocation");
  }
  const auto expected = [=](const xnn_runtime_value& value, size_t rows,
                             size_t columns) {
    const auto& shape = value.shape;
    return shape.num_dims == 4 && shape.dim[0] == 1 && shape.dim[1] == 1 &&
           shape.dim[2] == rows && shape.dim[3] == columns;
  };
  const auto int8 = [](xnn_datatype datatype) {
    return datatype == xnn_datatype_qint8 || datatype == xnn_datatype_qcint8;
  };
  size_t bmm_count = 0, qk_count = 0, pv_count = 0;
  for (size_t index = 0; index < runtime->num_ops; ++index) {
    const auto& op = runtime->opdata[index];
    size_t object_count = 0;
    for (const auto* object : op.operator_objects) {
      if (object) ++object_count;
    }
    // Removed nodes can retain bookkeeping entries without executing anything.
    if (object_count == 0) continue;
    const auto fail = [&](const std::string& detail) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Active attention operator ", index, ": ", detail,
          " (query_rows=", query_rows, ", depth=", depth, ", dim=", dim, ")"));
    };
    if (op.num_inputs > XNN_MAX_INPUTS || op.num_outputs > XNN_MAX_OUTPUTS) {
      return fail("invalid input/output arity");
    }
    bool int8_input = false, float_input = false;
    bool int8_output = false, float_output = false;
    for (uint32_t side = 0; side != 2; ++side) {
      const uint32_t count = side ? op.num_outputs : op.num_inputs;
      for (uint32_t edge = 0; edge < count; ++edge) {
        const uint32_t id = side ? op.outputs[edge] : op.inputs[edge];
        if (id >= runtime->num_values) return fail("invalid value ID");
        const auto datatype = runtime->values[id].datatype;
        if (datatype != xnn_datatype_fp32 && !int8(datatype)) {
          return fail(absl::StrCat(
              "unexpected datatype ", static_cast<int>(datatype),
              " on executed value ", id,
              "; dynamic/packed activation quantization is forbidden"));
        }
        if (side) {
          int8_output |= int8(datatype);
          float_output |= datatype == xnn_datatype_fp32;
        } else {
          int8_input |= int8(datatype);
          float_input |= datatype == xnn_datatype_fp32;
        }
      }
    }

    if (op.type == xnn_node_type_batch_matrix_multiply) {
      if (object_count != 1 || op.num_inputs != 2 || op.num_outputs != 1) {
        return fail("expected one operator object and two inputs for each BMM");
      }
      for (const auto* object : op.operator_objects) {
        if (object && object->type !=
                          xnn_operator_type_batch_matrix_multiply_nc_f32_qc8w) {
          return fail(absl::StrCat("expected F32/QC8W BMM, got ",
                                  xnn_operator_type_to_string(object->type)));
        }
      }
      const auto& lhs = runtime->values[op.inputs[0]];
      const auto& rhs = runtime->values[op.inputs[1]];
      const auto& output = runtime->values[op.outputs[0]];
      if (lhs.datatype != xnn_datatype_fp32 || !int8(rhs.datatype) ||
          output.datatype != xnn_datatype_fp32) {
        return fail("BMM must consume FP32 activation and INT8 cache, yielding FP32");
      }
      if (op.flags & XNN_FLAG_TRANSPOSE_A) {
        return fail("unexpected transposed query/probability operand");
      }
      if (!expected(rhs, depth, dim)) {
        return fail("cache operand must be exactly [1,1,depth,dim]");
      }
      if (op.flags & XNN_FLAG_TRANSPOSE_B) {
        if (!expected(lhs, query_rows, dim) ||
            !expected(output, query_rows, depth)) {
          return fail("QK dimensions differ from active query/cache extent");
        }
        ++qk_count;
      } else {
        if (!expected(lhs, query_rows, depth) ||
            !expected(output, query_rows, dim)) {
          return fail("PV dimensions differ from active probability/cache extent");
        }
        ++pv_count;
      }
      ++bmm_count;
      continue;
    }

    // A generic unary operator can implement dequantization without having
    // "Convert" in its operator name. Check the actual edge types as well.
    if ((int8_input && float_output) || (float_input && int8_output)) {
      return fail("non-BMM operation crosses the FP32/INT8 boundary");
    }
    for (const auto* object : op.operator_objects) {
      if (!object) continue;
      if (object->type == xnn_operator_type_convert_nc_qs8_qc8) {
        if (op.num_inputs != 1 || op.num_outputs != 1) {
          return fail("unexpected QS8->QC8 adapter arity");
        }
        const auto& src = runtime->values[op.inputs[0]];
        const auto& dst = runtime->values[op.outputs[0]];
        if (src.datatype != xnn_datatype_qint8 ||
            dst.datatype != xnn_datatype_qcint8 ||
            !expected(src, depth, dim) || !expected(dst, depth, dim)) {
          return fail("QS8->QC8 adapter must copy only the active INT8 extent");
        }
      } else {
        const std::string name = xnn_operator_type_to_string(object->type);
        if (name.find("Convert") != std::string::npos ||
            name.find("Batch Matrix Multiply") != std::string::npos) {
          return fail(absl::StrCat("unexpected arithmetic operator ", name));
        }
      }
    }
  }
  if (bmm_count != 2 || qk_count != 1 || pv_count != 1) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Active attention requires exactly one QK and one PV F32/QC8W BMM; got ",
        bmm_count, " BMMs, ", qk_count, " QK, ", pv_count, " PV"));
  }
  return absl::OkStatus();
}

}  // namespace litert::tensor::examples::gemma4::native
#endif  // LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_ACTIVE_RUNTIME_AUDIT_H_
