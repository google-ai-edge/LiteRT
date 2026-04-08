// Copyright 2024 Google LLC.
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

#include "litert/tools/dump.h"

#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"

#if !defined(LITERT_WINDOWS_OS)
#include <dlfcn.h>
#endif  // !defined(LITERT_WINDOWS_OS)

#ifndef __ANDROID__
#if __has_include(<link.h>)
#include <link.h>
#endif
#endif

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/internal/litert_c_types_printing.h"
#include "litert/compiler/plugin/compiler_plugin.h"
#include "litert/core/model/model.h"

namespace litert::internal {

namespace {

static constexpr int kMaxDisplayCount = 16;

void DumpNode(const LiteRtTensorT& tensor, std::ostream& out) {
  switch (tensor.Type().first) {
    case kLiteRtRankedTensorType:
      Dump(tensor.Type().second.ranked_tensor_type, out);
      break;
    case kLiteRtUnrankedTensorType:
      Dump(tensor.Type().second.unranked_tensor_type.element_type, out);
      break;
    default:
      out << "UKNOWN_TENSOR_TYPE" << tensor.Type().first;
  }
  Dump(tensor.Qparams(), out);
}

void DumpNode(const LiteRtOpT& op, std::ostream& out) {
  Dump(op.OpCode(), out);
}

void DumpSignature(const std::vector<LiteRtTensor>& ins,
                   const std::vector<LiteRtTensor>& outs, std::ostream& out) {
  out << "(";
  for (auto it = ins.begin(); it < ins.end(); ++it) {
    DumpNode(**it, out);
    if (it != ins.end() - 1) {
      out << ", ";
    }
  }
  out << ")";

  out << " -> ";
  const bool paren_outs = outs.size() != 1;
  if (paren_outs) {
    out << "(";
  }
  for (auto it = outs.begin(); it < outs.end(); ++it) {
    DumpNode(**it, out);
    if (it != outs.end() - 1) {
      out << ", ";
    }
  }
  if (paren_outs) {
    out << ")";
  }
}

}  // namespace

void Dump(LiteRtOpCode code, std::ostream& out) {
  out << absl::StreamFormat("%v", code);
}

// Dump details about the given LiteRtElementType to the given stream.
void Dump(LiteRtElementType type, std::ostream& out) {
  switch (type) {
    case kLiteRtElementTypeFloat32:
      out << ::litert::kElementTypeStrF32;
      break;
    case kLiteRtElementTypeInt32:
      out << ::litert::kElementTypeStrI32;
      break;
    case kLiteRtElementTypeFloat64:
      out << ::litert::kElementTypeStrF64;
      break;
    case kLiteRtElementTypeInt64:
      out << ::litert::kElementTypeStrI64;
      break;
    case kLiteRtElementTypeFloat16:
      out << ::litert::kElementTypeStrF16;
      break;
    case kLiteRtElementTypeInt16:
      out << ::litert::kElementTypeStrI16;
      break;
    case kLiteRtElementTypeInt8:
      out << ::litert::kElementTypeStrI8;
      break;
    case kLiteRtElementTypeUInt8:
      out << ::litert::kElementTypeStrU8;
      break;
    case kLiteRtElementTypeInt4:
      out << ::litert::kElementTypeStrI4;
      break;
    case kLiteRtElementTypeInt2:
      out << ::litert::kElementTypeStrI2;
      break;
    case kLiteRtElementTypeBool:
      out << ::litert::kElementTypeStrI1;
      break;
    default:
      out << "UKNNOWN_ELEMENT_TYPE: " << type;
  }
}

void Dump(const LiteRtRankedTensorType& type, std::ostream& out) {
  out << "<";
  for (int i = 0; i < type.layout.rank; ++i) {
    out << type.layout.dimensions[i] << "x";
  }
  Dump(type.element_type, out);
  out << ">";
}

void Dump(const LiteRtTensorT& tensor, std::ostream& out) {
  out << "LiteRtTensor : ";
  DumpNode(tensor, out);
  out << " [ ";
  if (tensor.DefiningOp() == nullptr) {
    out << "*";
  } else {
    DumpNode(*tensor.DefiningOp(), out);
  }
  out << " ] ";

  out << "(";
  for (auto it = tensor.Users().begin(); it < tensor.Users().end(); ++it) {
    DumpNode(**it, out);
    if (it != tensor.Users().end() - 1) {
      out << ", ";
    }
  }
  out << ")";
  out << "\n";
}

void Dump(const LiteRtOpT& op, std::ostream& out) {
  out << "LiteRtOp : [ ";
  DumpNode(op, out);
  if (auto custom_code = op.CustomCode(); custom_code.HasValue()) {
    out << " : " << std::string(*custom_code);
  }
  out << " ] ";
  DumpSignature(op.Inputs(), op.Outputs(), out);
  out << "\n";
}

void Dump(const LiteRtSubgraphT& subgraph, std::ostream& out) {
  constexpr absl::string_view kSubgraphTpl =
      "LiteRtSubgraph : [ #ops=%d #tensors=%d ] ";
  out << absl::StreamFormat(kSubgraphTpl, subgraph.Ops().size(),
                            subgraph.Tensors().size());
  DumpSignature(subgraph.Inputs(), subgraph.Outputs(), out);
  out << "\n";
}

void Dump(const CompilerPlugin& plugin, std::ostream& out) {
  constexpr absl::string_view kPluginDumpTpl =
      "SocManufacturer: %s\nSocModels: { ";
  out << absl::StreamFormat(kPluginDumpTpl, plugin.SocManufacturer());

  for (auto it = plugin.SocModels().begin(); it < plugin.SocModels().end();
       ++it) {
    out << *it;
    if (it != plugin.SocModels().end() - 1) {
      out << ",";
    }
    out << " ";
  }

  out << "}\n";
}

void Dump(const LiteRtModelT& model, std::ostream& out) {
  out << absl::StreamFormat("LiteRtModel : [ #subgraphs=%d ]\n",
                            model.Subgraphs().size());
}

void DumpOptions(const LiteRtOpT& op, std::ostream& out) {
  auto& opts = litert::internal::GetTflOptions(op);
  if (opts.value == nullptr) {
    out << "null options\n";
    return;
  }
  switch (op.OpCode()) {
    case kLiteRtOpCodeTflAdd:
      out << "fused_activation_function: "
          << opts.AsAddOptions()->fused_activation_function << "\n";
      break;
    case kLiteRtOpCodeTflMul:
      out << "fused_activation_function: "
          << opts.AsMulOptions()->fused_activation_function << "\n";
      break;
    case kLiteRtOpCodeTflBatchMatmul:
      out << "adj_x: " << opts.AsBatchMatMulOptions()->adj_x << "\n";
      out << "adj_y: " << opts.AsBatchMatMulOptions()->adj_y << "\n";
      out << "asymmetric_quantize_input: "
          << opts.AsBatchMatMulOptions()->asymmetric_quantize_inputs << "\n";
      break;
    case kLiteRtOpCodeTflConcatenation:
      out << "axis: " << opts.AsConcatenationOptions()->axis << "\n";
      out << "fused_activation_function: "
          << opts.AsConcatenationOptions()->fused_activation_function << "\n";
      break;
    case kLiteRtOpCodeTflDiv:
      out << "fused_activation_function: "
          << opts.AsDivOptions()->fused_activation_function << "\n";
      break;
    case kLiteRtOpCodeTflFullyConnected:
      out << "weights_format: "
          << opts.AsFullyConnectedOptions()->weights_format << "\n";
      out << "keep_num_dims: " << opts.AsFullyConnectedOptions()->keep_num_dims
          << "\n";
      out << "quantized_bias_type: "
          << opts.AsFullyConnectedOptions()->quantized_bias_type << "\n";
      out << "asymmetric_quantize_input: "
          << opts.AsFullyConnectedOptions()->asymmetric_quantize_inputs << "\n";
      out << "fused_activation_function: "
          << opts.AsFullyConnectedOptions()->fused_activation_function << "\n";
      break;
    case kLiteRtOpCodeTflSoftmax:
      out << "beta: " << opts.AsSoftmaxOptions()->beta << "\n";
      break;
    case kLiteRtOpCodeTflStridedSlice:
      out << "begin_mask: " << opts.AsStridedSliceOptions()->begin_mask << "\n";
      out << "end_mask: " << opts.AsStridedSliceOptions()->end_mask << "\n";
      out << "ellipsis_mask: " << opts.AsStridedSliceOptions()->ellipsis_mask
          << "\n";
      out << "new_axis_mask: " << opts.AsStridedSliceOptions()->new_axis_mask
          << "\n";
      out << "shrink_axis_mask: "
          << opts.AsStridedSliceOptions()->shrink_axis_mask << "\n";
      out << "offset: " << opts.AsStridedSliceOptions()->offset << "\n";
      break;
    case kLiteRtOpCodeTflSub:
      out << "fused_activation_function: "
          << opts.AsSubOptions()->fused_activation_function << "\n";
      break;
    case kLiteRtOpCodeTflReshape:
      out << "new_shape: ";
      if (opts.AsReshapeOptions() != nullptr) {
        const int32_t* new_shape = opts.AsReshapeOptions()->new_shape.data();
        int32_t new_shape_size = opts.AsReshapeOptions()->new_shape.size();
        for (int i = 0; i < new_shape_size; ++i) {
          out << new_shape[i] << " ";
        }
      }
      break;
    case kLiteRtOpCodeTflSum:
      out << "keepdims: " << opts.AsReducerOptions()->keep_dims << "\n";
      break;
    case kLiteRtOpCodeTflPack:
      out << "axis: " << opts.AsPackOptions()->axis << "\n";
      break;
    default:
      out << "No options for op code: " << op.OpCode();
      break;
  }
}

void Dump(Quantization quantization, std::ostream& out) {
  int max_display_count;
  switch (quantization.first) {
    case kLiteRtQuantizationNone:
      return;
    case kLiteRtQuantizationPerTensor:
      out << absl::StreamFormat(" <q PerTensor [ .z = %ld, .s = %f ]>",
                                quantization.second.per_tensor.zero_point,
                                quantization.second.per_tensor.scale);
      return;
    case kLiteRtQuantizationPerChannel:
      max_display_count =
          kMaxDisplayCount < quantization.second.per_channel.num_channels
              ? kMaxDisplayCount
              : quantization.second.per_channel.num_channels;
      out << absl::StreamFormat(" <q PerChannel [ .z = [ ");
      for (int i = 0; i < max_display_count; ++i) {
        out << absl::StreamFormat(
            "%ld, ", quantization.second.per_channel.zero_points[i]);
      }
      out << "...], .s = [ ";
      for (int i = 0; i < max_display_count; ++i) {
        out << absl::StreamFormat("%f, ",
                                  quantization.second.per_channel.scales[i]);
      }
      out << "...], ";
      out << absl::StreamFormat(
          ".d = %d>", quantization.second.per_channel.quantized_dimension);
      return;
    default:
      out << " <q UNKNOWN>";
      return;
  }
}

}  // namespace litert::internal
