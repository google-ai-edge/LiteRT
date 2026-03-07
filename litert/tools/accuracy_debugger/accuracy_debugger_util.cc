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

#include "litert/tools/accuracy_debugger/accuracy_debugger_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/compiler/plugin/algo.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_serialize.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/tools/dump_ops_util.h"
#include "litert/tools/tensor_utils.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::tools {

namespace {

using ::LiteRtModelT;
using ::LiteRtOpT;
using ::LiteRtSubgraphT;
using ::LiteRtTensorT;

std::string CsvEscape(const std::string& s) {
  if (s.find_first_of(",\"\n\r") == std::string::npos) {
    return s;
  }
  return absl::StrFormat("\"%s\"", absl::StrReplaceAll(s, {{"\"", "\"\""}}));
}

}  // namespace

namespace internal {

// Helper to get the index of a tensor in the original model's flat list of
// tensors.
int GetOriginalTensorIndex(const LiteRtTensorT* t, const LiteRtModelT& model) {
  int idx = 0;
  for (const auto* sg : model.Subgraphs()) {
    for (const auto* tensor : sg->Tensors()) {
      if (tensor == t) return idx;
      idx++;
    }
  }
  return -1;
}

ExtractedModel ExtractOps(const std::vector<const LiteRtOpT*>& ops,
                          const LiteRtModelT& original_model) {
  ExtractedModel em;
  if (ops.empty()) return em;

  // Find the original subgraph these ops belong to.
  const LiteRtSubgraphT* original_subgraph = nullptr;
  for (const auto* sg : original_model.Subgraphs()) {
    for (const auto* op : sg->Ops()) {
      if (op == ops.front()) {
        original_subgraph = sg;
        break;
      }
    }
    if (original_subgraph) break;
  }

  const auto& original_op_codes =
      litert::internal::GetTflOpCodes(original_model);
  std::vector<litert::internal::TflOpCodePtr> new_op_codes;
  absl::flat_hash_map<int32_t, int32_t> opcode_map;

  auto get_or_add_opcode = [&](int32_t old_ind) -> int32_t {
    if (old_ind < 0 || old_ind >= (int32_t)original_op_codes.size()) return -1;
    if (opcode_map.contains(old_ind)) return opcode_map[old_ind];
    int32_t new_ind = new_op_codes.size();
    new_op_codes.push_back(
        std::make_unique<tflite::OperatorCodeT>(*original_op_codes[old_ind]));
    opcode_map[old_ind] = new_ind;
    return new_ind;
  };

  absl::flat_hash_map<const LiteRtTensorT*, LiteRtTensorT*> global_tensor_map;

  auto get_or_create_tensor =
      [&](const LiteRtTensorT* original_tensor,
          LiteRtSubgraphT& target_sg) -> LiteRtTensorT& {
    if (global_tensor_map.contains(original_tensor)) {
      return *global_tensor_map[original_tensor];
    }
    auto& new_tensor = target_sg.EmplaceTensor();
    litert::internal::CloneTo(*original_tensor, new_tensor);
    if (litert::internal::IsConstant(*original_tensor)) {
      auto buffer = original_tensor->Weights().Buffer();
      ::SetWeightsFromOwnedBuffer(
          new_tensor.Weights(),
          litert::OwningBufferRef<uint8_t>(buffer.Data(), buffer.Size()));
    }
    global_tensor_map[original_tensor] = &new_tensor;
    return new_tensor;
  };

  auto& main_sg = em.model.EmplaceSubgraph();

  // Map from original decomposition subgraph index to new model's subgraph
  // index.
  absl::flat_hash_map<int, int> decomp_sg_cache;

  for (const auto* op : ops) {
    auto& main_op = main_sg.EmplaceOp();
    litert::internal::CloneTo(*op, main_op);
    litert::internal::SetTflOpCodeInd(
        main_op, get_or_add_opcode(litert::internal::GetTflOpCodeInd(*op)));

    LiteRtOpCode opcode_code = op->OpCode();
    if (opcode_code == kLiteRtOpCodeShloComposite) {
      auto old_opts2 = litert::internal::GetTflOptions2(*op);
      if (old_opts2.type == tflite::BuiltinOptions2_StableHLOCompositeOptions) {
        const auto* composite_opts = old_opts2.AsStableHLOCompositeOptions();
        int decomp_sg_idx = composite_opts->decomposition_subgraph_index;
        if (decomp_sg_idx >= 0 &&
            decomp_sg_idx < (int)original_model.Subgraphs().size()) {
          int new_decomp_sg_idx = -1;
          if (decomp_sg_cache.contains(decomp_sg_idx)) {
            new_decomp_sg_idx = decomp_sg_cache[decomp_sg_idx];
          } else {
            const auto& original_decomp_sg =
                *original_model.Subgraphs()[decomp_sg_idx];
            auto& new_decomp_sg = em.model.EmplaceSubgraph();
            new_decomp_sg_idx = em.model.NumSubgraphs() - 1;
            decomp_sg_cache[decomp_sg_idx] = new_decomp_sg_idx;

            absl::flat_hash_map<const LiteRtTensorT*, LiteRtTensorT*>
                decomp_tensor_map;

            for (const auto* original_tensor : original_decomp_sg.Tensors()) {
              auto& new_tensor = new_decomp_sg.EmplaceTensor();
              litert::internal::CloneTo(*original_tensor, new_tensor);
              if (litert::internal::IsConstant(*original_tensor)) {
                auto buffer = original_tensor->Weights().Buffer();
                ::SetWeightsFromOwnedBuffer(new_tensor.Weights(),
                                            litert::OwningBufferRef<uint8_t>(
                                                buffer.Data(), buffer.Size()));
              }
              decomp_tensor_map[original_tensor] = &new_tensor;
            }

            for (const auto* original_op : original_decomp_sg.Ops()) {
              auto& new_op_in_decomp = new_decomp_sg.EmplaceOp();
              litert::internal::CloneTo(*original_op, new_op_in_decomp);
              litert::internal::SetTflOpCodeInd(
                  new_op_in_decomp,
                  get_or_add_opcode(
                      litert::internal::GetTflOpCodeInd(*original_op)));
              for (const auto* in_t : original_op->Inputs()) {
                litert::internal::AttachInput(
                    in_t ? decomp_tensor_map.at(in_t) : nullptr,
                    new_op_in_decomp);
              }
              for (const auto* out_t : original_op->Outputs()) {
                litert::internal::AttachOutput(
                    out_t ? decomp_tensor_map.at(out_t) : nullptr,
                    new_op_in_decomp);
              }
            }

            for (const auto* original_in : original_decomp_sg.Inputs()) {
              new_decomp_sg.Inputs().push_back(
                  decomp_tensor_map.at(original_in));
            }
            for (const auto* original_out : original_decomp_sg.Outputs()) {
              new_decomp_sg.Outputs().push_back(
                  decomp_tensor_map.at(original_out));
            }

            auto decomp_tensors = new_decomp_sg.Tensors();
            for (int i = 0; i < (int)decomp_tensors.size(); ++i)
              decomp_tensors[i]->SetTensorIndex(i);
          }

          auto native_opts =
              std::make_unique<tflite::StableHLOCompositeOptionsT>(
                  *composite_opts);
          native_opts->decomposition_subgraph_index = new_decomp_sg_idx;

          tflite::BuiltinOptions2Union tfl_composite_opts;
          tfl_composite_opts.type =
              tflite::BuiltinOptions2_StableHLOCompositeOptions;
          tfl_composite_opts.value = native_opts.release();
          litert::internal::SetTflOptions2(main_op,
                                           std::move(tfl_composite_opts));
        }
      }
    }

    for (const auto* input_tensor : op->Inputs()) {
      if (input_tensor == nullptr) {
        litert::internal::AttachInput(nullptr, main_op);
        continue;
      }
      auto& new_tensor = get_or_create_tensor(input_tensor, main_sg);
      litert::internal::AttachInput(&new_tensor, main_op);
    }
    for (const auto* output_tensor : op->Outputs()) {
      if (output_tensor == nullptr) {
        litert::internal::AttachOutput(nullptr, main_op);
        continue;
      }
      auto& new_tensor = get_or_create_tensor(output_tensor, main_sg);
      litert::internal::AttachOutput(&new_tensor, main_op);
    }
  }

  // Re-identify Subgraph IO deterministically.
  absl::flat_hash_set<LiteRtTensorT*> produced_tensors_new;
  for (auto* op : main_sg.Ops()) {
    for (auto* out : op->Outputs()) {
      if (out) produced_tensors_new.insert(out);
    }
  }

  std::vector<LiteRtTensorT*> island_inputs_new;
  for (auto* op : main_sg.Ops()) {
    for (auto* in : op->Inputs()) {
      if (in && !produced_tensors_new.contains(in) &&
          !litert::internal::IsConstant(*in)) {
        bool found = false;
        for (auto* existing : island_inputs_new)
          if (existing == in) found = true;
        if (!found) island_inputs_new.push_back(in);
      }
    }
  }

  // Only include produced tensors that are original subgraph outputs OR
  // consumed by ops NOT in this island.
  std::vector<LiteRtTensorT*> island_outputs_new;
  absl::flat_hash_set<const LiteRtOpT*> island_ops_orig(ops.begin(), ops.end());

  for (auto* t_new : produced_tensors_new) {
    const LiteRtTensorT* t_orig = nullptr;
    for (auto const& [orig, cloned] : global_tensor_map) {
      if (cloned == t_new) {
        t_orig = orig;
        break;
      }
    }
    if (!t_orig) continue;

    bool is_original_output = false;
    if (original_subgraph) {
      for (const auto* out : original_subgraph->Outputs()) {
        if (out == t_orig) {
          is_original_output = true;
          break;
        }
      }
    }

    bool has_external_consumer = false;
    if (original_subgraph) {
      for (const auto* op : original_subgraph->Ops()) {
        if (island_ops_orig.contains(op)) continue;
        for (const auto* in : op->Inputs()) {
          if (in == t_orig) {
            has_external_consumer = true;
            break;
          }
        }
        if (has_external_consumer) break;
      }
    }

    if (is_original_output || has_external_consumer) {
      island_outputs_new.push_back(t_new);
    }
  }

  if (island_outputs_new.empty()) {
    for (auto* t : produced_tensors_new) {
      island_outputs_new.push_back(t);
    }
  }

  absl::flat_hash_map<const LiteRtTensorT*, const LiteRtTensorT*>
      reverse_tensor_map;
  for (auto const& [orig, cloned] : global_tensor_map) {
    if (cloned != nullptr) reverse_tensor_map[cloned] = orig;
  }

  auto sort_by_orig = [&](LiteRtTensorT* a, LiteRtTensorT* b) {
    return GetOriginalTensorIndex(reverse_tensor_map.at(a), original_model) <
           GetOriginalTensorIndex(reverse_tensor_map.at(b), original_model);
  };

  std::sort(island_inputs_new.begin(), island_inputs_new.end(), sort_by_orig);
  std::sort(island_outputs_new.begin(), island_outputs_new.end(), sort_by_orig);

  for (auto* in : island_inputs_new) main_sg.Inputs().push_back(in);
  for (auto* out : island_outputs_new) main_sg.Outputs().push_back(out);

  auto tensors = main_sg.Tensors();
  for (int i = 0; i < (int)tensors.size(); ++i) tensors[i]->SetTensorIndex(i);

  for (auto* in_t : main_sg.Inputs()) {
    em.inputs.push_back(reverse_tensor_map.at(in_t));
  }
  for (auto* out_t : main_sg.Outputs()) {
    em.outputs.push_back(reverse_tensor_map.at(out_t));
  }

  em.model.EmplaceSignature(::MakeDefaultSignature(&main_sg));
  litert::internal::SetTflOpCodes(em.model, std::move(new_op_codes));

  return em;
}

ExtractedModel ExtractOp(const LiteRtOpT& op,
                         const LiteRtModelT& original_model) {
  return ExtractOps({&op}, original_model);
}

ExtractedModel ExtractIsland(const std::vector<LiteRtOp>& partition,
                             const LiteRtModelT& original_model) {
  std::vector<const LiteRtOpT*> ops;
  ops.reserve(partition.size());
  for (auto* op : partition) ops.push_back(op);
  return ExtractOps(ops, original_model);
}

std::vector<std::vector<LiteRtOp>> PartitionIntoIslands(
    const LiteRtSubgraphT& subgraph,
    const std::vector<std::string>& boundary_tensors) {
  if (boundary_tensors.empty()) return {};

  absl::flat_hash_set<absl::string_view> boundaries;
  for (const auto& name : boundary_tensors) {
    boundaries.insert(name);
  }

  std::vector<LiteRtOpWithPartitionIndex> selected_ops;
  LiteRtParamIndex current_partition = 0;

  for (auto* op : subgraph.Ops()) {
    selected_ops.push_back({op, current_partition});

    bool hit_boundary = false;
    for (auto* out : op->Outputs()) {
      if (out && boundaries.contains(out->Name())) {
        hit_boundary = true;
        break;
      }
    }
    if (hit_boundary) {
      current_partition++;
    }
  }

  return litert::internal::GroupPartitionsV2(
      selected_ops, const_cast<LiteRtSubgraph>(&subgraph));
}

litert::Expected<std::vector<double>> GetFloats(TensorBuffer& buffer,
                                                size_t num_elements) {
  std::vector<double> data(num_elements);
  auto type_res = buffer.TensorType();
  if (!type_res) return litert::Unexpected(type_res.Error().Status());
  auto element_type = type_res->ElementType();

  if (element_type == ElementType::Float32) {
    std::vector<float> float_data(num_elements);
    LITERT_RETURN_IF_ERROR(buffer.Read<float>(absl::MakeSpan(float_data)));
    for (size_t i = 0; i < num_elements; ++i)
      data[i] = static_cast<double>(float_data[i]);
  } else if (element_type == ElementType::Int32) {
    std::vector<int32_t> int_data(num_elements);
    LITERT_RETURN_IF_ERROR(buffer.Read<int32_t>(absl::MakeSpan(int_data)));
    for (size_t i = 0; i < num_elements; ++i)
      data[i] = static_cast<double>(int_data[i]);
  } else if (element_type == ElementType::Int64) {
    std::vector<int64_t> int_data(num_elements);
    LITERT_RETURN_IF_ERROR(buffer.Read<int64_t>(absl::MakeSpan(int_data)));
    for (size_t i = 0; i < num_elements; ++i)
      data[i] = static_cast<double>(int_data[i]);
  } else if (element_type == ElementType::Int16) {
    std::vector<int16_t> int_data(num_elements);
    LITERT_RETURN_IF_ERROR(buffer.Read<int16_t>(absl::MakeSpan(int_data)));
    for (size_t i = 0; i < num_elements; ++i)
      data[i] = static_cast<double>(int_data[i]);
  } else if (element_type == ElementType::UInt16) {
    std::vector<uint16_t> int_data(num_elements);
    LITERT_RETURN_IF_ERROR(buffer.Read<uint16_t>(absl::MakeSpan(int_data)));
    for (size_t i = 0; i < num_elements; ++i)
      data[i] = static_cast<double>(int_data[i]);
  } else if (element_type == ElementType::Int8) {
    std::vector<int8_t> int_data(num_elements);
    LITERT_RETURN_IF_ERROR(buffer.Read<int8_t>(absl::MakeSpan(int_data)));
    for (size_t i = 0; i < num_elements; ++i)
      data[i] = static_cast<double>(int_data[i]);
  } else if (element_type == ElementType::UInt8) {
    std::vector<uint8_t> int_data(num_elements);
    LITERT_RETURN_IF_ERROR(buffer.Read<uint8_t>(absl::MakeSpan(int_data)));
    for (size_t i = 0; i < num_elements; ++i)
      data[i] = static_cast<double>(int_data[i]);
  } else if (element_type == ElementType::Bool) {
    std::vector<uint8_t> bool_data(num_elements);
    LITERT_RETURN_IF_ERROR(buffer.Read<uint8_t>(absl::MakeSpan(bool_data)));
    for (size_t i = 0; i < num_elements; ++i)
      data[i] = bool_data[i] ? 1.0 : 0.0;
  } else if (element_type == ElementType::Float16) {
    std::vector<uint16_t> fp16_data(num_elements);
    LITERT_RETURN_IF_ERROR(buffer.Read<uint16_t>(absl::MakeSpan(fp16_data)));
    for (size_t i = 0; i < num_elements; ++i) {
      uint16_t h = fp16_data[i];
      uint32_t s = (h >> 15) & 0x1, e = (h >> 10) & 0x1f, f = h & 0x3ff, r;
      if (e == 0) {
        if (f == 0) {
          r = (s << 31);
        } else {
          e = 127 - 14;
          while (!(f & 0x400)) {
            f <<= 1;
            e -= 1;
          }
          f &= 0x3ff;
          r = (s << 31) | (e << 23) | (f << 13);
        }
      } else if (e == 0x1f) {
        r = (s << 31) | (0xff << 23) | (f << 13);
      } else {
        r = (s << 31) | ((e + 127 - 15) << 23) | (f << 13);
      }
      float val;
      std::memcpy(&val, &r, 4);
      data[i] = static_cast<double>(val);
    }
  } else {
    auto size_res = buffer.Size();
    if (!size_res) return litert::Unexpected(size_res.Error().Status());
    std::vector<uint8_t> raw_data(*size_res);
    LITERT_RETURN_IF_ERROR(buffer.Read<uint8_t>(absl::MakeSpan(raw_data)));
    for (size_t i = 0; i < std::min(num_elements, raw_data.size()); ++i)
      data[i] = static_cast<double>(raw_data[i]);
  }
  return data;
}

litert::Expected<ComparisonResult> CompareBuffers(
    TensorBuffer& cpu_buffer, TensorBuffer& accel_buffer,
    const AccuracyThresholds& thresholds, const std::string& op_info) {
  auto cpu_type_res = cpu_buffer.TensorType();
  if (!cpu_type_res) return litert::Unexpected(cpu_type_res.Error().Status());
  size_t total_elements = 1;
  for (size_t d = 0; d < cpu_type_res->Layout().Rank(); ++d)
    total_elements *= cpu_type_res->Layout().Dimensions()[d];

  LITERT_ASSIGN_OR_RETURN(auto cpu_data, GetFloats(cpu_buffer, total_elements));
  LITERT_ASSIGN_OR_RETURN(auto accel_data,
                          GetFloats(accel_buffer, total_elements));

  ComparisonResult res;
  res.failed = false;
  double sum_squared_error = 0, sum_sq_cpu = 0, sum_sq_accel = 0,
         dot_product = 0, sum_cpu = 0, sum_accel = 0;
  double max_abs_cpu = 0;

  res.min_ref = std::numeric_limits<float>::max();
  res.max_ref = -std::numeric_limits<float>::max();
  res.min_accel = std::numeric_limits<float>::max();
  res.max_accel = -std::numeric_limits<float>::max();

  for (size_t i = 0; i < total_elements; ++i) {
    double diff = std::abs(cpu_data[i] - accel_data[i]);
    sum_squared_error += diff * diff;
    if (diff > res.max_diff) res.max_diff = static_cast<float>(diff);
    dot_product += cpu_data[i] * accel_data[i];
    sum_sq_cpu += cpu_data[i] * cpu_data[i];
    sum_sq_accel += accel_data[i] * accel_data[i];
    sum_cpu += cpu_data[i];
    sum_accel += accel_data[i];
    if (std::abs(cpu_data[i]) > max_abs_cpu)
      max_abs_cpu = std::abs(cpu_data[i]);

    if (std::isnan(accel_data[i])) res.has_nan_accel = true;
    if (cpu_data[i] < res.min_ref)
      res.min_ref = static_cast<float>(cpu_data[i]);
    if (cpu_data[i] > res.max_ref)
      res.max_ref = static_cast<float>(cpu_data[i]);
    if (accel_data[i] < res.min_accel)
      res.min_accel = static_cast<float>(accel_data[i]);
    if (accel_data[i] > res.max_accel)
      res.max_accel = static_cast<float>(accel_data[i]);
  }

  res.mse = sum_squared_error / total_elements;
  res.mean_diff = sum_cpu / total_elements - sum_accel / total_elements;
  double mag_cpu = std::sqrt(sum_sq_cpu), mag_accel = std::sqrt(sum_sq_accel);
  res.cosine_similarity = (mag_cpu > 0 && mag_accel > 0)
                              ? (dot_product / (mag_cpu * mag_accel))
                              : (mag_cpu == mag_accel ? 1.0 : 0.0);
  if (sum_squared_error > 0) {
    res.snr = 10.0 * std::log10(sum_sq_cpu / sum_squared_error);
    res.psnr = 20.0 * std::log10(max_abs_cpu / std::sqrt(res.mse));
  } else {
    res.snr = res.psnr = std::numeric_limits<double>::infinity();
  }

  double mean_cpu = sum_cpu / total_elements,
         mean_accel = sum_accel / total_elements, num = 0, den_cpu = 0,
         den_accel = 0;
  for (size_t i = 0; i < total_elements; ++i) {
    num += (cpu_data[i] - mean_cpu) * (accel_data[i] - mean_accel);
    den_cpu += (cpu_data[i] - mean_cpu) * (cpu_data[i] - mean_cpu);
    den_accel += (accel_data[i] - mean_accel) * (accel_data[i] - mean_accel);
  }
  res.pearson_correlation = (den_cpu > 0 && den_accel > 0)
                                ? (num / std::sqrt(den_cpu * den_accel))
                                : (sum_squared_error == 0 ? 1.0 : 0.0);

  if (res.max_diff > thresholds.max_diff)
    res.failing_metrics.push_back("max_diff");
  if (res.mse > thresholds.mse) res.failing_metrics.push_back("mse");
  if (res.cosine_similarity < thresholds.cosine_similarity)
    res.failing_metrics.push_back("cos_sim");
  if (res.snr < thresholds.snr) res.failing_metrics.push_back("snr");
  if (res.psnr < thresholds.psnr) res.failing_metrics.push_back("psnr");
  res.failed = !res.failing_metrics.empty();
  return res;
}

}  // namespace internal

void AccuracyDebuggerSummary::LogSummary(
    const AccuracyDebuggerOptions& options) const {
  if (all_ops.empty()) {
    std::cout
        << "\nNo operations were successfully processed. Summary is empty."
        << std::endl;
    return;
  }
  std::vector<OpStats> sorted = all_ops;
  if (options.sort_by == "max_diff") {
    std::sort(sorted.begin(), sorted.end(),
              [](const OpStats& a, const OpStats& b) {
                return a.metrics.max_diff > b.metrics.max_diff;
              });
  } else if (options.sort_by == "mse") {
    std::sort(sorted.begin(), sorted.end(),
              [](const OpStats& a, const OpStats& b) {
                return a.metrics.mse > b.metrics.mse;
              });
  } else if (options.sort_by == "cos_sim") {
    std::sort(
        sorted.begin(), sorted.end(), [](const OpStats& a, const OpStats& b) {
          return a.metrics.cosine_similarity < b.metrics.cosine_similarity;
        });
  } else if (options.sort_by == "snr") {
    std::sort(sorted.begin(), sorted.end(),
              [](const OpStats& a, const OpStats& b) {
                return a.metrics.snr < b.metrics.snr;
              });
  } else if (options.sort_by == "psnr") {
    std::sort(sorted.begin(), sorted.end(),
              [](const OpStats& a, const OpStats& b) {
                return a.metrics.psnr < b.metrics.psnr;
              });
  } else {
    // Default sort by index.
    std::sort(sorted.begin(), sorted.end(),
              [](const OpStats& a, const OpStats& b) {
                return a.global_index < b.global_index;
              });
  }
  static const char* kBanner =
      "\n================================================================"
      "========================================================================"
      "==";
  static const char* kSeparator =
      "----------------------------------------------------------------"
      "------------------------------------------------------------------------"
      "--";
  std::cout << kBanner << std::endl;
  std::cout << "Accuracy Debugger Summary" << std::endl;
  std::cout << "Total operations: " << all_ops.size() << std::endl;
  std::cout << "Regressions:      " << failing_op_indices.size() << std::endl;
  std::cout << kBanner << std::endl;
  std::cout << absl::StreamFormat(
                   "%-6s %-35s | %-12s | %-10s | %-5s | %-20s | %-20s | %-15s",
                   "Index", "Op Code", "Max Diff", "Cos Sim", "NaN",
                   "Ref Range", "Accel Range", "Status")
            << std::endl;
  std::cout << kSeparator << std::endl;

  size_t max_rows = sorted.size();
  if (options.summary_max_rows >= 0) {
    max_rows = std::min<size_t>(max_rows, options.summary_max_rows);
  }

  for (size_t i = 0; i < max_rows; ++i) {
    const auto& op = sorted[i];
    std::string status = "OK";
    for (const auto& m : op.metrics.failing_metrics) {
      if (absl::StrContains(m, "SKIPPED") || absl::StrContains(m, "FAILED")) {
        status = m;
        break;
      }
    }
    if (status == "OK" && op.metrics.failed) status = "REGRESSION";

    std::string ref_range =
        absl::StrFormat("[%.3f, %.3f]", op.metrics.min_ref, op.metrics.max_ref);
    std::string accel_range = absl::StrFormat(
        "[%.3f, %.3f]", op.metrics.min_accel, op.metrics.max_accel);
    std::string has_nan = op.metrics.has_nan_accel ? "Y" : "N";

    std::cout
        << absl::StreamFormat(
               "[%4d] %-35s | %12.6f | %10.8f | %5s | %-20s | %-20s | %-15s",
               op.global_index, op.op_code, op.metrics.max_diff,
               op.metrics.cosine_similarity, has_nan, ref_range, accel_range,
               status)
        << std::endl;
  }
  std::cout << kBanner << std::endl;
}

void AccuracyDebuggerSummary::SaveToCsv(const std::string& path) const {
  std::ofstream csv_file(path);
  if (!csv_file.is_open()) {
    ABSL_LOG(ERROR) << "Failed to open CSV file for writing: " << path;
    return;
  }

  csv_file << "Index,Op Code,Tensor Name,Max Diff,MSE,Cos Sim,SNR (dB),PSNR "
              "(dB),NaN,Min Ref,Max Ref,Min Accel,Max Accel,Status\n";
  for (const auto& op : all_ops) {
    std::string status = "OK";
    for (const auto& m : op.metrics.failing_metrics) {
      if (absl::StrContains(m, "SKIPPED") || absl::StrContains(m, "FAILED")) {
        status = m;
        break;
      }
    }
    if (status == "OK" && op.metrics.failed) status = "REGRESSION";

    csv_file << absl::StrFormat(
        "%d,%s,%s,%.6f,%.6f,%.8f,%.2f,%.2f,%s,%.6f,%.6f,%.6f,%.6f,%s\n",
        op.global_index, op.op_code, CsvEscape(op.tensor_name),
        op.metrics.max_diff, op.metrics.mse, op.metrics.cosine_similarity,
        op.metrics.snr, op.metrics.psnr, op.metrics.has_nan_accel ? "Y" : "N",
        op.metrics.min_ref, op.metrics.max_ref, op.metrics.min_accel,
        op.metrics.max_accel, status);
  }
  std::cout << "Summary saved to: " << path << std::endl;
}

std::string GetOpName(const LiteRtOpT& op) {
  std::string name = OpCodeToString(op.OpCode());
  if (op.OpCode() == kLiteRtOpCodeShloComposite) {
    auto opts2 = litert::internal::GetTflOptions2(op);
    if (opts2.type == tflite::BuiltinOptions2_StableHLOCompositeOptions) {
      const auto* composite_opts = opts2.AsStableHLOCompositeOptions();
      if (composite_opts && !composite_opts->name.empty()) {
        name = absl::StrFormat("%s(%s)", name, composite_opts->name);
      }
    }
  }
  return name;
}

std::string SanitizeFilename(const std::string& name) {
  return absl::StrReplaceAll(name, {{"(", "_"},
                                   {")", "_"},
                                   {" ", "_"},
                                   {"[", "_"},
                                   {"]", "_"},
                                   {"/", "_"},
                                   {":", "_"}});
}

absl::Status RunAccuracyDebugger(litert::Environment& env, LiteRtModelT& model,
                                 litert::Options& accel_opts,
                                 const AccuracyDebuggerOptions& options,
                                 AccuracyDebuggerSummary* summary) {
  auto cpu_opts_res = litert::Options::Create();
  if (!cpu_opts_res) return absl::InternalError("Opts failed");
  if (options.use_gpu_ref) {
    cpu_opts_res->SetHardwareAccelerators(litert::HwAccelerators::kGpu);
    auto gpu_opts = cpu_opts_res->GetGpuOptions();
    if (!gpu_opts) return absl::InternalError("GPU opts failed");
    gpu_opts->SetPrecision(litert::GpuOptions::Precision::kFp32);
  } else {
    cpu_opts_res->SetHardwareAccelerators(litert::HwAccelerators::kCpu);
  }

  absl::flat_hash_map<const LiteRtTensorT*, std::vector<char>>
      cpu_tensor_values;
  absl::flat_hash_map<const LiteRtTensorT*, std::vector<char>>
      accel_tensor_values;

  std::vector<const LiteRtSubgraphT*> subgraphs;
  if (options.signature_index < model.Signatures().size()) {
    subgraphs.push_back(
        &model.Signatures()[options.signature_index]->GetSubgraph());
  } else if (!model.Subgraphs().empty()) {
    subgraphs.push_back(model.MainSubgraph());
  }

  absl::flat_hash_map<const LiteRtTensorT*, int> global_consumer_counts;
  for (const auto* sg : subgraphs) {
    for (auto* op : sg->Ops()) {
      for (const auto* in : op->Inputs())
        if (in) global_consumer_counts[in]++;
    }
    for (const auto* out : sg->Outputs())
      if (out) global_consumer_counts[out]++;
  }

  std::filesystem::create_directories(options.output_dir);

  struct WorkUnit {
    std::vector<const LiteRtOpT*> ops;
    std::string opcode_str;
    bool skip_accel = false;
  };
  std::vector<WorkUnit> work_units;

  for (const auto* subgraph : subgraphs) {
    if (!options.boundary_tensors.empty()) {
      auto islands =
          internal::PartitionIntoIslands(*subgraph, options.boundary_tensors);
      for (size_t i = 0; i < islands.size(); ++i) {
        WorkUnit unit;
        for (auto* op : islands[i]) unit.ops.push_back(op);
        unit.opcode_str =
            absl::StrFormat("Partition_%d(%d Ops)", i, unit.ops.size());
        work_units.push_back(std::move(unit));
      }
    } else {
      for (const auto* op : subgraph->Ops()) {
        WorkUnit unit;
        unit.ops.push_back(op);
        unit.opcode_str = GetOpName(*op);
        if (options.skip_unsupported_npu_ops &&
            (op->OpCode() == kLiteRtOpCodeTflEmbeddingLookup ||
             op->OpCode() == kLiteRtOpCodeTflCustom)) {
          unit.skip_accel = true;
        }
        work_units.push_back(std::move(unit));
      }
    }
  }

  int global_op_counter = 0;
  for (auto& unit : work_units) {
    if (options.max_ops != -1 && global_op_counter >= options.max_ops) break;

    std::string sanitized_opcode = SanitizeFilename(unit.opcode_str);
    std::string op_info =
        absl::StrFormat("[%d] %s", global_op_counter, unit.opcode_str);

    internal::ExtractedModel em = internal::ExtractOps(unit.ops, model);
    auto serialized_res = litert::internal::SerializeModel(std::move(em.model));
    if (!serialized_res) return absl::InternalError("Serialize failed");
    auto serialized = std::move(*serialized_res);

    if (options.dump_only) {
      std::string dump_path = absl::StrFormat(
          "%s/op_%04d_%s_i%d_o%d.tflite", options.output_dir, global_op_counter,
          sanitized_opcode, em.inputs.size(), em.outputs.size());
      std::ofstream dump_file(dump_path, std::ios::binary);
      if (!dump_file.is_open()) {
        ABSL_LOG(ERROR) << "Failed to open dump file: " << dump_path;
      } else {
        dump_file.write(reinterpret_cast<const char*>(serialized.Data()),
                        serialized.Size());
        std::cout << "Dumped model to: " << dump_path << std::endl;
      }
      global_op_counter++;
      continue;
    }

    // --- CPU Path (Golden Reference) ---
    auto cpu_model_res = CompiledModel::Create(env, serialized, *cpu_opts_res);
    if (!cpu_model_res) {
      ComparisonResult final_res;
      final_res.failed = true;
      final_res.failing_metrics.push_back("CPU_COMPILE_FAILED");
      if (summary)
        summary->all_ops.push_back(
            {(int)global_op_counter, unit.opcode_str, "", final_res});
      global_op_counter++;
      continue;
    }
    auto& cpu_model = *cpu_model_res;

    auto cpu_in_exp = cpu_model.CreateInputBuffers(0);
    if (!cpu_in_exp) return absl::InternalError("CPU inputs failed");
    auto cpu_inputs = std::move(cpu_in_exp.Value());

    if (!options.input_dir.empty() && global_op_counter == 0) {
      auto fill_status = tensor_utils::FillInputBuffersWithCustomData(
          cpu_model, 0, cpu_inputs, options.input_dir);
      if (!fill_status) {
        ABSL_LOG(WARNING) << "Failed to fill inputs from " << options.input_dir
                          << ": " << fill_status.Error().Message();
      }
    }

    auto cpu_out_exp = cpu_model.CreateOutputBuffers(0);
    if (!cpu_out_exp) return absl::InternalError("CPU outputs failed");
    auto cpu_outputs = std::move(cpu_out_exp.Value());

    for (size_t i = 0; i < em.inputs.size(); ++i) {
      const auto* orig = em.inputs[i];
      if (orig && cpu_tensor_values.contains(orig)) {
        (void)cpu_inputs[i].Write<char>(
            absl::MakeSpan(cpu_tensor_values[orig]));
      } else {
        (void)tensor_utils::FillBufferWithRandomData(cpu_inputs[i]);
        auto size_res = cpu_inputs[i].Size();
        if (!size_res) continue;
        std::vector<char> tmp(*size_res);
        (void)cpu_inputs[i].Read<char>(absl::MakeSpan(tmp));
        if (orig) {
          cpu_tensor_values[orig] = tmp;
          accel_tensor_values[orig] = std::move(tmp);
        }
      }
    }
    auto cpu_run_status =
        cpu_model.Run(static_cast<size_t>(0), cpu_inputs, cpu_outputs);
    if (!cpu_run_status) {
      ComparisonResult final_res;
      final_res.failed = true;
      final_res.failing_metrics.push_back("CPU_RUN_FAILED");
      if (summary) {
        summary->all_ops.push_back(
            {(int)global_op_counter, unit.opcode_str, "", final_res});
        summary->failing_op_indices.push_back(global_op_counter);
      }
      global_op_counter++;
      continue;
    }

    for (size_t i = 0; i < em.outputs.size(); ++i) {
      auto size_res = cpu_outputs[i].Size();
      if (!size_res) continue;
      std::vector<char> cpu_data(*size_res);
      (void)cpu_outputs[i].Read<char>(absl::MakeSpan(cpu_data));
      if (em.outputs[i]) {
        cpu_tensor_values[em.outputs[i]] = std::move(cpu_data);
      }
    }

    // --- Accel Path (Test Path) ---
    ComparisonResult worst_res;
    worst_res.failed = false;
    bool accel_run_success = false;
    bool any_tensor_failed = false;

    if (unit.skip_accel) {
      worst_res.failing_metrics.push_back("SKIPPED_FOR_NPU");
    } else {
      auto accel_model_res = CompiledModel::Create(env, serialized, accel_opts);
      if (!accel_model_res) {
        worst_res.failed = true;
        worst_res.failing_metrics.push_back("ACCEL_COMPILE_FAILED");
      } else {
        auto& accel_model = *accel_model_res;
        auto fully_accel_exp = accel_model.IsFullyAccelerated();
        if (!fully_accel_exp.HasValue() || !fully_accel_exp.Value()) {
          worst_res.failed = true;
          worst_res.failing_metrics.push_back("NOT_PARTITIONED");
        } else {
          auto accel_in_exp = accel_model.CreateInputBuffers(0);
          auto accel_out_exp = accel_model.CreateOutputBuffers(0);
          if (!accel_in_exp || !accel_out_exp) {
            worst_res.failed = true;
            worst_res.failing_metrics.push_back("ACCEL_BUFFERS_FAILED");
          } else {
            auto accel_inputs = std::move(accel_in_exp.Value());
            auto accel_outputs = std::move(accel_out_exp.Value());
            for (size_t i = 0; i < em.inputs.size(); ++i) {
              const auto* orig = em.inputs[i];
              if (!orig) continue;
              const auto& source_map = options.use_accel_output_as_input
                                           ? accel_tensor_values
                                           : cpu_tensor_values;
              if (source_map.contains(orig)) {
                (void)accel_inputs[i].Write<char>(
                    absl::MakeSpan(source_map.at(orig)));
              }
            }
            auto run_status = accel_model.Run(static_cast<size_t>(0),
                                              accel_inputs, accel_outputs);
            if (!run_status) {
              worst_res.failed = true;
              worst_res.failing_metrics.push_back("ACCEL_RUN_FAILED");
            } else {
              accel_run_success = true;
              for (size_t i = 0; i < em.outputs.size(); ++i) {
                auto comp_res = internal::CompareBuffers(
                    cpu_outputs[i], accel_outputs[i], options.thresholds,
                    absl::StrFormat("%s out %d", op_info, i));
                if (comp_res) {
                  std::string t_name = em.outputs[i]
                                           ? std::string(em.outputs[i]->Name())
                                           : "unknown";
                  ABSL_LOG(INFO) << absl::StreamFormat(
                      "  Comparing tensor: %s | Max Diff: %.6f | MSE: %.6f | "
                      "Cos Sim: %.6f",
                      t_name, comp_res->max_diff, comp_res->mse,
                      comp_res->cosine_similarity);

                  if (comp_res->failed) any_tensor_failed = true;

                  if (summary) {
                    summary->all_ops.push_back({(int)global_op_counter,
                                                unit.opcode_str, t_name,
                                                *comp_res});
                  }

                  // Track worst result for save_failing_models decision.
                  if (comp_res->failed) {
                    worst_res = *comp_res;
                  } else if (!worst_res.failed) {
                    worst_res = *comp_res;
                  }
                }
              }

              for (size_t i = 0; i < em.outputs.size(); ++i) {
                if (!em.outputs[i]) continue;
                auto size_res = accel_outputs[i].Size();
                if (!size_res) continue;
                std::vector<char> accel_data(*size_res);
                (void)accel_outputs[i].Read<char>(absl::MakeSpan(accel_data));
                accel_tensor_values[em.outputs[i]] = std::move(accel_data);
              }
            }
          }
        }
      }
    }

    if (!accel_run_success) {
      // If path failed, push a single entry.
      if (summary) {
        summary->all_ops.push_back(
            {(int)global_op_counter, unit.opcode_str, "", worst_res});
      }
      for (size_t i = 0; i < em.outputs.size(); ++i) {
        if (em.outputs[i]) {
          accel_tensor_values[em.outputs[i]] = cpu_tensor_values[em.outputs[i]];
        }
      }
    }

    if ((worst_res.failed || any_tensor_failed) && summary)
      summary->failing_op_indices.push_back(global_op_counter);

    if (options.save_failing_models &&
        (worst_res.failed || any_tensor_failed)) {
      std::string base = absl::StrFormat(
          "%s/op_%04d_%s_i%d_o%d", options.output_dir, global_op_counter,
          sanitized_opcode, em.inputs.size(), em.outputs.size());
      std::ofstream model_file(base + ".tflite", std::ios::binary);
      model_file.write(reinterpret_cast<const char*>(serialized.Data()),
                       serialized.Size());
      for (size_t i = 0; i < em.inputs.size(); ++i) {
        const auto* orig = em.inputs[i];
        const auto& source_map = options.use_accel_output_as_input
                                     ? accel_tensor_values
                                     : cpu_tensor_values;
        if (source_map.contains(orig)) {
          std::ofstream in_file(base + absl::StrFormat("_in_%d.bin", i),
                                std::ios::binary);
          in_file.write(source_map.at(orig).data(), source_map.at(orig).size());
        }
      }
    }

    for (const auto* op : unit.ops) {
      for (const auto* in_t : op->Inputs()) {
        if (in_t && global_consumer_counts.contains(in_t) &&
            --global_consumer_counts[in_t] == 0) {
          cpu_tensor_values.erase(in_t);
          accel_tensor_values.erase(in_t);
        }
      }
    }
    global_op_counter++;
  }

  if (summary) {
    summary->LogSummary(options);
    summary->SaveToCsv(
        absl::StrFormat("%s/accuracy_summary.csv", options.output_dir));
  }
  return absl::OkStatus();
}

}  // namespace litert::tools
