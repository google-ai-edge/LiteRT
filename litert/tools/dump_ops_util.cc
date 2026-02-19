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

#include "litert/tools/dump_ops_util.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"
#include "litert/core/model/model_serialize.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/tools/dump.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::tools {

namespace {

std::string OpCodeToString(LiteRtOpCode code) {
  std::ostringstream oss;
  litert::internal::Dump(code, oss);
  if (oss.str().empty()) {
    return "UNKNOWN";
  }
  return oss.str();
}

std::string OpSignature(const LiteRtOpT& op) {
  std::ostringstream oss;
  litert::internal::Dump(op, oss);
  return oss.str();
}

}  // namespace

absl::Status DumpOps(LiteRtModelT& model, const DumpOptions& options,
                     DumpStats* stats) {
  std::filesystem::create_directories(options.output_dir);

  absl::flat_hash_set<std::string>* unique_ops_ptr;
  absl::flat_hash_set<std::string> local_unique_ops;
  if (stats) {
    unique_ops_ptr = &stats->unique_op_signatures;
  } else {
    unique_ops_ptr = &local_unique_ops;
  }

  int op_counter = 0;

  ABSL_LOG(INFO) << "Model has " << model.NumSubgraphs() << " subgraphs";

  for (auto* subgraph : model.Subgraphs()) {
    ABSL_LOG(INFO) << "Subgraph has " << subgraph->Ops().size() << " ops";
    for (auto* op : subgraph->Ops()) {
      std::string opcode_str = OpCodeToString(op->OpCode());
      ABSL_LOG(INFO) << "Processing op: " << opcode_str;

      if (options.filter_opcode.has_value() &&
          opcode_str.find(options.filter_opcode.value()) == std::string::npos) {
        continue;
      }

      if (options.unique) {
        std::string signature = OpSignature(*op);
        if (unique_ops_ptr->contains(signature)) {
          continue;
        }
        unique_ops_ptr->insert(signature);
      }

      LiteRtModelT new_model;
      int32_t old_opcode_ind = litert::internal::GetTflOpCodeInd(*op);
      const auto& old_op_codes = litert::internal::GetTflOpCodes(model);
      if (old_opcode_ind >= 0 && old_opcode_ind < old_op_codes.size()) {
        std::vector<litert::internal::TflOpCodePtr> new_op_codes;
        new_op_codes.push_back(std::make_unique<tflite::OperatorCodeT>(
            *old_op_codes[old_opcode_ind]));
        litert::internal::SetTflOpCodes(new_model, std::move(new_op_codes));
      }

      auto& new_subgraph = new_model.EmplaceSubgraph();

      auto& new_op = new_subgraph.EmplaceOp();
      litert::internal::CloneTo(*op, new_op);

      if (old_opcode_ind >= 0 && old_opcode_ind < old_op_codes.size()) {
        litert::internal::SetTflOpCodeInd(new_op, 0);
      }

      std::vector<int> input_indices;
      std::vector<int> graph_input_indices;
      for (const auto& input_tensor : op->Inputs()) {
        auto& new_tensor = new_subgraph.EmplaceTensor();
        litert::internal::CloneTo(*input_tensor, new_tensor);
        if (litert::internal::IsConstant(*input_tensor)) {
          SetWeightsFromUnownedBuffer(new_tensor.Weights(),
                                      input_tensor->Weights().Buffer());
        }

        int tensor_idx = new_subgraph.Tensors().size() - 1;
        input_indices.push_back(tensor_idx);

        if (!litert::internal::IsConstant(*input_tensor)) {
          graph_input_indices.push_back(tensor_idx);
        }
      }

      std::vector<int> output_indices;
      for (const auto& output_tensor : op->Outputs()) {
        auto& new_tensor = new_subgraph.EmplaceTensor();
        litert::internal::CloneTo(*output_tensor, new_tensor);
        int tensor_idx = new_subgraph.Tensors().size() - 1;
        output_indices.push_back(tensor_idx);
      }

      // NOTE: We do this after all tensors are created to avoid reference
      // invalidation caused by std::vector reallocation in EmplaceTensor.
      auto tensors = new_subgraph.Tensors();
      for (int i = 0; i < tensors.size(); ++i) {
        tensors[i]->SetTensorIndex(i);
      }
      for (int idx : input_indices) {
        litert::internal::AttachInput(tensors[idx], new_op);
      }
      if (graph_input_indices.empty()) {
        ABSL_LOG(ERROR) << "No graph inputs for op " << op_counter << " "
                        << opcode_str << ", skipping";
        continue;
      }
      for (int idx : graph_input_indices) {
        new_subgraph.Inputs().push_back(tensors[idx]);
      }
      for (int idx : output_indices) {
        litert::internal::AttachOutput(tensors[idx], new_op);
        new_subgraph.Outputs().push_back(tensors[idx]);
      }

      new_model.EmplaceSignature(MakeDefaultSignature(&new_subgraph));

      // Serialize
      auto serialized = litert::internal::SerializeModel(std::move(new_model));
      if (!serialized) {
        ABSL_LOG(ERROR) << "Failed to serialize op " << op_counter;
        continue;
      }

      std::string filename_prefix =
          options.filename_prefix.empty()
              ? ""
              : absl::StrCat(options.filename_prefix, "_");
      std::string filename = absl::StrCat(filename_prefix, "op_", op_counter,
                                          "_", opcode_str, ".tflite");
      std::string path =
          (std::filesystem::path(options.output_dir) / filename).string();

      std::ofstream out(path, std::ios::binary);
      if (!out.is_open()) {
        ABSL_LOG(ERROR) << "Failed to open file for writing: " << path;
        continue;
      }
      serialized->WriteStr(out);
      out.close();

      op_counter++;
      if (stats) {
        stats->total_ops_dumped++;
        stats->op_code_counts[opcode_str]++;
      }
    }
  }

  ABSL_LOG(INFO) << "Dumped " << op_counter << " ops to " << options.output_dir;
  return absl::OkStatus();
}

}  // namespace litert::tools
