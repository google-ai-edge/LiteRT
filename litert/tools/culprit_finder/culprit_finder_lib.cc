// Copyright 2025 Google LLC.
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

#include "litert/tools/culprit_finder/culprit_finder_lib.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/cc/internal/litert_tflite_error_status_builder.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/tools/culprit_finder/culprit_finder_utils.h"
#include "litert/tools/culprit_finder/interpreter_handler.h"
#include "litert/tools/culprit_finder/model_metadata_lib.h"
#include "litert/tools/culprit_finder/tflite_input_manager.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"
#include "tflite/profiling/memory_usage_monitor.h"
#include "tflite/tools/command_line_flags.h"
#include "tflite/tools/delegates/delegate_provider.h"
#include "tflite/tools/tool_params.h"

namespace litert::tools {

constexpr char kModelFileFlag[] = "graph";
constexpr char kSearchStrategyFlag[] = "search_strategy";

// Search strategy enums.
constexpr char kBinarySearchStrategyEnum[] = "binary";
constexpr char kLinearSearchStrategyEnum[] = "linear";

// Binary search specific flags.
constexpr char kBinarySearchReverseSweepFlag[] = "binary_search_reverse_sweep";

// Linear search specific flags.
constexpr char kLinearSearchStrideSizeFlag[] = "linear_search_stride_size";
constexpr char kLinearSearchNodeFilterFlag[] = "linear_search_node_filter";
constexpr char kLinearSearchReportCountFlag[] = "linear_search_report_count";

// Find NAN specific flags.
constexpr char kFindNanFlag[] = "find_nan";
// Find numeric error specific flags.
constexpr char kFindNumericErrorFlag[] = "find_numeric_error";
constexpr char kMinNumericErrorFlag[] = "min_numeric_error";

using ::tflite::Flag;
using ::tflite::Flags;
using ::tflite::Interpreter;
using ::tflite::tools::ProvidedDelegateList;
using ::tflite::tools::ToolParam;

CulpritFinder::CulpritFinder(int* argc, const char** argv)
    : delegate_list_util_(&params_) {
  SetDefaultParams();
  bool parse_result = InitFromCmdlineArgs(argc, argv);

  if (!parse_result) {
    LITERT_LOG(LITERT_ERROR, "Failed to parse command line arguments");
    return;
  }
  LogParams();

  if (GetModelPath().empty()) {
    LITERT_LOG(LITERT_ERROR, "Model path is empty");
    return;
  }

  litert::Expected<std::unique_ptr<InterpreterHandler>> interpreter_handler =
      InterpreterHandler::Create(GetModelPath());
  if (!interpreter_handler) {
    LITERT_LOG(LITERT_ERROR, "Failed to load model from path: %s",
               GetModelPath().c_str());
    return;
  }
  interpreter_handler_ = std::move(*interpreter_handler);
  const std::vector<ProvidedDelegateList::ProvidedDelegate> delegates =
      delegate_list_util_.CreateAllRankedDelegates();
  if (delegates.size() != 1) {
    LITERT_LOG(LITERT_ERROR, "Expected 1 delegate, got %d", delegates.size());
    return;
  };
}

void CulpritFinder::SetDefaultParams() {
  params_.AddParam(kModelFileFlag, ToolParam::Create<std::string>(""));
  params_.AddParam(kSearchStrategyFlag,
                   ToolParam::Create<std::string>(kLinearSearchStrategyEnum));
  params_.AddParam(kLinearSearchStrideSizeFlag, ToolParam::Create<int>(1));
  params_.AddParam(kLinearSearchNodeFilterFlag,
                   ToolParam::Create<std::string>(""));
  params_.AddParam(kLinearSearchReportCountFlag, ToolParam::Create<int>(5));
  params_.AddParam(kBinarySearchReverseSweepFlag,
                   ToolParam::Create<bool>(false));
  params_.AddParam(kFindNanFlag, ToolParam::Create<bool>(true));
  params_.AddParam(kFindNumericErrorFlag, ToolParam::Create<bool>(true));
  params_.AddParam(kMinNumericErrorFlag, ToolParam::Create<float>(0.0001));
  delegate_list_util_.AddAllDelegateParams();
}

void CulpritFinder::LogParams() {
  LOG_TOOL_PARAM(params_, std::string, kModelFileFlag, "Model file", true);
  LOG_TOOL_PARAM(params_, std::string, kSearchStrategyFlag, "Search strategy",
                 true);
  LOG_TOOL_PARAM(params_, int, kLinearSearchStrideSizeFlag,
                 "Linear search stride size", true);
  LOG_TOOL_PARAM(params_, std::string, kLinearSearchNodeFilterFlag,
                 "Linear search node filter", true);
  LOG_TOOL_PARAM(params_, int, kLinearSearchReportCountFlag,
                 "Linear search report count", true);
  LOG_TOOL_PARAM(params_, bool, kBinarySearchReverseSweepFlag,
                 "Binary search find end first", true);
  LOG_TOOL_PARAM(params_, bool, kFindNanFlag, "Find NAN", true);
  LOG_TOOL_PARAM(params_, bool, kFindNumericErrorFlag, "Find numeric error",
                 true);
  LOG_TOOL_PARAM(params_, float, kMinNumericErrorFlag, "Min numeric error",
                 true);
  for (const std::unique_ptr<tflite::tools::DelegateProvider>&
           delegate_provider :
       tflite::tools::GetRegisteredDelegateProviders()) {
    delegate_provider->LogParams(params_, true);
  }
}

std::vector<tflite::Flag> CulpritFinder::GetFlags() {
  std::vector<tflite::Flag> flag_list = {
      CreateFlag<std::string>(kModelFileFlag, &params_,
                              "Path to test tflite model file."),
      CreateFlag<std::string>(kSearchStrategyFlag, &params_,
                              "Search strategy (binary or linear)."),
      CreateFlag<int>(kLinearSearchStrideSizeFlag, &params_,
                      "If provided, the culprit finder will run the linear "
                      "search for steps of this size."),
      CreateFlag<std::string>(
          kLinearSearchNodeFilterFlag, &params_,
          "A comma-separated list of node types to filter out."),
      CreateFlag<int>(kLinearSearchReportCountFlag, &params_,
                      "If provided, the culprit finder will report the "
                      "inference results for this many nodes."),
      CreateFlag<bool>(kBinarySearchReverseSweepFlag, &params_,
                       "If true, find the end node first. Default is false."),
      CreateFlag<bool>(kFindNanFlag, &params_,
                       "If specified, searches for NANs."),
      CreateFlag<bool>(kFindNumericErrorFlag, &params_,
                       "If specified, searches for numeric errors."),
      CreateFlag<float>(kMinNumericErrorFlag, &params_,
                        "Minimum absolute difference to consider an "
                        "inference as an error."),
  };
  delegate_list_util_.AppendCmdlineFlags(flag_list);
  return flag_list;
}

bool CulpritFinder::InitFromCmdlineArgs(int* argc, const char** argv) {
  const std::vector<Flag> flags = GetFlags();
  bool parse_result = Flags::Parse(argc, argv, flags);
  if (!parse_result || params_.Get<bool>("help")) {
    std::string usage = Flags::Usage(argv[0], flags);
    LITERT_LOG(LITERT_ERROR, "%s", usage.c_str());
    // Returning false intentionally when "--help=true" is specified so that
    // the caller could check the return value to decide stopping the
    // execution.
    parse_result = false;
  }
  return parse_result;
}

std::string CulpritFinder::GetModelPath() {
  return params_.Get<std::string>(kModelFileFlag);
}

tflite::tools::TfLiteDelegatePtr CulpritFinder::GetDelegate(int start_node,
                                                            int end_node) {
  params_.Set<int>("first_delegate_node_index", start_node, 0);
  params_.Set<int>("last_delegate_node_index", end_node, 0);
  std::vector<ProvidedDelegateList::ProvidedDelegate> delegates =
      delegate_list_util_.CreateAllRankedDelegates();
  if (delegates.empty()) {
    return tflite::tools::CreateNullDelegate();
  }
  return std::move(delegates[0].delegate);
}

TfLiteStatus CulpritFinder::CalculateErrorStats(
    const int start_node, const int end_node,
    absl::Span<const int> intermediate_outputs, OverallStat& overall_stat) {
  bool is_crash = false;
  try {
    LITERT_ASSIGN_OR_RETURN(
        interpreter_with_delegate_,
        interpreter_handler_->PrepareInterpreter(
            GetDelegate(start_node, end_node), intermediate_outputs),
        AsTfLiteStatus(_ << "Failed to prepare interpreter."));

    TfLiteStatus status = interpreter_handler_->RunInference(
        *interpreter_with_delegate_, *input_manager_);
    if (status != kTfLiteOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to run inference");
      return kTfLiteError;
    }
  } catch (const std::exception& exc) {
    LITERT_LOG(LITERT_ERROR, "Failed to run inference due to a crash.");
    is_crash = true;
  }

  GetOverallStat(start_node, end_node, interpreter_.get(),
                 interpreter_with_delegate_.get(), is_crash, overall_stat);
  return kTfLiteOk;
}

TfLiteStatus CulpritFinder::RunCulpritFinder() {
  std::unique_ptr<tflite::profiling::memory::MemoryUsageMonitor>
      peak_memory_reporter;
  peak_memory_reporter =
      std::make_unique<tflite::profiling::memory::MemoryUsageMonitor>(50);
  peak_memory_reporter->Start();
  std::string search_strategy;
  if (params_.HasValueSet<std::string>(kSearchStrategyFlag)) {
    search_strategy = params_.Get<std::string>(kSearchStrategyFlag);
  } else if (params_.Get<bool>(kFindNanFlag)) {
    search_strategy = kBinarySearchStrategyEnum;
  } else {
    search_strategy = kLinearSearchStrategyEnum;
  }
  TfLiteStatus status = kTfLiteOk;
  if (search_strategy == kLinearSearchStrategyEnum) {
    if (RunCulpritFinderLinearSearch() != kTfLiteOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to run culprit finder linear search");
      status = kTfLiteError;
    }
  } else if (search_strategy == kBinarySearchStrategyEnum) {
    status = RunCulpritFinderBinarySearch();
  } else {
    LITERT_LOG(LITERT_ERROR, "Unsupported search strategy: %s",
               search_strategy.c_str());
    status = kTfLiteError;
  }
  peak_memory_reporter->Stop();
  LITERT_LOG(LITERT_INFO, "### Peak memory usage in MB: %f",
             peak_memory_reporter->GetPeakMemUsageInMB());
  return status;
}

TfLiteStatus CulpritFinder::PrepareCulpritFinder() {
  LITERT_ASSIGN_OR_RETURN(
      interpreter_,
      interpreter_handler_->PrepareInterpreter(
          tflite::tools::CreateNullDelegate()),
      AsTfLiteStatus(_ << "Failed to prepare interpreter."));

  LITERT_LOG(LITERT_INFO, "Reference interpreter prepared");

  LITERT_ASSIGN_OR_RETURN(model_metadata_,
                          ModelMetadata::Create(interpreter_.get()),
                          AsTfLiteStatus(_ << "Failed to create model info."));

  input_manager_ = std::make_unique<TfliteInputManager>(interpreter_.get());
  if (input_manager_->PrepareInputData() != kTfLiteOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to prepare input data");
    return kTfLiteError;
  }
  if (interpreter_handler_->RunInference(*interpreter_, *input_manager_) !=
      kTfLiteOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to run reference inference");
    return kTfLiteError;
  }
  LITERT_LOG(LITERT_INFO, "Reference inference run completed!");
  return kTfLiteOk;
}

bool CulpritFinder::CulpritSearchMatchCondition(
    const OverallStat& overall_stat) {
  if (params_.Get<bool>(kFindNanFlag) &&
      !overall_stat.nan_output_indices.empty()) {
    return true;
  }
  if (params_.Get<bool>(kFindNumericErrorFlag) &&
      overall_stat.total_error >= params_.Get<float>(kMinNumericErrorFlag)) {
    return true;
  }
  if (overall_stat.is_crash) {
    return true;
  }
  return false;
}

TfLiteStatus CulpritFinder::RunCulpritFinderBinarySearch() {
  if (PrepareCulpritFinder() != kTfLiteOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to prepare culprit finder");
    return kTfLiteError;
  }
  int start_node = 0;
  int end_node = interpreter_->nodes_size();

  OverallStat temp_overall_stat;
  if (CalculateErrorStats(start_node, end_node, temp_overall_stat) !=
      kTfLiteOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to calculate error stats");
    return kTfLiteError;
  } else if (!CulpritSearchMatchCondition(temp_overall_stat)) {
    LITERT_LOG(LITERT_INFO, "No nan outputs/numeric errors found");
    return kTfLiteOk;
  }

  if (params_.Get<bool>(kBinarySearchReverseSweepFlag)) {
    end_node = BinarySearchFindEndNode(start_node, end_node);
    LITERT_LOG(LITERT_INFO, "### Found min end_node: %d", end_node);
    start_node = BinarySearchFindStartNode(start_node, end_node);
  } else {
    start_node = BinarySearchFindStartNode(start_node, end_node);
    LITERT_LOG(LITERT_INFO, "### Found max start_node: %d", start_node);
    end_node = BinarySearchFindEndNode(start_node, end_node);
  }

  LITERT_LOG(LITERT_INFO, "### Culprit node range: [ %d - %d ]", start_node,
             end_node);
  NodeRangeAnalysis(start_node, end_node);
  return kTfLiteOk;
}

int CulpritFinder::BinarySearchFindStartNode(int start_node, int end_node) {
  int start_node_range_start = start_node;
  int start_node_range_end = end_node;
  while (start_node_range_start <= start_node_range_end) {
    OverallStat overall_stat;
    const int mid_node =
        std::floor((start_node_range_start + start_node_range_end) / 2);
    LITERT_LOG(LITERT_INFO,
               "Looking for start node in node range: [%d - %d] by computing "
               "error stats for range [%d - %d]",
               start_node_range_start, start_node_range_end, mid_node,
               end_node);
    const std::vector<int> output_tensors =
        model_metadata_->GetOutputTensorsOfNode(end_node);
    if (CalculateErrorStats(mid_node, end_node, output_tensors, overall_stat) !=
        kTfLiteOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to calculate error stats");
      return kTfLiteError;
    }
    if (CulpritSearchMatchCondition(overall_stat)) {
      start_node = mid_node;
      start_node_range_start = mid_node + 1;
    } else {
      start_node_range_end = mid_node - 1;
    }
  }
  return start_node;
}

int CulpritFinder::BinarySearchFindEndNode(int start_node, int end_node) {
  int end_node_range_start = start_node;
  int end_node_range_end = end_node;

  while (end_node_range_start <= end_node_range_end) {
    OverallStat overall_stat;
    const int mid_node =
        std::floor((end_node_range_start + end_node_range_end) / 2);
    LITERT_LOG(LITERT_INFO,
               "Looking for end node in node range: [%d - %d] by computing "
               "error stats for range [%d - %d]",
               end_node_range_start, end_node_range_end, start_node, mid_node);
    const std::vector<int> output_tensors =
        model_metadata_->GetOutputTensorsOfNode(mid_node);
    if (CalculateErrorStats(start_node, mid_node, output_tensors,
                            overall_stat) != kTfLiteOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to calculate error stats");
      return kTfLiteError;
    }
    if (CulpritSearchMatchCondition(overall_stat)) {
      end_node = mid_node;
      end_node_range_end = mid_node - 1;
    } else {
      end_node_range_start = mid_node + 1;
    }
  }
  return end_node;
}

TfLiteStatus CulpritFinder::RunCulpritFinderLinearSearch() {
  if (PrepareCulpritFinder() != kTfLiteOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to prepare culprit finder");
    return kTfLiteError;
  }
  std::vector<std::pair<int, int>> node_ranges;
  node_ranges.reserve(interpreter_->nodes_size());
  LITERT_LOG(LITERT_INFO, "Nodes size: %d", interpreter_->nodes_size());
  LITERT_LOG(LITERT_INFO, "Subgraphs size: %d", interpreter_->subgraphs_size());
  const std::vector<int> execution_plan = interpreter_->execution_plan();
  LITERT_LOG(LITERT_INFO, "Execution plan size: %d", execution_plan.size());
  const int batch_size = params_.Get<int>(kLinearSearchStrideSizeFlag);
  std::unordered_set<std::string> filter_node_names(
      absl::StrSplit(params_.Get<std::string>(kLinearSearchNodeFilterFlag), ',',
                     absl::SkipEmpty()));
  for (int i = 0; i < execution_plan.size() - batch_size + 1; ++i) {
    if (!filter_node_names.empty() &&
        filter_node_names.find(model_metadata_->GetNodeIdentifier(
            execution_plan[i], /*with_index=*/false)) ==
            filter_node_names.end()) {
      continue;
    }
    node_ranges.push_back(
        {execution_plan[i], execution_plan[i + batch_size - 1]});
  }
  LITERT_LOG(LITERT_INFO, "### Node ranges size: %d", node_ranges.size());
  for (const auto& [node_start, node_end] : node_ranges) {
    OverallStat overall_stat;
    const std::vector<int> output_tensors =
        model_metadata_->GetOutputTensorsOfNode(node_end);
    if (CalculateErrorStats(node_start, node_end, output_tensors,
                            overall_stat) != kTfLiteOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to calculate error stats");
      return kTfLiteError;
    }
    if (CulpritSearchMatchCondition(overall_stat)) {
      overall_stats_.push_back({overall_stat.total_error, overall_stat});
    }
    LITERT_LOG(LITERT_INFO, "Done with Node range: [%d - %d]", node_start,
               node_end);
  }
  MakeReport();
  return kTfLiteOk;
}

TfLiteStatus CulpritFinder::NodeRangeAnalysis(int start_node, int end_node) {
  // Once we find the smallest node range that causes a NaN/NumericDifference,
  // we want to drill down into each node in the range and see if we can narrow
  // down the culprit further. We can do this by looking at output tensors of
  // each individual node in the range. To do the above we simply add the output
  // tensors of the specific node as the output tensor of the model. This does 2
  // things.
  // 1. It makes the output tensor of the model available as the output of the
  // model for inspection.
  // 2. It also splits the delegate into 2 parts. One that contains the nodes
  // before the node of interest and another of the nodes after the node of
  // interest.
  // We then run the model with the resulting delegate(s) and see if the NaN
  // still persists. If it does, it means that this is a valid split and this
  // specific node is not fused on the delegate side. If it doesn't, it means
  // we can ignore the output tensors generated for this node and continue to
  // the next node. This process can be repeated multiple times to see which
  // node generates how much deviation.

  LITERT_LOG(LITERT_INFO, "Beginning NodeRangeAnalysis: %d - %d", start_node,
             end_node);
  const std::vector<int> node_ids =
      model_metadata_->GetNodeIdsInRange(start_node, end_node);
  std::vector<int> all_output_tensors = std::vector<int>();
  for (int node_id : node_ids) {
    std::vector<int> output_tensors =
        model_metadata_->GetOutputTensorsOfNode(node_id);
    all_output_tensors.insert(all_output_tensors.end(), output_tensors.begin(),
                              output_tensors.end());
  }

  for (int node_id : node_ids) {
    std::vector<int> output_tensors =
        model_metadata_->GetOutputTensorsOfNode(node_id);
    OverallStat overall_stat;
    if (CalculateErrorStats(start_node, node_id, output_tensors,
                            overall_stat) != kTfLiteOk) {
      LITERT_LOG(LITERT_ERROR, "Failed to calculate error stats");
      return kTfLiteError;
    }

    if (CulpritSearchMatchCondition(overall_stat)) {
      LogOverallStat(overall_stat);
    }

    LITERT_LOG(LITERT_INFO, "Done with Node range: [%d - %d] with node: %d",
               start_node, node_id, node_id);
  }

  return kTfLiteOk;
}

inline void CulpritFinder::LogOverallStat(const OverallStat& overall_stat) {
  LITERT_LOG(LITERT_INFO, "Overall stat:");

  if (overall_stat.delegated_node_range.first !=
      overall_stat.delegated_node_range.second) {
    LITERT_LOG(LITERT_INFO, "  Delegated node range: [%s, %s]",
               model_metadata_
                   ->GetNodeIdentifier(overall_stat.delegated_node_range.first,
                                       /*with_index=*/true)
                   .c_str(),
               model_metadata_
                   ->GetNodeIdentifier(overall_stat.delegated_node_range.second,
                                       /*with_index=*/true)
                   .c_str());
  } else {
    LITERT_LOG(LITERT_INFO, "  Delegated node range: %s",
               model_metadata_
                   ->GetNodeIdentifier(overall_stat.delegated_node_range.first,
                                       /*with_index=*/true)
                   .c_str());
  }
  LITERT_LOG(LITERT_INFO, "  Min elementwise error: %f",
             overall_stat.min_error);
  LITERT_LOG(LITERT_INFO, "  Max elementwise error: %f",
             overall_stat.max_error);
  LITERT_LOG(LITERT_INFO, "  Total average error: %f",
             overall_stat.total_error);
  LITERT_LOG(LITERT_INFO, "  NAN output indices: ");
  for (int nan_output_index : overall_stat.nan_output_indices) {
    LITERT_LOG(LITERT_INFO, "%s, ",
               model_metadata_->GetTensorIdentifier(nan_output_index).c_str());
  }
}

void CulpritFinder::MakeReport() {
  std::sort(
      overall_stats_.begin(), overall_stats_.end(),
      [](const std::pair<float, OverallStat>& a,
         const std::pair<float, OverallStat>& b) { return a.first > b.first; });

  std::unordered_map<std::string, std::vector<int>>
      node_type_to_overall_stats_index;
  std::unordered_map<std::string, std::vector<int>>
      node_type_to_overall_stats_index_with_nan;
  for (int i = 0; i < overall_stats_.size(); ++i) {
    const std::string node_type = model_metadata_->GetNodeIdentifier(
        overall_stats_[i].second.delegated_node_range.first,
        /*with_index=*/false);
    node_type_to_overall_stats_index[node_type].push_back(i);
    if (!overall_stats_[i].second.nan_output_indices.empty()) {
      node_type_to_overall_stats_index_with_nan[node_type].push_back(i);
    }
  }

  // Sort the node to overall stats index by the number of occurrences for each
  // node type.
  std::vector<std::pair<std::string, std::vector<int>>>
      sorted_node_type_to_overall_stats_index(
          node_type_to_overall_stats_index.begin(),
          node_type_to_overall_stats_index.end());
  std::sort(sorted_node_type_to_overall_stats_index.begin(),
            sorted_node_type_to_overall_stats_index.end(),
            [](const std::pair<std::string, std::vector<int>>& a,
               const std::pair<std::string, std::vector<int>>& b) {
              return a.second.size() > b.second.size();
            });

  // Sort the node to overall stats index with nan by the number of occurrences
  // for each node type.
  std::vector<std::pair<std::string, std::vector<int>>>
      sorted_node_type_to_overall_stats_index_with_nan(
          node_type_to_overall_stats_index_with_nan.begin(),
          node_type_to_overall_stats_index_with_nan.end());
  std::sort(sorted_node_type_to_overall_stats_index_with_nan.begin(),
            sorted_node_type_to_overall_stats_index_with_nan.end(),
            [](const std::pair<std::string, std::vector<int>>& a,
               const std::pair<std::string, std::vector<int>>& b) {
              return a.second.size() > b.second.size();
            });

  LITERT_LOG(LITERT_INFO, "CULPRIT FINDER REPORT");
  LITERT_LOG(LITERT_INFO,
             "-------------------------------------------------------------");
  LITERT_LOG(LITERT_INFO,
             "Total number of nodes with errors: "
             "%d",
             overall_stats_.size());

  if (params_.Get<int>(kLinearSearchReportCountFlag) <= 0) {
    LITERT_LOG(LITERT_INFO, "No linear search report count provided");
    return;
  }

  LITERT_LOG(LITERT_INFO,
             "Top %d node ranges sorted by error (node_range, op_name(s), "
             "input/output shapes, total_error):",
             params_.Get<int>(kLinearSearchReportCountFlag));
  for (int i = 0; i < overall_stats_.size() &&
                  i < params_.Get<int>(kLinearSearchReportCountFlag);
       ++i) {
    const int node_start_index =
        overall_stats_[i].second.delegated_node_range.first;
    const int node_end_index =
        overall_stats_[i].second.delegated_node_range.second;

    LITERT_LOG(
        LITERT_INFO, "%d - %d, %s - %s, %s, %f", node_start_index,
        node_end_index,
        model_metadata_
            ->GetNodeIdentifier(node_start_index, /*with_index=*/false)
            .c_str(),
        model_metadata_->GetNodeIdentifier(node_end_index, /*with_index=*/false)
            .c_str(),
        model_metadata_->GetNodeShapes(node_start_index).c_str(),
        overall_stats_[i].first);
  }

  LITERT_LOG(LITERT_INFO,
             "-------------------------------------------------------------");
  LITERT_LOG(LITERT_INFO, "Top %d node(s) with most errors (op_name, count):",
             params_.Get<int>(kLinearSearchReportCountFlag));
  for (int i = 0; i < sorted_node_type_to_overall_stats_index.size() &&
                  i < params_.Get<int>(kLinearSearchReportCountFlag);
       ++i) {
    LITERT_LOG(LITERT_INFO, "%s, %zu",
               sorted_node_type_to_overall_stats_index[i].first.c_str(),
               sorted_node_type_to_overall_stats_index[i].second.size());
  }

  LITERT_LOG(LITERT_INFO,
             "-------------------------------------------------------------");
  if (!sorted_node_type_to_overall_stats_index_with_nan.empty()) {
    LITERT_LOG(LITERT_INFO,
               "Top %d Node(s) signatures with most nans (op_name, count):",
               params_.Get<int>(kLinearSearchReportCountFlag));
    for (int i = 0;
         i < sorted_node_type_to_overall_stats_index_with_nan.size() &&
         i < params_.Get<int>(kLinearSearchReportCountFlag);
         ++i) {
      LITERT_LOG(
          LITERT_INFO, "%s, %zu",
          sorted_node_type_to_overall_stats_index_with_nan[i].first.c_str(),
          sorted_node_type_to_overall_stats_index_with_nan[i].second.size());
    }
  }
}

}  // namespace litert::tools
