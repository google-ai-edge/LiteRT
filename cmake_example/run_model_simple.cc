#include <cstdlib>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "litert/c/litert_common.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"

ABSL_FLAG(std::string, graph, "", "Model filename");
ABSL_FLAG(std::string, accelerator, "cpu", "Accelerators: cpu,gpu,npu");
ABSL_FLAG(size_t, signature_index, 0, "Signature index");
ABSL_FLAG(bool, compare_numerical, false, "Fill tensors with sample values");
ABSL_FLAG(size_t, sample_size, 5, "Sample size to print");

namespace litert {
namespace {

HwAcceleratorSet GetAccelerator() {
  HwAcceleratorSet acc(static_cast<int>(HwAccelerators::kNone));

  for (absl::string_view a : absl::StrSplit(absl::GetFlag(FLAGS_accelerator), ',')) {
    if (a == "gpu") acc |= HwAccelerators::kGpu;
    else if (a == "npu") acc |= HwAccelerators::kNpu;
    else acc |= HwAccelerators::kCpu;
  }

  return acc;
}

Expected<Environment> GetEnvironment() {
  return Environment::Create({});
}

Expected<Options> GetOptions() {
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  options.SetHardwareAccelerators(GetAccelerator());
  return options;
}

size_t GetTotalElements(const TensorBuffer& buffer) {
  auto type = buffer.TensorType().value();
  const auto& layout = type.Layout();

  size_t total = 1;
  for (size_t d = 0; d < layout.Rank(); ++d)
    total *= layout.Dimensions()[d];

  return total;
}

Expected<void> FillInputBuffer(TensorBuffer& buffer) {
  if (!absl::GetFlag(FLAGS_compare_numerical))
    return {};

  const size_t total = GetTotalElements(buffer);

  std::vector<float> data(total);

  for (size_t i = 0; i < total; ++i)
    data[i] = static_cast<float>(i % 10) * 0.1f;

  return buffer.Write<float>(absl::MakeConstSpan(data));
}

Expected<void> RunModel() {

  const std::string& graph = absl::GetFlag(FLAGS_graph);

  if (graph.empty())
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Model filename empty. Use --graph.");

  LITERT_ASSIGN_OR_RETURN(auto env, GetEnvironment());
  LITERT_ASSIGN_OR_RETURN(auto options, GetOptions());

  ABSL_LOG(INFO) << "Model: " << graph;

  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model,
      CompiledModel::Create(env, graph, options));

  const size_t sig = absl::GetFlag(FLAGS_signature_index);

  ABSL_LOG(INFO) << "Signature index: " << sig;

  LITERT_ASSIGN_OR_RETURN(
      auto input_buffers,
      compiled_model.CreateInputBuffers(sig));

  for (auto& buf : input_buffers)
    LITERT_RETURN_IF_ERROR(FillInputBuffer(buf));

  LITERT_ASSIGN_OR_RETURN(
      auto output_buffers,
      compiled_model.CreateOutputBuffers(sig));

  ABSL_LOG(INFO) << "Running model...";

  LITERT_RETURN_IF_ERROR(
      compiled_model.Run(sig, input_buffers, output_buffers));

  ABSL_LOG(INFO) << "Model run completed";

  return {};
}

}  
}  

int main(int argc, char** argv) {

  absl::ParseCommandLine(argc, argv);

  auto result = litert::RunModel();

  if (!result) {
    ABSL_LOG(ERROR) << result.Error().Message();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
