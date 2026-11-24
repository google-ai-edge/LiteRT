// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/backends/ir_backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "IR/QnnIrGraph.h"  // from @qairt
#include "QnnBackend.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/qualcomm/core/backends/graph_config_builder.h"
#include "litert/vendors/qualcomm/core/backends/qnn_backend.h"
#include "litert/vendors/qualcomm/core/common.h"
#include "litert/vendors/qualcomm/core/schema/soc_table.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/utils/miscs.h"
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {
namespace {

// Per-channel layout: element i belongs to channel (i / inner) % axis_dim.
struct DequantParams {
  std::vector<float> scales;
  std::vector<std::int32_t> zero_points;
  std::size_t inner = 1;  // Elements between consecutive channel indices.
};

bool GetDequantParams(const TensorWrapper& tensor, DequantParams& params) {
  if (const auto* per_tensor = std::get_if<ScaleOffsetQuantizeParamsWrapper>(
          &tensor.GetQuantParams())) {
    params.scales = {per_tensor->GetScale()};
    params.zero_points = {per_tensor->GetZeroPoint()};
    return true;
  }
  if (const auto* per_channel =
          std::get_if<AxisScaleOffsetQuantizeParamsWrapper>(
              &tensor.GetQuantParams())) {
    params.scales = per_channel->GetScales();
    params.zero_points = per_channel->GetZeroPoints();
    const auto& dims = tensor.GetDimensions();
    const auto axis = static_cast<std::size_t>(per_channel->GetAxis());
    if (axis >= dims.size() || params.scales.size() != dims[axis]) {
      return false;
    }
    for (std::size_t i = axis + 1; i < dims.size(); ++i) {
      params.inner *= dims[i];
    }
    return true;
  }
  return false;
}

template <typename T>
bool DequantizeStaticData(TensorWrapper& tensor) {
  DequantParams params;
  const auto data = tensor.GetTensorData<T>();
  if (!data || !GetDequantParams(tensor, params)) {
    return false;
  }
  std::vector<float> values(data->size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    const std::size_t channel = (i / params.inner) % params.scales.size();
    values[i] = Dequantize((*data)[i], params.scales[channel],
                           params.zero_points[channel]);
  }
  tensor.OverrideDataType(QNN_DATATYPE_FLOAT_32);
  return tensor.SetTensorData<float>(values);
}

bool StripTensor(TensorWrapper& tensor) {
  if (!tensor.IsQuant()) {
    return true;
  }
  if (tensor.IsTensorStatic()) {
    bool is_dequantized = false;
    switch (tensor.GetDataType()) {
      case QNN_DATATYPE_SFIXED_POINT_8:
        is_dequantized = DequantizeStaticData<std::int8_t>(tensor);
        break;
      case QNN_DATATYPE_UFIXED_POINT_8:
        is_dequantized = DequantizeStaticData<std::uint8_t>(tensor);
        break;
      case QNN_DATATYPE_SFIXED_POINT_16:
        is_dequantized = DequantizeStaticData<std::int16_t>(tensor);
        break;
      case QNN_DATATYPE_UFIXED_POINT_16:
        is_dequantized = DequantizeStaticData<std::uint16_t>(tensor);
        break;
      case QNN_DATATYPE_SFIXED_POINT_32:
        is_dequantized = DequantizeStaticData<std::int32_t>(tensor);
        break;
      default:
        break;
    }
    if (!is_dequantized) {
      QNN_LOG_ERROR("Cannot strip quantization of static tensor: %s",
                    tensor.ToString().c_str());
      return false;
    }
  } else {
    // Non-static tensors carry no data, so only the type needs to change.
    tensor.OverrideDataType(QNN_DATATYPE_FLOAT_32);
  }
  tensor.SetQuantParams(UndefinedQuantizeParamsWrapper());
  return true;
}

}  // namespace

IrBackend::IrBackend(const QNN_INTERFACE_VER_TYPE* qnn_api)
    : QnnBackend(qnn_api) {}

bool IrBackend::Init(const Options& options,
                     std::optional<::qnn::SocInfo> soc_info) {
  // Log Handle
  auto local_log_handle = CreateLogHandle(options.GetLogLevel());
  if (!local_log_handle && options.GetLogLevel() != ::qnn::LogLevel::kOff) {
    QNN_LOG_ERROR("Failed to create log handle!");
    return false;
  }

  // Backend Handle
  std::array<const QnnBackend_Config_t*, 1> backend_configs = {nullptr};

  auto local_backend_handle = CreateBackendHandle(
      local_log_handle.get(), absl::MakeSpan(backend_configs));
  if (!local_backend_handle) {
    QNN_LOG_ERROR("Failed to create backend handle!");
    return false;
  }

  if (soc_info) {
    soc_info_ = *soc_info;
  }

  log_handle_ = std::move(local_log_handle);
  backend_handle_ = std::move(local_backend_handle);
  return true;
}

GraphConfigBuilder IrBackend::BuildGraphConfigs(
    const Options& options, absl::string_view qnn_graph_name) {
  GraphConfigBuilder config_builder;

  std::filesystem::path dlc_dir = std::string(options.GetDlcDir());
  auto& dlc_path = config_builder.Store<std::string>(
      (dlc_dir / absl::StrCat(qnn_graph_name, ".dlc")).generic_string());

  QnnIrGraph_CustomConfig_t custom_config = QNN_IR_GRAPH_CUSTOM_CONFIG_INIT;
  custom_config.option = QNN_IR_GRAPH_CONFIG_OPTION_SERIALIZATION;
  custom_config.serializationOption.serializationType =
      QNN_IR_GRAPH_SERIALIZATION_TYPE_FLAT_BUFFER;
  custom_config.serializationOption.outputPath = dlc_path.c_str();
  config_builder.AddCustomConfig(custom_config);

  return config_builder;
}

void IrQuantStripper::Record(absl::Span<const TensorWrapperRef> tensors) {
  for (const TensorWrapper& tensor : tensors) {
    if (tensor.IsQuant()) {
      original_.emplace(&tensor, tensor.GetQuantParams());
    }
  }
}

bool IrQuantStripper::Apply(TensorPool& tensor_pool) {
  bool all_stripped = true;
  tensor_pool.ForEach([&](TensorWrapper& tensor) {
    // A builder replaced the params with a placeholder (scale 0); restore.
    const auto recorded = original_.find(&tensor);
    const auto* current =
        std::get_if<ScaleOffsetQuantizeParamsWrapper>(&tensor.GetQuantParams());
    if (recorded != original_.end() && current != nullptr &&
        current->GetScale() == 0.0f) {
      tensor.SetQuantParams(recorded->second);
    }
    all_stripped &= StripTensor(tensor);
  });
  return all_stripped;
}

}  // namespace qnn
