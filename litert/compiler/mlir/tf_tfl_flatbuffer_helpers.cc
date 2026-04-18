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
#include "litert/compiler/mlir/tf_tfl_flatbuffer_helpers.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "tflite/converter/converter_flags.pb.h"
#include "tflite/converter/model_flags.pb.h"
#include "tflite/converter/quantization/common/quantization_lib/quantization_config.h"
#include "tflite/converter/tools/optimize/reduced_precision_metadata.h"
#include "tflite/converter/types.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace litert {
namespace {

using ::tensorflow::quantization::PyFunctionLibrary;
using ::tflite::optimize::ReducedPrecisionSupport;

// Converts the tflite::IODataType to tensorflow::DataType. Only contains the
// conversion mapping for constants defined in TFLite Python API.
tensorflow::DataType ConvertIODataTypeToDataType(tflite::IODataType dtype) {
  switch (dtype) {
    case tflite::IODataType::FLOAT:
      return tensorflow::DT_FLOAT;
    case tflite::IODataType::FLOAT16:
      return tensorflow::DT_HALF;
    case tflite::IODataType::FLOAT64:
      return tensorflow::DT_DOUBLE;
    case tflite::IODataType::QUANTIZED_UINT8:
      return tensorflow::DT_QUINT8;
    case tflite::IODataType::QUANTIZED_INT8:
      return tensorflow::DT_QINT8;
    case tflite::IODataType::QUANTIZED_INT16:
      return tensorflow::DT_QINT16;
    case tflite::IODataType::INT8:
      return tensorflow::DT_INT8;
    case tflite::IODataType::INT16:
      return tensorflow::DT_INT16;
    case tflite::IODataType::UINT16:
      return tensorflow::DT_UINT16;
    case tflite::IODataType::INT32:
      return tensorflow::DT_INT32;
    case tflite::IODataType::UINT32:
      return tensorflow::DT_UINT32;
    case tflite::IODataType::INT64:
      return tensorflow::DT_INT64;
    case tflite::IODataType::UINT8:
      return tensorflow::DT_UINT8;
    case tflite::IODataType::UINT64:
      return tensorflow::DT_UINT64;
    case tflite::IODataType::STRING:
      return tensorflow::DT_STRING;
    case tflite::IODataType::BOOL:
      return tensorflow::DT_BOOL;
    case tflite::IODataType::COMPLEX64:
      return tensorflow::DT_COMPLEX64;
    case tflite::IODataType::COMPLEX128:
      return tensorflow::DT_COMPLEX128;
    case tflite::IODataType::RESOURCE:
      return tensorflow::DT_RESOURCE;
    case tflite::IODataType::VARIANT:
      return tensorflow::DT_VARIANT;
    default:
      return tensorflow::DT_INVALID;
  }
}

absl::StatusOr<std::pair<double, double>> InputStatsToMinMax(
    double mean, double std, tensorflow::DataType type) {
  // Only qint8 and quint8 are considered here.
  double qmin, qmax;
  if (type == tensorflow::DT_QUINT8) {
    qmin = 0.0;
    qmax = 255.0;
  } else if (type == tensorflow::DT_QINT8) {
    qmin = -128.0;
    qmax = 127.0;
  } else {
    return absl::InvalidArgumentError("Only int8 and uint8 are considered.");
  }
  return std::make_pair((qmin - mean) / std, (qmax - mean) / std);
}

}  // namespace

absl::Status PopulateQuantizationSpecs(
    const tflite::ModelFlags& model_flags,
    tflite::ConverterFlags& converter_flags,
    mlir::TFL::QuantizationSpecs* quant_specs,
    std::vector<std::string>* node_names, std::vector<std::string>* node_dtypes,
    std::vector<std::optional<std::vector<int>>>* node_shapes,
    std::vector<std::optional<double>>* node_mins,
    std::vector<std::optional<double>>* node_maxs) {
  quant_specs->inference_input_type =
      ConvertIODataTypeToDataType(converter_flags.inference_input_type());
  tensorflow::DataType inference_type =
      ConvertIODataTypeToDataType(converter_flags.inference_type());
  // Use non-float flag `inference_input_type` to override the `inference_type`
  // because we have to apply quantization to satisfy that.
  if (quant_specs->inference_input_type != tensorflow::DT_FLOAT) {
    inference_type = quant_specs->inference_input_type;
  }

  for (auto& flag : model_flags.input_arrays()) {
    node_names->push_back(flag.name());
    // TOCO doesn't required `data_type` to be filled for every input.
    // If it's not filled, make it an empty string so the importer will use
    // the data type in the NodeDef.
    auto tflite_data_type = flag.data_type();
    if (tflite_data_type == tflite::IODataType::IO_DATA_TYPE_UNKNOWN) {
      node_dtypes->push_back("");
    } else {
      node_dtypes->push_back(
          DataType_Name(ConvertIODataTypeToDataType(tflite_data_type)));
    }
    if (flag.shape().unknown_rank()) {
      node_shapes->push_back(std::nullopt);
    } else {
      node_shapes->push_back(std::vector<int>(flag.shape().dims().begin(),
                                              flag.shape().dims().end()));
    }
    // Currently, only UINT8 and INT8 require inputs stats
    if (inference_type == tensorflow::DT_QINT8 ||
        inference_type == tensorflow::DT_QUINT8) {
      if (flag.has_mean_value() && flag.has_std_value()) {
        TF_ASSIGN_OR_RETURN(
            auto min_max, InputStatsToMinMax(flag.mean_value(),
                                             flag.std_value(), inference_type));
        node_mins->push_back(min_max.first);
        node_maxs->push_back(min_max.second);
      } else {
        node_mins->push_back(std::nullopt);
        node_maxs->push_back(std::nullopt);
      }
    }
  }

  if (mlir::TFL::GetInputNodeQuantSpecs(*node_names, *node_mins, *node_maxs,
                                        inference_type, quant_specs)) {
    return absl::InvalidArgumentError("Failed to get input quant spec.");
  }

  // Some extra flag related to post training quantization. If post-training
  // quantization is enabled, `inference_type` and `inference_input_type` are
  // not used by MLIR passes.
  if (converter_flags.post_training_quantize()) {
    quant_specs->weight_quantization = true;
    quant_specs->disable_per_channel =
        converter_flags.disable_per_channel_quantization();
    if (converter_flags.quantize_to_float16()) {
      quant_specs->inference_type = tensorflow::DT_HALF;
      quant_specs->inference_input_type = tensorflow::DT_HALF;
    } else {
      quant_specs->inference_type = tensorflow::DT_QINT8;
      quant_specs->inference_input_type = tensorflow::DT_QINT8;
    }
  } else {
    // These flags are incompatible with post_training_quantize() as only
    // QAT models can provide required ranges.
    quant_specs->disable_infer_tensor_range =
        converter_flags.disable_infer_tensor_range();
    quant_specs->use_fake_quant_num_bits =
        converter_flags.use_fake_quant_num_bits();
  }

  // Add information about half-precision support if fp16 quantization applies.
  // TODO(b/195945955): Add e2e test for this.
  if (converter_flags.quantize_to_float16() ||
      converter_flags.allow_bfloat16()) {
    ReducedPrecisionSupport mask = ReducedPrecisionSupport::None;
    if (converter_flags.quantize_to_float16()) {
      mask |= ReducedPrecisionSupport::Float16Inference;
    }
    if (converter_flags.allow_bfloat16()) {
      mask |= ReducedPrecisionSupport::Bfloat16Inference;
    }
    if (converter_flags.accumulation_type() == tflite::IODataType::FLOAT16) {
      mask |= ReducedPrecisionSupport::Float16Accumulation;
    } else {
      mask |= ReducedPrecisionSupport::Float32Accumulation;
    }
    quant_specs->support_mask = mask;
  }

  // Other flags.
  if (converter_flags.has_default_ranges_min()) {
    quant_specs->default_ranges.first = converter_flags.default_ranges_min();
  }
  if (converter_flags.has_default_ranges_max()) {
    quant_specs->default_ranges.second = converter_flags.default_ranges_max();
  }
  quant_specs->enable_mlir_dynamic_range_quantizer =
      converter_flags.enable_mlir_dynamic_range_quantizer();
  quant_specs->enable_mlir_variable_quantization =
      converter_flags.enable_mlir_variable_quantization();
  quant_specs->disable_per_channel_for_dense_layers =
      converter_flags.disable_per_channel_quantization_for_dense_layers();
  return absl::OkStatus();
}

}  // namespace litert
