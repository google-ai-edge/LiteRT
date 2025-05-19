/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "litert/samples/async_segmentation/segmentation_model.h"

#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <GLES2/gl2.h>
#include <GLES3/gl3.h>
#include <GLES3/gl32.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_event_type.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_event.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_requirements.h"
#include "litert/cc/options/accelerator_options.h"

// TODO(b/383176413): Add API to CompiledModel to create buffers of custom
// buffer type.
litert::Expected<std::vector<litert::TensorBuffer>> CreateGlInputBuffers(
    LiteRtEnvironment env, litert::CompiledModel& compiled_model,
    litert::Signature& signature) {
  LiteRtSubgraph subgraph_handle = signature.Subgraph();
  litert::Subgraph subgraph = litert::Subgraph(subgraph_handle);

  std::vector<litert::TensorBuffer> input_buffers;
  input_buffers.reserve(subgraph.Inputs().size());
  for (litert::Tensor& input_tensor : subgraph.Inputs()) {
    LITERT_ASSIGN_OR_RETURN(
        litert::TensorBufferRequirements input_buffer_requirements,
        compiled_model.GetInputBufferRequirements(signature.Key(),
                                                  input_tensor.Name()));
    LITERT_ASSIGN_OR_RETURN(litert::RankedTensorType ranked_tensor_type,
                            input_tensor.RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(size_t buffer_size,
                            input_buffer_requirements.BufferSize());
    LITERT_ASSIGN_OR_RETURN(auto input_buffer,
                            litert::TensorBuffer::CreateManaged(
                                env, kLiteRtTensorBufferTypeGlBuffer,
                                ranked_tensor_type, buffer_size));
    input_buffers.push_back(std::move(input_buffer));
  }
  return input_buffers;
}

// TODO(b/383176413): Add API to CompiledModel to create buffers of custom
// buffer type.
litert::Expected<std::vector<litert::TensorBuffer>> CreateGlOutputBuffers(
    LiteRtEnvironment env, litert::CompiledModel& compiled_model,
    litert::Signature& signature) {
  LiteRtSubgraph subgraph_handle = signature.Subgraph();
  litert::Subgraph subgraph = litert::Subgraph(subgraph_handle);

  std::vector<litert::TensorBuffer> output_buffers;
  output_buffers.reserve(subgraph.Outputs().size());
  for (litert::Tensor& output_tensor : subgraph.Outputs()) {
    LITERT_ASSIGN_OR_RETURN(
        litert::TensorBufferRequirements input_buffer_requirements,
        compiled_model.GetOutputBufferRequirements(signature.Key(),
                                                   output_tensor.Name()));
    LITERT_ASSIGN_OR_RETURN(litert::RankedTensorType ranked_tensor_type,
                            output_tensor.RankedTensorType());
    LITERT_ASSIGN_OR_RETURN(size_t buffer_size,
                            input_buffer_requirements.BufferSize());
    LITERT_ASSIGN_OR_RETURN(auto output_buffer,
                            litert::TensorBuffer::CreateManaged(
                                env, kLiteRtTensorBufferTypeGlBuffer,
                                ranked_tensor_type, buffer_size));
    output_buffers.push_back(std::move(output_buffer));
  }
  return output_buffers;
}

litert::Options CreateGpuOptions() {
  LITERT_ASSIGN_OR_ABORT(auto gpu_options, litert::GpuOptions::Create());
  LITERT_ABORT_IF_ERROR(
      gpu_options.SetDelegatePrecision(kLiteRtDelegatePrecisionFp32));
  LITERT_ABORT_IF_ERROR(
      gpu_options.SetBufferStorageType(kLiteRtDelegateBufferStorageTypeBuffer));

  LITERT_ASSIGN_OR_ABORT(litert::Options options, litert::Options::Create());
  options.SetHardwareAccelerators(kLiteRtHwAcceleratorGpu);
  options.AddOpaqueOptions(std::move(gpu_options));
  return options;
}

bool SegmentationModel::InitializeModel(const std::string& model_path,
                                        std::string npu_library_path) {
  std::cout << "SegmentationModel: Initializing model ... Path: " << model_path
            << std::endl;
  LITERT_ASSIGN_OR_ABORT(model_, litert::Model::CreateFromFile(model_path));
  LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create({}));

  switch (current_accelerator_) {
    case AcceleratorType::CPU: {
      LITERT_ASSIGN_OR_ABORT(
          compiled_model_,
          litert::CompiledModel::Create(env, model_, kLiteRtHwAcceleratorCpu));
      break;
    }
    case AcceleratorType::GPU: {
      if (use_gl_buffers_) {
        // If using GL buffers, we need to set specific options for GPU.
        litert::Options options = CreateGpuOptions();
        LITERT_ASSIGN_OR_ABORT(compiled_model_, litert::CompiledModel::Create(
                                                    env, model_, options));
      } else {
        LITERT_ASSIGN_OR_ABORT(compiled_model_,
                               litert::CompiledModel::Create(
                                   env, model_, kLiteRtHwAcceleratorGpu));
      }
      break;
    }
    case AcceleratorType::NPU: {
      // Environment setup.
      const std::vector<litert::Environment::Option> environment_options = {
        litert::Environment::Option{
            litert::Environment::OptionTag::DispatchLibraryDir,
            absl::string_view(npu_library_path),
        },
      };
      LITERT_ASSIGN_OR_ABORT(env,
                            litert::Environment::Create(environment_options));
      LITERT_ASSIGN_OR_ABORT(
          compiled_model_,
          litert::CompiledModel::Create(env, model_, kLiteRtHwAcceleratorNpu));
      break;
    }
  }

  env_ = std::make_unique<litert::Environment>(std::move(env));

  LITERT_ASSIGN_OR_ABORT(auto signatures, model_.GetSignatures());

  size_t signature_index = 0;

  if (use_gl_buffers_) {
    LITERT_ASSIGN_OR_ABORT(input_buffers_,
                           CreateGlInputBuffers(env.Get(), compiled_model_,
                                                signatures[signature_index]));

    LITERT_ASSIGN_OR_ABORT(output_buffers_,
                           CreateGlOutputBuffers(env.Get(), compiled_model_,
                                                 signatures[signature_index]));

  } else {
    LITERT_ASSIGN_OR_ABORT(input_buffers_,
                           compiled_model_.CreateInputBuffers(signature_index));

    LITERT_ASSIGN_OR_ABORT(
        output_buffers_, compiled_model_.CreateOutputBuffers(signature_index));
  }
  std::cout << "SegmentationModel: Model initialized." << std::endl;
  std::cout << "SegmentationModel: warming up model..." << std::endl;
  auto start_time = absl::Now();
  compiled_model_.Run(signature_index, input_buffers_, output_buffers_);
  auto end_time = absl::Now();
  auto duration = end_time - start_time;
  std::cout << "SegmentationModel: warming up took: " << duration
            << " microseconds" << std::endl;
  return true;
}

bool SegmentationModel::RunSegmentation() {
  std::cout << "SegmentationModel: Running segmentation on preprocessed "
               "buffer on accelerator: ";
  switch (current_accelerator_) {
    case AcceleratorType::GPU:
      std::cout << "GPU";
      break;
    case AcceleratorType::NPU:
      std::cout << "NPU";
      break;
    default:
      std::cout << "CPU";
      break;
  }
  std::cout << std::endl;

  if (use_gl_buffers_) {
    // Create and set event for input tensor buffer.
    std::cout << "SegmentationModel: Creating event for input tensor buffer."
              << std::endl;
    LITERT_ASSIGN_OR_ABORT(
        auto event,
        litert::Event::CreateManaged(env_->Get(), LiteRtEventTypeEglSyncFence));
    input_buffers_[0].SetEvent(std::move(event));

    bool async = false;
    auto execution_result =
        compiled_model_.RunAsync(0, input_buffers_, output_buffers_, async);
    std::cout << "SegmentationModel: Async execution: " << async
              << " LiteRT model." << std::endl;
    if (!execution_result.HasValue()) {
      std::cerr << "SegmentationModel: Failed to execute LiteRT model."
                << std::endl;
      return false;
    }
  } else {
    auto execution_result =
        compiled_model_.Run(0, input_buffers_, output_buffers_);
    std::cout << "SegmentationModel: Sync execution LiteRT model." << std::endl;
    if (!execution_result.HasValue()) {
      std::cerr << "SegmentationModel: Failed to execute LiteRT model."
                << std::endl;
      return false;
    }
  }
  return true;
}
