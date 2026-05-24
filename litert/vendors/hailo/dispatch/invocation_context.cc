#include "litert/vendors/hailo/dispatch/invocation_context.h"

#include <vector>
#include <string>
#include <memory>
#include <cstring>

#include "litert/c/internal/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/util/tensor_type_util.h"

litert::Expected<LiteRtDispatchInvocationContextT::Ptr>
LiteRtDispatchInvocationContextT::Create(
    LiteRtDispatchDeviceContextT& device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer,
    const char* function_name,
    int num_inputs,
    int num_outputs) {
  if (exec_bytecode_buffer == nullptr || exec_bytecode_buffer->base_addr == nullptr) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument, "Invalid exec bytecode buffer");
  }

  // Load HEF from buffer memory view.
  auto hef_exp = hailort::Hef::create_from_buffer(
      hailort::MemoryView(const_cast<void*>(exec_bytecode_buffer->base_addr), exec_bytecode_buffer->size));
  if (!hef_exp) {
    LITERT_LOG(LITERT_ERROR, "Failed to load HEF from buffer: status = %d", hef_exp.status());
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to load HEF");
  }
  auto hef = std::move(hef_exp.value());

  // Configure network group.
  auto configure_params_exp = device_context.vdevice().create_configure_params(hef);
  if (!configure_params_exp) {
    LITERT_LOG(LITERT_ERROR, "Failed to create configure params: status = %d", configure_params_exp.status());
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to create configure params");
  }
  auto configure_params = std::move(configure_params_exp.value());

  auto network_groups_exp = device_context.vdevice().configure(hef, configure_params);
  if (!network_groups_exp) {
    LITERT_LOG(LITERT_ERROR, "Failed to configure network group: status = %d", network_groups_exp.status());
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to configure network group");
  }
  auto network_groups = std::move(network_groups_exp.value());
  if (network_groups.empty()) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "No configured network groups found");
  }
  auto network_group = network_groups[0];

  // Configure Virtual Input/Output Streams parameters.
  auto input_vstreams_params_exp = network_group->make_input_vstream_params(
      false, HAILO_FORMAT_TYPE_AUTO, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_QUEUE_SIZE);
  if (!input_vstreams_params_exp) {
    LITERT_LOG(LITERT_ERROR, "Failed to make input vstream params: status = %d", input_vstreams_params_exp.status());
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to make input vstream params");
  }

  auto output_vstreams_params_exp = network_group->make_output_vstream_params(
      false, HAILO_FORMAT_TYPE_AUTO, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_QUEUE_SIZE);
  if (!output_vstreams_params_exp) {
    LITERT_LOG(LITERT_ERROR, "Failed to make output vstream params: status = %d", output_vstreams_params_exp.status());
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to make output vstream params");
  }

  // Create Virtual Streams.
  auto input_vstreams_exp = hailort::VStream::create_input_vstreams(*network_group, input_vstreams_params_exp.value());
  if (!input_vstreams_exp) {
    LITERT_LOG(LITERT_ERROR, "Failed to create input vstreams: status = %d", input_vstreams_exp.status());
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to create input vstreams");
  }

  auto output_vstreams_exp = hailort::VStream::create_output_vstreams(*network_group, output_vstreams_params_exp.value());
  if (!output_vstreams_exp) {
    LITERT_LOG(LITERT_ERROR, "Failed to create output vstreams: status = %d", output_vstreams_exp.status());
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to create output vstreams");
  }

  auto input_vstreams = std::move(input_vstreams_exp.value());
  auto output_vstreams = std::move(output_vstreams_exp.value());

  if (static_cast<int>(input_vstreams.size()) != num_inputs) {
    LITERT_LOG(LITERT_ERROR, "Mismatch in inputs: model has %d, LiteRT has %d",
               static_cast<int>(input_vstreams.size()), num_inputs);
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument, "Inputs count mismatch");
  }

  if (static_cast<int>(output_vstreams.size()) != num_outputs) {
    LITERT_LOG(LITERT_ERROR, "Mismatch in outputs: model has %d, LiteRT has %d",
               static_cast<int>(output_vstreams.size()), num_outputs);
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument, "Outputs count mismatch");
  }

  LITERT_LOG(LITERT_INFO, "Hailo InvocationContext initialized successfully.");

  return Ptr(new LiteRtDispatchInvocationContextT(
      device_context, network_group, std::move(input_vstreams), std::move(output_vstreams)));
}

litert::Expected<void>
LiteRtDispatchInvocationContextT::AttachInput(int index, LiteRtTensorBufferHandle handle) {
  if (index < 0 || index >= static_cast<int>(attached_inputs_.size())) {
    return litert::Unexpected(kLiteRtStatusErrorIndexOOB, "Input index out of bounds");
  }
  attached_inputs_[index] = handle;
  return {};
}

litert::Expected<void>
LiteRtDispatchInvocationContextT::AttachOutput(int index, LiteRtTensorBufferHandle handle) {
  if (index < 0 || index >= static_cast<int>(attached_outputs_.size())) {
    return litert::Unexpected(kLiteRtStatusErrorIndexOOB, "Output index out of bounds");
  }
  attached_outputs_[index] = handle;
  return {};
}

litert::Expected<void>
LiteRtDispatchInvocationContextT::DetachInput(int index, LiteRtTensorBufferHandle handle) {
  if (index < 0 || index >= static_cast<int>(attached_inputs_.size())) {
    return litert::Unexpected(kLiteRtStatusErrorIndexOOB, "Input index out of bounds");
  }
  attached_inputs_[index] = (LiteRtTensorBufferHandle)0;
  return {};
}

litert::Expected<void>
LiteRtDispatchInvocationContextT::DetachOutput(int index, LiteRtTensorBufferHandle handle) {
  if (index < 0 || index >= static_cast<int>(attached_outputs_.size())) {
    return litert::Unexpected(kLiteRtStatusErrorIndexOOB, "Output index out of bounds");
  }
  attached_outputs_[index] = (LiteRtTensorBufferHandle)0;
  return {};
}

litert::Expected<void>
LiteRtDispatchInvocationContextT::Invoke() {
  // Validate attachments.
  for (size_t i = 0; i < attached_inputs_.size(); ++i) {
    if (attached_inputs_[i] == (LiteRtTensorBufferHandle)0) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Input stream not attached");
    }
  }
  for (size_t i = 0; i < attached_outputs_.size(); ++i) {
    if (attached_outputs_[i] == (LiteRtTensorBufferHandle)0) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Output stream not attached");
    }
  }

  // Write inputs to virtual streams.
  for (size_t i = 0; i < input_vstreams_.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(void* host_addr, device_context_.GetHostMemoryAddress(attached_inputs_[i]));
    size_t frame_size = input_vstreams_[i].get_frame_size();
    auto status = input_vstreams_[i].write(hailort::MemoryView(host_addr, frame_size));
    if (status != HAILO_SUCCESS) {
      LITERT_LOG(LITERT_ERROR, "Failed to write to input vstream %zu: status = %d", i, status);
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to write input stream");
    }
  }

  // Read outputs from virtual streams.
  for (size_t i = 0; i < output_vstreams_.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(void* host_addr, device_context_.GetHostMemoryAddress(attached_outputs_[i]));
    size_t frame_size = output_vstreams_[i].get_frame_size();
    auto status = output_vstreams_[i].read(hailort::MemoryView(host_addr, frame_size));
    if (status != HAILO_SUCCESS) {
      LITERT_LOG(LITERT_ERROR, "Failed to read from output vstream %zu: status = %d", i, status);
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to read output stream");
    }
  }

  return {};
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetInputRequirements(
    int input_index, const LiteRtRankedTensorType& tensor_type) {
  LiteRtTensorBufferType supported_tensor_buffer_types[] = {
      kLiteRtTensorBufferTypeHostMemory,
  };
  int num_supported_types = sizeof(supported_tensor_buffer_types) / sizeof(supported_tensor_buffer_types[0]);

  auto buffer_size = litert::internal::GetNumPackedBytes(tensor_type);
  if (!buffer_size) {
    return litert::Unexpected(buffer_size.Error());
  }

  LiteRtTensorBufferRequirements requirements;
  auto status = device_context_.runtime_context()->create_tensor_buffer_requirements(
      num_supported_types, supported_tensor_buffer_types, *buffer_size, 0, nullptr, &requirements);
  if (status != kLiteRtStatusOk) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to create tensor buffer requirements");
  }

  return requirements;
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetOutputRequirements(
    int output_index, const LiteRtRankedTensorType& tensor_type) {
  LiteRtTensorBufferType supported_tensor_buffer_types[] = {
      kLiteRtTensorBufferTypeHostMemory,
  };
  int num_supported_types = sizeof(supported_tensor_buffer_types) / sizeof(supported_tensor_buffer_types[0]);

  auto buffer_size = litert::internal::GetNumPackedBytes(tensor_type);
  if (!buffer_size) {
    return litert::Unexpected(buffer_size.Error());
  }

  LiteRtTensorBufferRequirements requirements;
  auto status = device_context_.runtime_context()->create_tensor_buffer_requirements(
      num_supported_types, supported_tensor_buffer_types, *buffer_size, 0, nullptr, &requirements);
  if (status != kLiteRtStatusOk) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to create tensor buffer requirements");
  }

  return requirements;
}
