#ifndef ODML_LITERT_LITERT_VENDORS_HAILO_DISPATCH_INVOCATION_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_HAILO_DISPATCH_INVOCATION_CONTEXT_H_

#include <cstdint>
#include <memory>
#include <vector>
#include <utility>

#include "litert/c/litert_common.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/hailo/dispatch/device_context.h"

#include "hailo/hailort.hpp"

class LiteRtDispatchInvocationContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchInvocationContextT>;

  ~LiteRtDispatchInvocationContextT() = default;

  static litert::Expected<Ptr> Create(
      LiteRtDispatchDeviceContextT& device_context,
      LiteRtDispatchExecutableType exec_type,
      const LiteRtMemBuffer* exec_bytecode_buffer,
      const char* function_name,
      int num_inputs,
      int num_outputs);

  litert::Expected<void> AttachInput(int index, LiteRtTensorBufferHandle handle);
  litert::Expected<void> AttachOutput(int index, LiteRtTensorBufferHandle handle);
  litert::Expected<void> DetachInput(int index, LiteRtTensorBufferHandle handle);
  litert::Expected<void> DetachOutput(int index, LiteRtTensorBufferHandle handle);

  litert::Expected<void> Invoke();

  // Requirements negotiation
  litert::Expected<LiteRtTensorBufferRequirements> GetInputRequirements(
      int input_index, const LiteRtRankedTensorType& tensor_type);
  litert::Expected<LiteRtTensorBufferRequirements> GetOutputRequirements(
      int output_index, const LiteRtRankedTensorType& tensor_type);

 private:
  explicit LiteRtDispatchInvocationContextT(
      LiteRtDispatchDeviceContextT& device_context,
      std::shared_ptr<hailort::ConfiguredNetworkGroup> network_group,
      std::vector<hailort::InputVStream> input_vstreams,
      std::vector<hailort::OutputVStream> output_vstreams)
      : device_context_(device_context),
        network_group_(network_group),
        input_vstreams_(std::move(input_vstreams)),
        output_vstreams_(std::move(output_vstreams)) {
    attached_inputs_.resize(input_vstreams_.size(), (LiteRtTensorBufferHandle)0);
    attached_outputs_.resize(output_vstreams_.size(), (LiteRtTensorBufferHandle)0);
  }

  LiteRtDispatchDeviceContextT& device_context_;
  std::shared_ptr<hailort::ConfiguredNetworkGroup> network_group_;
  std::vector<hailort::InputVStream> input_vstreams_;
  std::vector<hailort::OutputVStream> output_vstreams_;

  std::vector<LiteRtTensorBufferHandle> attached_inputs_;
  std::vector<LiteRtTensorBufferHandle> attached_outputs_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_HAILO_DISPATCH_INVOCATION_CONTEXT_H_
