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

#include <sys/eventfd.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_event.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/core/dispatch_op_schema.h"
#include "litert/core/model/model.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "tflite/schema/schema_generated.h"

// A fake vendor consumes the real tensor-buffer and callback APIs, but never
// loads rpcmem or QNN. Run this test on Android, where FastRPC is enabled.
struct LiteRtDispatchDeviceContextT {
  std::unordered_map<LiteRtTensorBufferHandle, LiteRtTensorBuffer> buffers;
  uint64_t next = 1;
};
struct LiteRtDispatchInvocationContextT {
  LiteRtDispatchDeviceContext device;
  LiteRtTensorBufferHandle inputs[2]{};
  LiteRtTensorBufferHandle output = 0;
  LiteRtOptions options = nullptr;
};
extern "C" LiteRtStatus (*LiteRtStaticLinkedDispatchGetApi)(LiteRtDispatchApi*);

namespace {
struct Counters {
  int devices = 0;
  int invocations = 0;
  int registrations = 0;
  int unregistrations = 0;
  int attaches = 0;
  int invokes = 0;
  int frees = 0;
  int options_set = 0;
  int options_reset = 0;
  bool fail_attach = false;
  bool fail_register = false;
  bool fail_invoke = false;
  bool fail_create = false;
  bool fail_detach = false;
  bool fail_unregister = false;
} counts;

LiteRtStatus DeviceCreate(const LiteRtRuntimeContext* runtime, LiteRtOptions,
                          LiteRtDispatchDeviceContext* device) {
  EXPECT_EQ(runtime->abi_header.struct_size, sizeof(LiteRtRuntimeContext));
  EXPECT_NE(runtime->get_tensor_buffer_fast_rpc_buffer, nullptr);
  EXPECT_EQ(runtime->wrap_delegate, nullptr);
  EXPECT_EQ(runtime->get_external_litert_buffer_context_tensor_buffer, nullptr);
  *device = new LiteRtDispatchDeviceContextT;
  ++counts.devices;
  return kLiteRtStatusOk;
}
LiteRtStatus DeviceDestroy(LiteRtDispatchDeviceContext device) {
  EXPECT_TRUE(device->buffers.empty());
  delete device;
  --counts.devices;
  return kLiteRtStatusOk;
}
LiteRtStatus InvocationCreate(const LiteRtRuntimeContext*,
                              LiteRtDispatchDeviceContext device,
                              LiteRtDispatchExecutableType type,
                              const LiteRtMemBuffer* binary, const char* name,
                              int inputs, int outputs,
                              LiteRtDispatchInvocationContext* invocation) {
  EXPECT_EQ(type, kLiteRtDispatchExecutableTypeMlModel);
  EXPECT_EQ(binary->size, 4);
  EXPECT_EQ(static_cast<const uint8_t*>(binary->base_addr)[binary->offset],
            0xab);
  EXPECT_STREQ(name, "compiled_graph");
  EXPECT_EQ(inputs, 2);
  EXPECT_EQ(outputs, 1);
  if (counts.fail_create) return kLiteRtStatusErrorRuntimeFailure;
  *invocation = new LiteRtDispatchInvocationContextT{device};
  ++counts.invocations;
  return kLiteRtStatusOk;
}
LiteRtStatus InvocationDestroy(LiteRtDispatchInvocationContext invocation) {
  EXPECT_TRUE(invocation->device->buffers.empty());
  delete invocation;
  --counts.invocations;
  return kLiteRtStatusOk;
}
LiteRtStatus Requirements(LiteRtDispatchInvocationContext, int,
                          const LiteRtRankedTensorType*,
                          LiteRtTensorBufferRequirements* requirements) {
  const LiteRtTensorBufferType types[] = {kLiteRtTensorBufferTypeDmaBuf,
                                          kLiteRtTensorBufferTypeFastRpc};
  // More than the packed tensor size: the runtime must respect the vendor.
  return LiteRtCreateTensorBufferRequirements(2, types, 64, 0, nullptr,
                                              requirements);
}
LiteRtStatus Register(LiteRtDispatchDeviceContext device,
                      LiteRtTensorBuffer buffer,
                      LiteRtTensorBufferHandle* handle) {
  if (counts.fail_register) return kLiteRtStatusErrorRuntimeFailure;
  for (const auto& entry : device->buffers) {
    if (entry.second == buffer) {
      *handle = entry.first;
      return kLiteRtStatusOk;
    }
  }
  *handle = device->next++;
  device->buffers.emplace(*handle, buffer);
  ++counts.registrations;
  return kLiteRtStatusOk;
}
LiteRtStatus Unregister(LiteRtDispatchDeviceContext device,
                        LiteRtTensorBufferHandle handle) {
  if (counts.fail_unregister) return kLiteRtStatusErrorRuntimeFailure;
  EXPECT_EQ(device->buffers.erase(handle), 1);
  ++counts.unregistrations;
  return kLiteRtStatusOk;
}
LiteRtStatus AttachInput(LiteRtDispatchInvocationContext invocation, int index,
                         LiteRtTensorBufferHandle handle) {
  if (counts.fail_attach) return kLiteRtStatusErrorRuntimeFailure;
  invocation->inputs[index] = handle;
  ++counts.attaches;
  return kLiteRtStatusOk;
}
LiteRtStatus AttachOutput(LiteRtDispatchInvocationContext invocation, int,
                          LiteRtTensorBufferHandle handle) {
  invocation->output = handle;
  ++counts.attaches;
  return kLiteRtStatusOk;
}
LiteRtStatus Detach(LiteRtDispatchInvocationContext, int,
                    LiteRtTensorBufferHandle) {
  return counts.fail_detach ? kLiteRtStatusErrorRuntimeFailure
                            : kLiteRtStatusOk;
}
LiteRtStatus Invoke(LiteRtDispatchInvocationContext invocation) {
  ++counts.invokes;
  if (counts.fail_invoke) return kLiteRtStatusErrorRuntimeFailure;
  void* addresses[3];
  int fd;
  LiteRtGetTensorBufferFastRpcBuffer(
      invocation->device->buffers.at(invocation->inputs[0]), &addresses[0],
      &fd);
  LiteRtGetTensorBufferFastRpcBuffer(
      invocation->device->buffers.at(invocation->inputs[1]), &addresses[1],
      &fd);
  LiteRtGetTensorBufferFastRpcBuffer(
      invocation->device->buffers.at(invocation->output), &addresses[2], &fd);
  for (int i = 0; i < 2; ++i) {
    static_cast<float*>(addresses[2])[i] =
        static_cast<float*>(addresses[0])[i] -
        static_cast<float*>(addresses[1])[i];
  }
  return kLiteRtStatusOk;
}
LiteRtStatus SetOptions(LiteRtDispatchInvocationContext invocation,
                        LiteRtOptions options) {
  invocation->options = options;
  if (options)
    ++counts.options_set;
  else
    ++counts.options_reset;
  return kLiteRtStatusOk;
}
LiteRtStatus GetApi(LiteRtDispatchApi* api) {
  static LiteRtDispatchInterface interface{};
  interface.initialize = [](const LiteRtRuntimeContext*, LiteRtEnvironment,
                            LiteRtOptions) { return kLiteRtStatusOk; };
  interface.get_vendor_id = [](const char** vendor) {
    *vendor = "Qualcomm";
    return kLiteRtStatusOk;
  };
  interface.device_context_create = DeviceCreate;
  interface.device_context_destroy = DeviceDestroy;
  interface.invocation_context_create = InvocationCreate;
  interface.invocation_context_destroy = InvocationDestroy;
  interface.get_input_requirements = Requirements;
  interface.get_output_requirements = Requirements;
  interface.register_tensor_buffer = Register;
  interface.unregister_tensor_buffer = Unregister;
  interface.attach_input = AttachInput;
  interface.attach_output = AttachOutput;
  interface.detach_input = Detach;
  interface.detach_output = Detach;
  interface.invoke = Invoke;
  interface.invocation_context_set_options = SetOptions;
  api->version = {LITERT_API_VERSION_MAJOR, LITERT_API_VERSION_MINOR,
                  LITERT_API_VERSION_PATCH};
  api->interface = &interface;
  return kLiteRtStatusOk;
}

std::vector<uint8_t> MakeModel(bool dynamic = false, bool cpu = false,
                               bool two_ops = false,
                               bool two_signatures = false,
                               bool malformed_options = false) {
  tflite::ModelT model;
  model.version = 3;
  model.buffers.push_back(std::make_unique<tflite::BufferT>());
  auto code = std::make_unique<tflite::OperatorCodeT>();
  code->builtin_code =
      cpu ? tflite::BuiltinOperator_ADD : tflite::BuiltinOperator_CUSTOM;
  code->deprecated_builtin_code = code->builtin_code;
  code->custom_code = cpu ? "" : "DISPATCH_OP";
  model.operator_codes.push_back(std::move(code));
  auto graph = std::make_unique<tflite::SubGraphT>();
  graph->inputs = {0, 1};
  graph->outputs = {2};
  for (int i = 0; i < 3; ++i) {
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->type = tflite::TensorType_FLOAT32;
    tensor->shape = {2};
    tensor->has_rank = true;
    tensor->name = std::to_string(i);
    if (dynamic) tensor->shape_signature = {-1};
    graph->tensors.push_back(std::move(tensor));
  }
  auto op = std::make_unique<tflite::OperatorT>();
  op->inputs = {1, 0};  // Deliberately different from signature order.
  op->outputs = {2};
  op->custom_options_format = tflite::CustomOptionsFormat_FLEXBUFFERS;
  graph->operators.push_back(std::move(op));
  if (two_ops)
    graph->operators.push_back(
        std::make_unique<tflite::OperatorT>(*graph->operators[0]));
  model.subgraphs.push_back(std::move(graph));
  if (two_signatures) {
    for (int i = 0; i < 2; ++i) {
      auto signature = std::make_unique<tflite::SignatureDefT>();
      signature->signature_key = std::to_string(i);
      for (int j = 0; j < 3; ++j) {
        auto map = std::make_unique<tflite::TensorMapT>();
        map->name = std::to_string(j);
        map->tensor_index = j;
        (j < 2 ? signature->inputs : signature->outputs)
            .push_back(std::move(map));
      }
      model.signature_defs.push_back(std::move(signature));
    }
  }
  std::vector<uint8_t> result;
  size_t offset = 1;
  for (int pass = 0; pass < 2; ++pass) {
    auto custom =
        litert::internal::MakeDispatchOpOptions({4, offset, "compiled_graph"});
    for (auto& operation : model.subgraphs[0]->operators) {
      operation->custom_options.assign(custom.Data(),
                                       custom.Data() + custom.Size());
      if (malformed_options) operation->custom_options = {1, 2, 3};
    }
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
    result.assign(builder.GetBufferPointer(),
                  builder.GetBufferPointer() + builder.GetSize());
    if (pass == 0) offset = (result.size() + 63) & ~size_t{63};
  }
  result.resize(offset);
  result.insert(result.end(), {0xab, 0xcd, 0xef, 0x01});
  return result;
}

class DirectDispatchTest : public testing::Test {
 protected:
  void SetUp() override {
    counts = {};
    LiteRtStaticLinkedDispatchGetApi = GetApi;
    ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &env), kLiteRtStatusOk);
    ASSERT_EQ(LiteRtCreateOptions(&options), kLiteRtStatusOk);
    ASSERT_EQ(
        LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorNpu),
        kLiteRtStatusOk);
  }
  void TearDown() override {
    for (auto buffer : buffers) LiteRtDestroyTensorBuffer(buffer);
    LiteRtDestroyCompiledModel(compiled);
    LiteRtDestroyModel(model);
    LiteRtDestroyOptions(options);
    LiteRtDestroyEnvironment(env);
    EXPECT_EQ(counts.devices, 0);
    EXPECT_EQ(counts.invocations, 0);
    EXPECT_EQ(counts.registrations, counts.unregistrations);
    EXPECT_EQ(counts.frees, buffers.size());
  }
  void Load(std::vector<uint8_t> bytes = MakeModel()) {
    binary = std::move(bytes);
    ASSERT_EQ(
        LiteRtCreateModelFromBuffer(env, binary.data(), binary.size(), &model),
        kLiteRtStatusOk);
  }
  LiteRtStatus Compile() {
    return LiteRtCreateCompiledModel(env, model, options, &compiled);
  }
  LiteRtTensorBuffer Buffer(float a = 0, float b = 0, size_t size = 64,
                            int dimension = 2, size_t offset = 0) {
    void* data = nullptr;
    EXPECT_EQ(posix_memalign(&data, 64, size), 0);
    static_cast<float*>(data)[0] = a;
    static_cast<float*>(data)[1] = b;
    LiteRtRankedTensorType type{};
    type.element_type = kLiteRtElementTypeFloat32;
    type.layout.rank = 1;
    type.layout.dimensions[0] = dimension;
    LiteRtTensorBuffer buffer = nullptr;
    EXPECT_EQ(LiteRtCreateTensorBufferFromFastRpcBuffer(
                  &type, data, 42, size, offset,
                  [](void* address) {
                    free(address);
                    ++counts.frees;
                  },
                  &buffer),
              kLiteRtStatusOk);
    buffers.push_back(buffer);
    return buffer;
  }
  LiteRtStatus Run(size_t signature = 0) {
    LiteRtTensorBuffer inputs[] = {buffers[0], buffers[1]};
    return LiteRtRunCompiledModel(compiled, signature, 2, inputs, 1,
                                  &buffers[2]);
  }
  LiteRtEnvironment env = nullptr;
  LiteRtOptions options = nullptr;
  LiteRtModel model = nullptr;
  LiteRtCompiledModel compiled = nullptr;
  std::vector<uint8_t> binary;
  std::vector<LiteRtTensorBuffer> buffers;
};

TEST_F(DirectDispatchTest, RunsWithReorderedPortsAndReusesRegistrations) {
  Load();
  ASSERT_EQ(Compile(), kLiteRtStatusOk);
  Buffer(1, 2);
  Buffer(10, 20);
  auto output = Buffer();
  ASSERT_EQ(Run(), kLiteRtStatusOk);
  ASSERT_EQ(Run(), kLiteRtStatusOk);
  EXPECT_EQ(counts.registrations, 3);
  EXPECT_EQ(counts.attaches, 3);
  void* address;
  ASSERT_EQ(
      LiteRtLockTensorBuffer(output, &address, kLiteRtTensorBufferLockModeRead),
      kLiteRtStatusOk);
  EXPECT_EQ(static_cast<float*>(address)[0], 9);
  EXPECT_EQ(static_cast<float*>(address)[1], 18);
  EXPECT_EQ(LiteRtUnlockTensorBuffer(output), kLiteRtStatusOk);
}
TEST_F(DirectDispatchTest, RequirementsOnlyAdvertiseFastRpcAndPreservePadding) {
  Load();
  ASSERT_EQ(Compile(), kLiteRtStatusOk);
  LiteRtTensorBufferRequirements requirements;
  ASSERT_EQ(LiteRtGetCompiledModelInputBufferRequirements(compiled, 0, 0,
                                                          &requirements),
            kLiteRtStatusOk);
  int count;
  size_t size;
  LiteRtTensorBufferType type;
  LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(requirements,
                                                           &count);
  LiteRtGetTensorBufferRequirementsBufferSize(requirements, &size);
  LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(requirements, 0,
                                                             &type);
  EXPECT_EQ(count, 1);
  EXPECT_EQ(size, 64);
  EXPECT_EQ(type, kLiteRtTensorBufferTypeFastRpc);
}
TEST_F(DirectDispatchTest, RejectsCpuModel) {
  Load(MakeModel(false, true));
  EXPECT_EQ(Compile(), kLiteRtStatusErrorUnsupported);
}
TEST_F(DirectDispatchTest, RejectsMultiplePartitions) {
  Load(MakeModel(false, false, true));
  EXPECT_EQ(Compile(), kLiteRtStatusErrorUnsupported);
}
TEST_F(DirectDispatchTest, RejectsDynamicShapes) {
  Load(MakeModel(true));
  EXPECT_EQ(Compile(), kLiteRtStatusErrorUnsupported);
}
TEST_F(DirectDispatchTest, RejectsInvalidContextRangeWithoutOverflow) {
  Load();
  auto custom = litert::internal::MakeDispatchOpOptions(
      {64, std::numeric_limits<size_t>::max() - 8, "compiled_graph"});
  model->Subgraph(0).Ops()[0]->SetCustomOptions(std::move(custom));
  EXPECT_EQ(Compile(), kLiteRtStatusErrorInvalidFlatbuffer);
}
TEST_F(DirectDispatchTest, RejectsMalformedDispatchOptions) {
  Load();
  const uint8_t invalid[] = {1, 2, 3};
  model->Subgraph(0).Ops()[0]->SetCustomOptions(invalid, sizeof(invalid));
  EXPECT_EQ(Compile(), kLiteRtStatusErrorInvalidFlatbuffer);
}

TEST_F(DirectDispatchTest, FileLoaderLeavesDispatchValidationToCompiledModel) {
  const auto bytes = MakeModel(false, false, false, false, true);
  char filename[] = "/data/local/tmp/litert-aot-test-XXXXXX";
  const int fd = mkstemp(filename);
  ASSERT_GE(fd, 0);
  const auto written = write(fd, bytes.data(), bytes.size());
  close(fd);
  EXPECT_EQ(written, bytes.size());
  const auto load_status = LiteRtCreateModelFromFile(env, filename, &model);
  unlink(filename);
  ASSERT_EQ(load_status, kLiteRtStatusOk);
  EXPECT_EQ(Compile(), kLiteRtStatusErrorInvalidFlatbuffer);
  EXPECT_EQ(counts.devices, 0);
}
TEST_F(DirectDispatchTest, RejectsMixedAccelerationOptions) {
  Load();
  LiteRtSetOptionsHardwareAccelerators(
      options, kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu);
  EXPECT_EQ(Compile(), kLiteRtStatusErrorUnsupported);
}
TEST_F(DirectDispatchTest, RejectsIncorrectBufferShapeBeforeRegistering) {
  Load();
  ASSERT_EQ(Compile(), kLiteRtStatusOk);
  Buffer(1, 2, 64, 1);
  Buffer();
  Buffer();
  EXPECT_EQ(Run(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(counts.registrations, 0);
}
TEST_F(DirectDispatchTest, RejectsBufferSmallerThanVendorRequirement) {
  Load();
  ASSERT_EQ(Compile(), kLiteRtStatusOk);
  Buffer(1, 2, 8);
  Buffer();
  Buffer();
  EXPECT_EQ(Run(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(counts.registrations, 0);
}
TEST_F(DirectDispatchTest, RejectsMissingBuffersAndInvalidSignature) {
  Load();
  ASSERT_EQ(Compile(), kLiteRtStatusOk);
  Buffer();
  Buffer();
  Buffer();
  EXPECT_EQ(Run(1), kLiteRtStatusErrorIndexOOB);
  EXPECT_EQ(LiteRtRunCompiledModel(compiled, 0, 0, nullptr, 0, nullptr),
            kLiteRtStatusErrorInvalidArgument);
}
TEST_F(DirectDispatchTest, CleansUpFailedContextCreation) {
  Load();
  counts.fail_create = true;
  EXPECT_EQ(Compile(), kLiteRtStatusErrorRuntimeFailure);
  EXPECT_EQ(counts.devices, 0);
}
TEST_F(DirectDispatchTest, RecoversFromRegistrationFailure) {
  Load();
  ASSERT_EQ(Compile(), kLiteRtStatusOk);
  Buffer();
  Buffer();
  Buffer();
  counts.fail_register = true;
  EXPECT_EQ(Run(), kLiteRtStatusErrorRuntimeFailure);
  counts.fail_register = false;
  EXPECT_EQ(Run(), kLiteRtStatusOk);
}
TEST_F(DirectDispatchTest, RecoversFromAttachmentFailure) {
  Load();
  ASSERT_EQ(Compile(), kLiteRtStatusOk);
  Buffer();
  Buffer();
  Buffer();
  counts.fail_attach = true;
  EXPECT_EQ(Run(), kLiteRtStatusErrorRuntimeFailure);
  counts.fail_attach = false;
  EXPECT_EQ(Run(), kLiteRtStatusOk);
  EXPECT_EQ(counts.unregistrations, 1);
}
TEST_F(DirectDispatchTest, ResetsPerRunOptionsAfterFailedInvoke) {
  Load();
  ASSERT_EQ(Compile(), kLiteRtStatusOk);
  Buffer();
  Buffer();
  Buffer();
  counts.fail_invoke = true;
  EXPECT_EQ(LiteRtRunCompiledModelWithOptions(compiled, 0, 2, buffers.data(), 1,
                                              &buffers[2], options),
            kLiteRtStatusErrorRuntimeFailure);
  EXPECT_EQ(counts.options_set, 1);
  EXPECT_EQ(counts.options_reset, 1);
  counts.fail_invoke = false;
  EXPECT_EQ(Run(), kLiteRtStatusOk);
}
TEST_F(DirectDispatchTest, AsyncRequestReportsSynchronousCompletion) {
  Load();
  ASSERT_EQ(Compile(), kLiteRtStatusOk);
  Buffer();
  Buffer();
  Buffer();
  bool async = true;
  EXPECT_EQ(LiteRtRunCompiledModelAsync(compiled, 0, 2, buffers.data(), 1,
                                        &buffers[2], &async),
            kLiteRtStatusOk);
  EXPECT_FALSE(async);
}
TEST_F(DirectDispatchTest, KeepsSignatureContextsSeparate) {
  Load(MakeModel(false, false, false, true));
  ASSERT_EQ(Compile(), kLiteRtStatusOk);
  EXPECT_EQ(counts.devices, 2);
  Buffer();
  Buffer();
  Buffer();
  EXPECT_EQ(Run(0), kLiteRtStatusOk);
  EXPECT_EQ(Run(1), kLiteRtStatusOk);
  EXPECT_EQ(Run(0), kLiteRtStatusOk);
  EXPECT_EQ(counts.registrations, 6);
}
TEST_F(DirectDispatchTest, UnsupportedFeaturesFailExplicitly) {
  Load();
  ASSERT_EQ(Compile(), kLiteRtStatusOk);
  const int dimension = 4;
  EXPECT_EQ(LiteRtCompiledModelResizeInputTensor(compiled, 0, 0, &dimension, 1),
            kLiteRtStatusErrorUnsupported);
  EXPECT_EQ(LiteRtSetCompiledModelCancellationFunction(
                compiled, nullptr, [](void*) { return false; }),
            kLiteRtStatusErrorUnsupported);
  LiteRtProfiler profiler;
  EXPECT_EQ(LiteRtCompiledModelGetProfiler(compiled, &profiler),
            kLiteRtStatusErrorUnsupported);
}
TEST_F(DirectDispatchTest, RejectsOtherBufferBackends) {
  LiteRtRankedTensorType type{};
  type.element_type = kLiteRtElementTypeFloat32;
  type.layout.rank = 1;
  type.layout.dimensions[0] = 2;
  for (auto backend :
       {kLiteRtTensorBufferTypeHostMemory, kLiteRtTensorBufferTypeDmaBuf,
        kLiteRtTensorBufferTypeAhwb, kLiteRtTensorBufferTypeOpenClBuffer}) {
    LiteRtTensorBuffer buffer = nullptr;
    EXPECT_EQ(LiteRtCreateManagedTensorBuffer(env, backend, &type, 64, &buffer),
              kLiteRtStatusErrorUnsupported);
    EXPECT_EQ(buffer, nullptr);
  }
}

TEST_F(DirectDispatchTest,
       ReplacesBuffersAndPreservesRegistrationOnReleaseFailure) {
  Load();
  ASSERT_EQ(Compile(), kLiteRtStatusOk);
  Buffer(1, 2);
  Buffer(10, 20);
  Buffer();
  ASSERT_EQ(Run(), kLiteRtStatusOk);
  auto replacement = Buffer(3, 4);
  std::swap(buffers[0], buffers[3]);
  counts.fail_detach = true;
  EXPECT_EQ(Run(), kLiteRtStatusErrorRuntimeFailure);
  EXPECT_EQ(counts.unregistrations, 0);
  counts.fail_detach = false;
  counts.fail_unregister = true;
  EXPECT_EQ(Run(), kLiteRtStatusErrorRuntimeFailure);
  EXPECT_EQ(counts.unregistrations, 0);
  counts.fail_unregister = false;
  ASSERT_EQ(Run(), kLiteRtStatusOk);
  EXPECT_EQ(counts.registrations, 4);
  EXPECT_EQ(counts.unregistrations, 1);
  EXPECT_EQ(buffers[0], replacement);
}

TEST_F(DirectDispatchTest, RejectsNonzeroBufferOffsets) {
  Load();
  ASSERT_EQ(Compile(), kLiteRtStatusOk);
  Buffer(1, 2, 128, 2, 64);
  Buffer();
  Buffer();
  EXPECT_EQ(Run(), kLiteRtStatusErrorUnsupported);
  EXPECT_EQ(counts.registrations, 0);
}

TEST_F(DirectDispatchTest, RejectsOversizedFastRpcAllocation) {
  LiteRtRankedTensorType type{};
  type.element_type = kLiteRtElementTypeFloat32;
  type.layout.rank = 1;
  type.layout.dimensions[0] = 2;
  LiteRtTensorBuffer buffer = nullptr;
  EXPECT_EQ(LiteRtCreateManagedTensorBuffer(
                env, kLiteRtTensorBufferTypeFastRpc, &type,
                std::numeric_limits<size_t>::max(), &buffer),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(buffer, nullptr);
}

TEST_F(DirectDispatchTest, WaitsForInputEventsAndRejectsOutputEvents) {
  Load();
  ASSERT_EQ(Compile(), kLiteRtStatusOk);
  Buffer(1, 2);
  Buffer(10, 20);
  Buffer();
  // Event::Wait polls fence fds; a signaled eventfd provides a readable test
  // fd.
  const int fd = eventfd(1, EFD_CLOEXEC);
  ASSERT_GE(fd, 0);
  LiteRtEvent event;
  ASSERT_EQ(LiteRtCreateEventFromSyncFenceFd(env, fd, true, &event),
            kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetTensorBufferEvent(buffers[0], event), kLiteRtStatusOk);
  EXPECT_EQ(Run(), kLiteRtStatusOk);
  const int output_fd = eventfd(1, EFD_CLOEXEC);
  ASSERT_GE(output_fd, 0);
  ASSERT_EQ(LiteRtCreateEventFromSyncFenceFd(env, output_fd, true, &event),
            kLiteRtStatusOk);
  ASSERT_EQ(LiteRtSetTensorBufferEvent(buffers[2], event), kLiteRtStatusOk);
  EXPECT_EQ(Run(), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(counts.invokes, 1);
}

TEST_F(DirectDispatchTest, SwapsPreviouslyRegisteredInputBuffers) {
  Load();
  ASSERT_EQ(Compile(), kLiteRtStatusOk);
  Buffer(1, 2);
  Buffer(10, 20);
  Buffer();
  ASSERT_EQ(Run(), kLiteRtStatusOk);
  std::swap(buffers[0], buffers[1]);
  ASSERT_EQ(Run(), kLiteRtStatusOk);
  EXPECT_EQ(counts.registrations, 5);
  void* address;
  ASSERT_EQ(LiteRtLockTensorBuffer(buffers[2], &address,
                                   kLiteRtTensorBufferLockModeRead),
            kLiteRtStatusOk);
  EXPECT_EQ(static_cast<float*>(address)[0], -9);
  EXPECT_EQ(static_cast<float*>(address)[1], -18);
  EXPECT_EQ(LiteRtUnlockTensorBuffer(buffers[2]), kLiteRtStatusOk);
}

TEST_F(DirectDispatchTest, RejectsAliasingPortsBeforeRegistering) {
  Load();
  ASSERT_EQ(Compile(), kLiteRtStatusOk);
  Buffer();
  Buffer();
  LiteRtTensorBuffer inputs[] = {buffers[0], buffers[0]};
  EXPECT_EQ(LiteRtRunCompiledModel(compiled, 0, 2, inputs, 1, &buffers[1]),
            kLiteRtStatusErrorUnsupported);
  EXPECT_EQ(counts.registrations, 0);
}
}  // namespace
