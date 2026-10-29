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

#include "ml_drift_delegate/delegate/delegate_opencl.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gunit.h"
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "ml_drift/cl/buffer.h"  // from @ml_drift
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/cl/tensor.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_object_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "third_party/odml/infra/ml_drift_delegate/testdata/simple_add.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/runtime/tensor_identifier.h"
#include "litert/runtime/tfl_utils.h"
#include "ml_drift_delegate/delegate/cache/simple_cache.h"
#include "ml_drift_delegate/delegate/delegate_kernel_litert.h"
#include "ml_drift_delegate/delegate/delegate_options.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "ml_drift_delegate/delegate/gpu_backend_opencl.h"
#include "ml_drift_delegate/delegate/gpu_backend_opencl_litert.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/version.h"

namespace litert::ml_drift {
namespace {

TEST(DelegateOpenClTest, FdCachingWorksWithoutSerializationDir) {
  // MSAN does not support GPU tests.
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif

  std::string cache_path = testing::TempDir() + "/opencl_prog.cache";
  int fd = open(cache_path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
  ASSERT_GE(fd, 0);

  absl::Cleanup cleanup = [&]() {
    if (fd >= 0) close(fd);
    std::remove(cache_path.c_str());
  };

  const FileToc* fp = simple_add_create();
  auto model = tflite::FlatBufferModel::BuildFromBuffer(fp->data, fp->size);
  ASSERT_TRUE(model);
  std::unique_ptr<tflite::Interpreter> interpreter;
  ASSERT_EQ(
      tflite::InterpreterBuilder(
          *model, tflite::ops::builtin::BuiltinOpResolver())(&interpreter),
      kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);

  LiteRtEnvironment environment = nullptr;
  ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &environment), kLiteRtStatusOk);
  absl::Cleanup env_cleanup = [&]() {
    if (environment) LiteRtDestroyEnvironment(environment);
  };

  auto get_tensor_id = [&interpreter](const TfLiteOpaqueTensor* target_tensor)
      -> litert::internal::TfLiteTensorIdentifier {
    auto tensor_id = litert::internal::GetTensorIdentifier(
        *interpreter, reinterpret_cast<const TfLiteTensor*>(target_tensor));
    if (!tensor_id) {
      return {-1, -1};
    }
    return *tensor_id;
  };
  LiteRtExternalLiteRtBufferContextT buffer_context(environment, get_tensor_id);
  interpreter->SetExternalContext(kTfLiteLiteRtBufferContext, &buffer_context);

  auto options = MlDriftClDelegateDefaultOptionsPtr();
  options->model_token = "test_token";
  options->serialization_dir = nullptr;  // Ensure null
  options->program_cache_fd = fd;
  options->serialize_program_cache = true;
  options->runtime_context = LrtGetRuntimeContext();

  auto delegate = CreateMlDriftClDelegate(std::move(options), environment);
  ASSERT_NE(delegate, nullptr);

  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(std::move(delegate)),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  // Verify original FD is still open
  EXPECT_GE(fcntl(fd, F_GETFD), 0);

  struct stat st;
  ASSERT_EQ(fstat(fd, &st), 0);
  EXPECT_GT(st.st_size, 0);
}

TEST(DelegateOpenClTest, HostIOBufferZeroSizeError) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }
  ::ml_drift::cl::Environment env;
  if (!::ml_drift::cl::CreateEnvironment(&env).ok()) {
    GTEST_SKIP() << "Unable to create OpenCL environment";
  }
  GpuBackendOpenCl backend(&env);
  backend.SetForceHostPtrIOBuffer(true);

  auto zero_res = backend.CreateInternalHostBuffer(
      ::ml_drift::DataType::kFloat32, /*size=*/0, /*input=*/true);
  EXPECT_EQ(zero_res.status().code(), absl::StatusCode::kInvalidArgument);

  auto valid_res = backend.CreateInternalHostBuffer(
      ::ml_drift::DataType::kFloat32, /*size=*/64, /*input=*/true);
  ASSERT_TRUE(valid_res.ok());
  auto buf = *std::move(valid_res);
  EXPECT_TRUE(buf->IsHostPtr());
  EXPECT_NE(buf->GetHostPtr(), nullptr);
  EXPECT_TRUE(buf->OwnsHostPtr());
  EXPECT_EQ(buf->GetSize(), 64);
}

TEST(DelegateOpenClTest, HostIOBufferSyncAndDataTransfer) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }
  ::ml_drift::cl::Environment env;
  if (!::ml_drift::cl::CreateEnvironment(&env).ok()) {
    GTEST_SKIP() << "Unable to create OpenCL environment";
  }
  GpuBackendOpenCl backend(&env);
  backend.SetForceHostPtrIOBuffer(true);

  auto res = backend.CreateInternalHostBuffer(::ml_drift::DataType::kFloat32,
                                              /*size=*/64, /*input=*/true);
  ASSERT_TRUE(res.ok());
  auto buf = *std::move(res);
  EXPECT_TRUE(buf->IsMapped());

  std::vector<uint8_t> write_data(64);
  for (size_t i = 0; i < 64; ++i) {
    write_data[i] = static_cast<uint8_t>(i + 1);
  }
  EXPECT_TRUE(buf->Write(write_data).ok());
  EXPECT_FALSE(buf->IsMapped());

  std::vector<uint8_t> read_data(64, 0);
  EXPECT_TRUE(buf->Read(absl::MakeSpan(read_data)).ok());
  EXPECT_TRUE(buf->IsMapped());
  EXPECT_EQ(write_data, read_data);

  uint8_t* host_bytes = static_cast<uint8_t*>(buf->GetHostPtr());
  ASSERT_NE(host_bytes, nullptr);
  host_bytes[0] = 42;
  host_bytes[1] = 99;
  EXPECT_TRUE(buf->SyncToDevice().ok());
  EXPECT_FALSE(buf->IsMapped());
  EXPECT_TRUE(buf->SyncToHost().ok());
  EXPECT_TRUE(buf->IsMapped());
  EXPECT_TRUE(backend.WaitForCompletion().ok());
  EXPECT_EQ(host_bytes[0], 42);
  EXPECT_EQ(host_bytes[1], 99);
}

TEST(DelegateOpenClTest, WrapExternalHostBufferAndDeviceBuffer) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }
  ::ml_drift::cl::Environment env;
  if (!::ml_drift::cl::CreateEnvironment(&env).ok()) {
    GTEST_SKIP() << "Unable to create OpenCL environment";
  }
  GpuBackendOpenCl backend(&env);
  backend.SetForceHostPtrIOBuffer(true);

  const size_t alignment = backend.GetHostPtrAlignment();
  ASSERT_GT(alignment, 0);
  void* host_ptr = nullptr;
  ASSERT_EQ(posix_memalign(&host_ptr, alignment, 64), 0);
  absl::Cleanup free_ptr = [host_ptr]() { free(host_ptr); };

  auto null_res = backend.WrapExternalHostBuffer(::ml_drift::DataType::kFloat32,
                                                 64, /*input=*/true, nullptr);
  EXPECT_EQ(null_res.status().code(), absl::StatusCode::kInvalidArgument);

  auto zero_wrap = backend.WrapExternalHostBuffer(
      ::ml_drift::DataType::kFloat32, 0, /*input=*/true, host_ptr);
  EXPECT_EQ(zero_wrap.status().code(), absl::StatusCode::kInvalidArgument);

  auto wrap_res = backend.WrapExternalHostBuffer(::ml_drift::DataType::kFloat32,
                                                 64, /*input=*/true, host_ptr);
  ASSERT_TRUE(wrap_res.ok());
  auto wrap_buf = *std::move(wrap_res);
  EXPECT_TRUE(wrap_buf->IsHostPtr());
  EXPECT_FALSE(wrap_buf->OwnsHostPtr());
  EXPECT_EQ(wrap_buf->GetHostPtr(), host_ptr);
  EXPECT_EQ(wrap_buf->GetSize(), 64);
  EXPECT_TRUE(wrap_buf->IsMapped());
  EXPECT_TRUE(wrap_buf->SyncToDevice().ok());
  EXPECT_FALSE(wrap_buf->IsMapped());
  EXPECT_TRUE(wrap_buf->SyncToHost().ok());
  EXPECT_TRUE(wrap_buf->IsMapped());

  auto dev_res = backend.CreateIOBufferWithSize(::ml_drift::DataType::kFloat32,
                                                /*size=*/64, /*input=*/true);
  ASSERT_TRUE(dev_res.ok());
  auto dev_buf = *std::move(dev_res);
  EXPECT_FALSE(dev_buf->IsHostPtr());
  EXPECT_FALSE(dev_buf->IsMapped());
  EXPECT_EQ(dev_buf->GetHostPtr(), nullptr);
  EXPECT_TRUE(dev_buf->SyncToDevice().ok());
  EXPECT_TRUE(dev_buf->SyncToHost().ok());
}

std::vector<int32_t> CollapseToMax4d(absl::Span<const int32_t> shape) {
  if (shape.size() <= 4) {
    return std::vector<int32_t>(shape.begin(), shape.end());
  }
  std::vector<int32_t> collapsed = {shape[0] * shape[1]};
  collapsed.insert(collapsed.end(), shape.begin() + 2, shape.end());
  return collapsed;
}

// Builds a single-subgraph TFLite FlatBuffer testing BATCH_MATMUL. When any
// runtime I/O tensor is 5D, wraps subgraph boundary tensors in 4D<->5D RESHAPE
// ops because OpenCL custom buffer descriptors only accept subgraph I/O up to
// rank 4 while IrModel delegates internal 5D RESHAPE + BATCH_MATMUL in one
// GPU partition.
std::vector<uint8_t> BuildBatchMatMulModelBuffer(
    absl::Span<const int32_t> lhs_shape, absl::Span<const int32_t> rhs_shape,
    absl::Span<const int32_t> out_shape, bool adj_x, bool adj_y,
    const std::vector<float>* const_rhs_data) {
  flatbuffers::FlatBufferBuilder builder;

  const bool need_reshape =
      lhs_shape.size() > 4 ||
      (const_rhs_data == nullptr && rhs_shape.size() > 4) ||
      out_shape.size() > 4;

  std::vector<flatbuffers::Offset<tflite::OperatorCode>> opcodes;
  opcodes.push_back(tflite::CreateOperatorCode(
      builder,
      /*deprecated_builtin_code=*/tflite::BuiltinOperator_BATCH_MATMUL,
      /*custom_code=*/0, /*version=*/1,
      /*builtin_code=*/tflite::BuiltinOperator_BATCH_MATMUL));
  if (need_reshape) {
    opcodes.push_back(tflite::CreateOperatorCode(
        builder,
        /*deprecated_builtin_code=*/tflite::BuiltinOperator_RESHAPE,
        /*custom_code=*/0, /*version=*/1,
        /*builtin_code=*/tflite::BuiltinOperator_RESHAPE));
  }
  auto opcodes_vec = builder.CreateVector(opcodes);

  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
  buffers.push_back(tflite::CreateBuffer(builder));
  uint32_t rhs_buffer_idx = 0;
  if (const_rhs_data != nullptr) {
    const auto* raw_bytes =
        reinterpret_cast<const uint8_t*>(const_rhs_data->data());
    auto data_vec =
        builder.CreateVector(raw_bytes, const_rhs_data->size() * sizeof(float));
    buffers.push_back(tflite::CreateBuffer(builder, data_vec));
    rhs_buffer_idx = 1;
  }

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  std::vector<flatbuffers::Offset<tflite::Operator>> ops;
  std::vector<int32_t> sg_inputs;

  auto add_tensor = [&](absl::Span<const int32_t> shape, uint32_t buf_idx,
                        const char* name) -> int32_t {
    const int32_t idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(tflite::CreateTensor(
        builder, builder.CreateVector(shape.data(), shape.size()),
        tflite::TensorType_FLOAT32, buf_idx, builder.CreateString(name)));
    return idx;
  };
  auto add_reshape = [&](int32_t src_idx, int32_t dst_idx,
                         absl::Span<const int32_t> target_shape) {
    auto opts = tflite::CreateReshapeOptions(
                    builder, builder.CreateVector(target_shape.data(),
                                                  target_shape.size()))
                    .Union();
    ops.push_back(tflite::CreateOperator(
        builder, /*opcode_index=*/1,
        builder.CreateVector(std::vector<int32_t>{src_idx}),
        builder.CreateVector(std::vector<int32_t>{dst_idx}),
        tflite::BuiltinOptions_ReshapeOptions, opts));
  };

  int32_t bmm_lhs_idx = -1;
  if (lhs_shape.size() > 4) {
    const int32_t sg_lhs_idx =
        add_tensor(CollapseToMax4d(lhs_shape), /*buf_idx=*/0, "lhs_in");
    sg_inputs.push_back(sg_lhs_idx);
    bmm_lhs_idx = add_tensor(lhs_shape, /*buf_idx=*/0, "lhs");
    add_reshape(sg_lhs_idx, bmm_lhs_idx, lhs_shape);
  } else {
    bmm_lhs_idx = add_tensor(lhs_shape, /*buf_idx=*/0, "lhs");
    sg_inputs.push_back(bmm_lhs_idx);
  }

  int32_t bmm_rhs_idx = -1;
  if (const_rhs_data != nullptr) {
    bmm_rhs_idx = add_tensor(rhs_shape, rhs_buffer_idx, "rhs");
  } else if (rhs_shape.size() > 4) {
    const int32_t sg_rhs_idx =
        add_tensor(CollapseToMax4d(rhs_shape), /*buf_idx=*/0, "rhs_in");
    sg_inputs.push_back(sg_rhs_idx);
    bmm_rhs_idx = add_tensor(rhs_shape, /*buf_idx=*/0, "rhs");
    add_reshape(sg_rhs_idx, bmm_rhs_idx, rhs_shape);
  } else {
    bmm_rhs_idx = add_tensor(rhs_shape, /*buf_idx=*/0, "rhs");
    sg_inputs.push_back(bmm_rhs_idx);
  }

  const int32_t bmm_out_idx = add_tensor(out_shape, /*buf_idx=*/0, "out");
  auto bmm_options =
      tflite::CreateBatchMatMulOptions(builder, adj_x, adj_y).Union();
  ops.push_back(tflite::CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector(std::vector<int32_t>{bmm_lhs_idx, bmm_rhs_idx}),
      builder.CreateVector(std::vector<int32_t>{bmm_out_idx}),
      tflite::BuiltinOptions_BatchMatMulOptions, bmm_options));

  int32_t sg_out_idx = bmm_out_idx;
  if (out_shape.size() > 4) {
    const std::vector<int32_t> sg_out_shape = CollapseToMax4d(out_shape);
    sg_out_idx = add_tensor(sg_out_shape, /*buf_idx=*/0, "out_4d");
    add_reshape(bmm_out_idx, sg_out_idx, sg_out_shape);
  }

  auto tensors_vec = builder.CreateVector(tensors);
  auto operators_vec = builder.CreateVector(ops);
  auto subgraph = tflite::CreateSubGraph(
      builder, tensors_vec, builder.CreateVector(sg_inputs),
      builder.CreateVector(std::vector<int32_t>{sg_out_idx}), operators_vec,
      builder.CreateString("main"));
  auto subgraphs_vec = builder.CreateVector(
      std::vector<flatbuffers::Offset<tflite::SubGraph>>{subgraph});

  auto model = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, opcodes_vec, subgraphs_vec,
      builder.CreateString("BatchMatMul test"), builder.CreateVector(buffers));
  builder.Finish(model, tflite::ModelIdentifier());
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

size_t NumElements(absl::Span<const int32_t> shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t{1},
                         std::multiplies<size_t>());
}

std::vector<float> MakeDeterministicValues(size_t count, float offset) {
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    values[i] = offset + 0.1f * static_cast<float>(
                                    (static_cast<int>(i * 7 + 3) % 19) - 9);
  }
  return values;
}

void VerifyBatchMatMulMatchesCpuOnOpenCl(
    const litert::Environment& env, absl::Span<const int32_t> base_lhs_shape,
    absl::Span<const int32_t> base_rhs_shape,
    absl::Span<const int32_t> out_shape, bool adj_x, bool adj_y, bool const_rhs,
    bool use_ir_model) {
  std::vector<int32_t> lhs_shape(base_lhs_shape.begin(), base_lhs_shape.end());
  if (adj_x) {
    std::swap(lhs_shape[lhs_shape.size() - 2], lhs_shape[lhs_shape.size() - 1]);
  }
  std::vector<int32_t> rhs_shape(base_rhs_shape.begin(), base_rhs_shape.end());
  if (adj_y) {
    std::swap(rhs_shape[rhs_shape.size() - 2], rhs_shape[rhs_shape.size() - 1]);
  }

  const std::vector<float> lhs_data =
      MakeDeterministicValues(NumElements(lhs_shape), /*offset=*/0.25f);
  const std::vector<float> rhs_data =
      MakeDeterministicValues(NumElements(rhs_shape), /*offset=*/-0.15f);
  const size_t out_count = NumElements(out_shape);

  const std::string case_label = absl::StrCat(
      "lhs=[", absl::StrJoin(lhs_shape, ","), "] rhs=[",
      absl::StrJoin(rhs_shape, ","), "] adj_x=", adj_x, " adj_y=", adj_y,
      " const_rhs=", const_rhs, " use_ir_model=", use_ir_model);
  SCOPED_TRACE(case_label);

  const std::vector<uint8_t> model_buf =
      BuildBatchMatMulModelBuffer(lhs_shape, rhs_shape, out_shape, adj_x, adj_y,
                                  const_rhs ? &rhs_data : nullptr);
  auto model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(model_buf.data()), model_buf.size());
  ASSERT_NE(model, nullptr);

  // 1. CPU reference execution.
  std::unique_ptr<tflite::Interpreter> cpu_interp;
  ASSERT_EQ(tflite::InterpreterBuilder(
                *model, tflite::ops::builtin::BuiltinOpResolver())(&cpu_interp),
            kTfLiteOk);
  ASSERT_NE(cpu_interp, nullptr);
  ASSERT_EQ(cpu_interp->AllocateTensors(), kTfLiteOk);
  std::memcpy(cpu_interp->typed_input_tensor<float>(0), lhs_data.data(),
              lhs_data.size() * sizeof(float));
  if (!const_rhs) {
    std::memcpy(cpu_interp->typed_input_tensor<float>(1), rhs_data.data(),
                rhs_data.size() * sizeof(float));
  }
  ASSERT_EQ(cpu_interp->Invoke(), kTfLiteOk);
  const float* cpu_out_ptr = cpu_interp->typed_output_tensor<float>(0);
  ASSERT_NE(cpu_out_ptr, nullptr);
  const std::vector<float> cpu_out(cpu_out_ptr, cpu_out_ptr + out_count);

  // 2. OpenCL delegated execution via CompiledModel.
  auto litert_options = litert::Options::Create();
  ASSERT_TRUE(litert_options);
  ASSERT_TRUE(
      litert_options->SetHardwareAccelerators(litert::HwAccelerators::kGpu));
  auto gpu_options = litert_options->GetOptions<litert::GpuOptions>();
  ASSERT_TRUE(gpu_options);
  ASSERT_TRUE(gpu_options->SetBackend(litert::GpuOptions::Backend::kOpenCl));
  ASSERT_TRUE(gpu_options->SetPrecision(litert::GpuOptions::Precision::kFp32));
  ASSERT_EQ(gpu_options->SetSerializeProgramCache(false), kLiteRtStatusOk);
  ASSERT_EQ(gpu_options->SetUseIrModel(use_ir_model), kLiteRtStatusOk);

  auto compiled_model = litert::CompiledModel::Create(
      env, litert::BufferRef<uint8_t>(model_buf.data(), model_buf.size()),
      *litert_options);
  ASSERT_TRUE(compiled_model) << compiled_model.Error().Message();
  auto fully_accelerated = compiled_model->IsFullyAccelerated();
  ASSERT_TRUE(fully_accelerated);
  EXPECT_TRUE(*fully_accelerated);

  auto input_buffers = compiled_model->CreateInputBuffers();
  ASSERT_TRUE(input_buffers);
  ASSERT_EQ(input_buffers->size(), const_rhs ? 1 : 2);
  ASSERT_TRUE((*input_buffers)[0].Write<float>(absl::MakeConstSpan(lhs_data)));
  if (!const_rhs) {
    ASSERT_TRUE(
        (*input_buffers)[1].Write<float>(absl::MakeConstSpan(rhs_data)));
  }
  auto output_buffers = compiled_model->CreateOutputBuffers();
  ASSERT_TRUE(output_buffers);
  ASSERT_EQ(output_buffers->size(), 1);

  ASSERT_TRUE(compiled_model->Run(*input_buffers, *output_buffers));

  std::vector<float> gpu_out(out_count);
  ASSERT_TRUE((*output_buffers)[0].Read<float>(absl::MakeSpan(gpu_out)));

  for (size_t i = 0; i < out_count; ++i) {
    EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-4f) << "at index " << i;
  }
}

TEST(DelegateOpenClTest, BatchMatMulAdjXAndAdjYRuntimeInputsMatchesCpu) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!IsOpenClSupported()) {
    GTEST_SKIP() << "OpenCL is not supported on this platform";
  }

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  struct ShapeCase {
    std::vector<int32_t> lhs;
    std::vector<int32_t> rhs;
    std::vector<int32_t> out;
    bool ir_only = false;
  };
  const std::vector<ShapeCase> cases = {
      {{4, 4}, {4, 4}, {4, 4}},
      {{3, 5}, {5, 4}, {3, 4}},
      {{2, 1, 5}, {2, 5, 4}, {2, 1, 4}},
      {{2, 3, 5}, {2, 5, 4}, {2, 3, 4}},
      {{2, 20, 9}, {2, 9, 7}, {2, 20, 7}},
      {{2, 2, 3, 5}, {2, 2, 5, 4}, {2, 2, 3, 4}},
      {{1, 2, 2, 1, 5}, {1, 2, 2, 5, 4}, {1, 2, 2, 1, 4}, /*ir_only=*/true},
      {{2, 1, 2, 3, 5}, {2, 1, 2, 5, 4}, {2, 1, 2, 3, 4}, /*ir_only=*/true},
  };

  for (bool use_ir_model : {false, true}) {
    for (const auto& tc : cases) {
      if (tc.ir_only && !use_ir_model) continue;
      for (bool adj_x : {false, true}) {
        for (bool adj_y : {false, true}) {
          VerifyBatchMatMulMatchesCpuOnOpenCl(*env, tc.lhs, tc.rhs, tc.out,
                                              adj_x, adj_y, /*const_rhs=*/false,
                                              use_ir_model);
        }
      }
    }
  }
}

TEST(DelegateOpenClTest, BatchMatMulAdjXAndAdjYConst2dRhsMatchesCpu) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!IsOpenClSupported()) {
    GTEST_SKIP() << "OpenCL is not supported on this platform";
  }

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  struct ShapeCase {
    std::vector<int32_t> lhs;
    std::vector<int32_t> rhs;
    std::vector<int32_t> out;
    bool ir_only = false;
  };
  const std::vector<ShapeCase> cases = {
      // Exact 4x4 square constant-RHS case from GitHub issue #8913.
      {{4, 4}, {4, 4}, {4, 4}},
      {{3, 5}, {5, 4}, {3, 4}},
      {{2, 3, 5}, {5, 4}, {2, 3, 4}},
      {{2, 2, 3, 5}, {5, 4}, {2, 2, 3, 4}},
      {{1, 2, 2, 1, 5}, {5, 4}, {1, 2, 2, 1, 4}, /*ir_only=*/true},
      {{2, 1, 2, 3, 5}, {5, 4}, {2, 1, 2, 3, 4}, /*ir_only=*/true},
  };

  for (bool use_ir_model : {false, true}) {
    for (const auto& tc : cases) {
      if (tc.ir_only && !use_ir_model) continue;
      for (bool adj_x : {false, true}) {
        for (bool adj_y : {false, true}) {
          VerifyBatchMatMulMatchesCpuOnOpenCl(*env, tc.lhs, tc.rhs, tc.out,
                                              adj_x, adj_y, /*const_rhs=*/true,
                                              use_ir_model);
        }
      }
    }
  }
}

TEST(DelegateOpenClTest, HostMemoryZeroCopyWrappingAndCaching) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }
  ::ml_drift::cl::Environment env;
  if (!::ml_drift::cl::CreateEnvironment(&env).ok()) {
    GTEST_SKIP() << "Unable to create OpenCL environment";
  }

  GpuBackendOpenClLitert backend(&env, /*egl_display=*/nullptr, SimpleCache(),
                                 LrtGetRuntimeContext());

  // 1. When zero-copy host ptr is forced, verify buffer requirements advertise
  // HostMemory only for non-external tensors (external tensors bind directly
  // via CreateTensorShared without converters).
  backend.SetForceHostPtrIOBuffer(true);
  auto req = backend.GetGpuBufferRequirements(
      ::ml_drift::TensorStorageType::kBuffer, ::ml_drift::DataType::kFloat32);
  ASSERT_TRUE(req.ok());
  bool has_host_mem = false;
  for (auto type : req->buffer_types) {
    if (type == kLiteRtTensorBufferTypeHostMemory) {
      has_host_mem = true;
      break;
    }
  }
  EXPECT_FALSE(has_host_mem);
  EXPECT_EQ(req->alignment, 0);

  auto non_ext_req = backend.GetGpuBufferRequirementsForNonExternalTensors();
  ASSERT_TRUE(non_ext_req.ok());
  bool non_ext_has_host_mem = false;
  for (auto type : non_ext_req->buffer_types) {
    if (type == kLiteRtTensorBufferTypeHostMemory) {
      non_ext_has_host_mem = true;
      break;
    }
  }
  EXPECT_TRUE(non_ext_has_host_mem);
  EXPECT_EQ(non_ext_req->alignment, backend.GetHostPtrAlignment());

  // 2. Test wrapping an external host pointer and caching behavior.
  const size_t alignment =
      std::max(backend.GetHostPtrAlignment(),
               static_cast<size_t>(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT));
  constexpr size_t kBufferSize = 64;
  void* host_ptr = nullptr;
  ASSERT_EQ(posix_memalign(&host_ptr, alignment, kBufferSize), 0);
  absl::Cleanup free_ptr = [host_ptr]() { free(host_ptr); };

  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 1;
  tensor_type.layout.dimensions[0] = 16;
  tensor_type.element_type = kLiteRtElementTypeFloat32;

  LiteRtTensorBuffer tensor_buffer = nullptr;
  ASSERT_EQ(LiteRtCreateTensorBufferFromHostMemory(
                &tensor_type, host_ptr, kBufferSize, /*deallocator=*/nullptr,
                &tensor_buffer),
            kLiteRtStatusOk);
  GpuTensorBufferPtr tb_ptr(tensor_buffer,
                            GpuTensorBufferDeleter{LiteRtDestroyTensorBuffer});

  EXPECT_EQ(backend.num_wrapped_host_buffers(), 0);

  auto handle_or = backend.GetGpuMemoryAllocated(tb_ptr);
  ASSERT_TRUE(handle_or.ok());
  EXPECT_NE(*handle_or, nullptr);
  EXPECT_EQ(backend.num_wrapped_host_buffers(), 1);

  // Second call with the same buffer returns the cached handle.
  auto cached_handle_or = backend.GetGpuMemoryAllocated(tb_ptr);
  ASSERT_TRUE(cached_handle_or.ok());
  EXPECT_EQ(*handle_or, *cached_handle_or);
  EXPECT_EQ(backend.num_wrapped_host_buffers(), 1);

  // 3. Test that ClearWrappedHostBuffers drops transient wrapped buffers and
  // allows subsequent re-wrapping.
  backend.ClearWrappedHostBuffers();
  EXPECT_EQ(backend.num_wrapped_host_buffers(), 0);
  EXPECT_EQ(backend.GetWrappedHostBufferForTesting(tb_ptr.get()), nullptr);

  auto rewrapped_handle_or = backend.GetGpuMemoryAllocated(tb_ptr);
  ASSERT_TRUE(rewrapped_handle_or.ok());
  EXPECT_NE(*rewrapped_handle_or, nullptr);
  EXPECT_EQ(backend.num_wrapped_host_buffers(), 1);

  // 4. Test error handling when zero-copy host pointer is not supported.
  backend.SetForceHostPtrIOBuffer(false);
  if (!backend.SupportsZeroCopyHostPtr(/*input=*/true) ||
      !backend.SupportsZeroCopyHostPtr(/*input=*/false)) {
    auto unsupported_or = backend.GetGpuMemoryAllocated(tb_ptr);
    EXPECT_EQ(unsupported_or.status().code(),
              absl::StatusCode::kFailedPrecondition);
  }
}

TEST(DelegateOpenClTest, HostMemoryUnalignedHostPointerFailsFast) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }
  ::ml_drift::cl::Environment env;
  if (!::ml_drift::cl::CreateEnvironment(&env).ok()) {
    GTEST_SKIP() << "Unable to create OpenCL environment";
  }
  GpuBackendOpenClLitert backend(&env, /*egl_display=*/nullptr, SimpleCache(),
                                 LrtGetRuntimeContext());
  backend.SetForceHostPtrIOBuffer(true);

  const size_t page_alignment = backend.GetHostPtrAlignment();
  ASSERT_GE(page_alignment, 4096);
  constexpr size_t kBufferSize = 64;

  // Allocate a page-aligned buffer with extra padding so we can create a
  // pointer aligned to 64 bytes but NOT page-aligned (offset by 64 bytes).
  void* raw_alloc = nullptr;
  ASSERT_EQ(posix_memalign(&raw_alloc, page_alignment, kBufferSize + 64), 0);
  absl::Cleanup free_raw = [raw_alloc]() { free(raw_alloc); };

  uint8_t* unaligned_host_ptr = static_cast<uint8_t*>(raw_alloc) + 64;
  ASSERT_EQ(reinterpret_cast<uintptr_t>(unaligned_host_ptr) % 64, 0);
  ASSERT_NE(reinterpret_cast<uintptr_t>(unaligned_host_ptr) % page_alignment,
            0);

  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 1;
  tensor_type.layout.dimensions[0] = 16;
  tensor_type.element_type = kLiteRtElementTypeFloat32;

  LiteRtTensorBuffer tensor_buffer = nullptr;
  ASSERT_EQ(LiteRtCreateTensorBufferFromHostMemory(
                &tensor_type, unaligned_host_ptr, kBufferSize,
                /*deallocator=*/nullptr, &tensor_buffer),
            kLiteRtStatusOk);
  GpuTensorBufferPtr tb_ptr(tensor_buffer,
                            GpuTensorBufferDeleter{LiteRtDestroyTensorBuffer});

  // Unaligned host buffer should fail fast with InvalidArgumentError.
  auto handle_or = backend.GetGpuMemoryAllocated(tb_ptr);
  EXPECT_FALSE(handle_or.ok());
  EXPECT_EQ(handle_or.status().code(), absl::StatusCode::kInvalidArgument);
}

class MockAsymmetricGpuBackendOpenClLitert : public GpuBackendOpenClLitert {
 public:
  using GpuBackendOpenClLitert::GpuBackendOpenClLitert;
  bool SupportsZeroCopyHostPtr(bool input) const override {
    // Return true for output (input=false), but false for input (input=true).
    return !input;
  }
};

TEST(DelegateOpenClTest, HostMemoryRequiresBothInputAndOutputZeroCopySupport) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }
  ::ml_drift::cl::Environment env;
  if (!::ml_drift::cl::CreateEnvironment(&env).ok()) {
    GTEST_SKIP() << "Unable to create OpenCL environment";
  }
  MockAsymmetricGpuBackendOpenClLitert backend(
      &env, /*egl_display=*/nullptr, SimpleCache(), LrtGetRuntimeContext());
  backend.SetForceHostPtrIOBuffer(true);

  const size_t alignment =
      std::max(backend.GetHostPtrAlignment(),
               static_cast<size_t>(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT));
  constexpr size_t kBufferSize = 64;
  void* host_ptr = nullptr;
  ASSERT_EQ(posix_memalign(&host_ptr, alignment, kBufferSize), 0);
  absl::Cleanup free_ptr = [host_ptr]() { free(host_ptr); };

  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 1;
  tensor_type.layout.dimensions[0] = 16;
  tensor_type.element_type = kLiteRtElementTypeFloat32;

  LiteRtTensorBuffer tensor_buffer = nullptr;
  ASSERT_EQ(LiteRtCreateTensorBufferFromHostMemory(
                &tensor_type, host_ptr, kBufferSize, /*deallocator=*/nullptr,
                &tensor_buffer),
            kLiteRtStatusOk);
  GpuTensorBufferPtr tb_ptr(tensor_buffer,
                            GpuTensorBufferDeleter{LiteRtDestroyTensorBuffer});

  auto handle_or = backend.GetGpuMemoryAllocated(tb_ptr);
  EXPECT_EQ(handle_or.status().code(), absl::StatusCode::kFailedPrecondition);
}

namespace {
void* g_override_host_addr = nullptr;

LiteRtStatus MockGetTensorBufferHostMemory(LiteRtTensorBuffer buffer,
                                           void** host_memory) {
  if (g_override_host_addr != nullptr) {
    *host_memory = g_override_host_addr;
    return kLiteRtStatusOk;
  }
  return LrtGetRuntimeContext()->get_tensor_buffer_host_memory(buffer,
                                                               host_memory);
}
}  // namespace

TEST(DelegateOpenClTest, HostMemoryWrappedCacheInvalidatesOnHostAddressChange) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }
  ::ml_drift::cl::Environment env;
  if (!::ml_drift::cl::CreateEnvironment(&env).ok()) {
    GTEST_SKIP() << "Unable to create OpenCL environment";
  }

  LiteRtRuntimeContext mock_runtime_context = *LrtGetRuntimeContext();
  mock_runtime_context.get_tensor_buffer_host_memory =
      &MockGetTensorBufferHostMemory;

  GpuBackendOpenClLitert backend(&env, /*egl_display=*/nullptr, SimpleCache(),
                                 &mock_runtime_context);
  backend.SetForceHostPtrIOBuffer(true);

  const size_t alignment =
      std::max(backend.GetHostPtrAlignment(),
               static_cast<size_t>(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT));
  constexpr size_t kBufferSize = 64;

  void* host_ptr1 = nullptr;
  ASSERT_EQ(posix_memalign(&host_ptr1, alignment, kBufferSize), 0);
  absl::Cleanup free_ptr1 = [host_ptr1]() { free(host_ptr1); };

  void* host_ptr2 = nullptr;
  ASSERT_EQ(posix_memalign(&host_ptr2, alignment, kBufferSize), 0);
  absl::Cleanup free_ptr2 = [host_ptr2]() { free(host_ptr2); };

  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 1;
  tensor_type.layout.dimensions[0] = 16;
  tensor_type.element_type = kLiteRtElementTypeFloat32;

  LiteRtTensorBuffer tensor_buffer = nullptr;
  ASSERT_EQ(LiteRtCreateTensorBufferFromHostMemory(
                &tensor_type, host_ptr1, kBufferSize, /*deallocator=*/nullptr,
                &tensor_buffer),
            kLiteRtStatusOk);
  GpuTensorBufferPtr tb_ptr(tensor_buffer,
                            GpuTensorBufferDeleter{LiteRtDestroyTensorBuffer});

  // Call 1 with host_ptr1.
  g_override_host_addr = host_ptr1;
  auto handle1_or = backend.GetGpuMemoryAllocated(tb_ptr);
  ASSERT_TRUE(handle1_or.ok());
  auto* wrapped1 = backend.GetWrappedHostBufferForTesting(tb_ptr.get());
  ASSERT_NE(wrapped1, nullptr);
  EXPECT_EQ(wrapped1->GetHostPtr(), host_ptr1);

  // Call 2 with host_ptr2 for the same TensorBuffer and size.
  g_override_host_addr = host_ptr2;
  auto handle2_or = backend.GetGpuMemoryAllocated(tb_ptr);
  ASSERT_TRUE(handle2_or.ok());
  auto* wrapped2 = backend.GetWrappedHostBufferForTesting(tb_ptr.get());
  ASSERT_NE(wrapped2, nullptr);
  EXPECT_EQ(wrapped2->GetHostPtr(), host_ptr2);
  g_override_host_addr = nullptr;
}

TEST(DelegateOpenClTest, IsHostMemoryTensorBufferIdentifiesBufferType) {
  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 1;
  tensor_type.layout.dimensions[0] = 16;
  tensor_type.element_type = kLiteRtElementTypeFloat32;

  alignas(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT) float data[16];
  for (float& val : data) val = 1.0f;
  LiteRtTensorBuffer host_tb = nullptr;
  ASSERT_EQ(
      LiteRtCreateTensorBufferFromHostMemory(&tensor_type, data, sizeof(data),
                                             /*deallocator=*/nullptr, &host_tb),
      kLiteRtStatusOk);
  GpuTensorBufferPtr host_tb_ptr(
      host_tb, GpuTensorBufferDeleter{LiteRtDestroyTensorBuffer});

  DelegateKernelLiteRt kernel(LrtGetRuntimeContext());
  EXPECT_TRUE(kernel.IsHostMemoryTensorBuffer(host_tb_ptr));
}

TEST(DelegateOpenClTest, DelegateKernelTracksInputAndOutputHostMemoryBuffers) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }

  // --- Subtest A: Input HostMemory buffer tracking (kills Mutant #4) ---
  {
    const FileToc* fp = simple_add_create();
    auto model = tflite::FlatBufferModel::BuildFromBuffer(fp->data, fp->size);
    ASSERT_TRUE(model);
    std::unique_ptr<tflite::Interpreter> interpreter;
    ASSERT_EQ(
        tflite::InterpreterBuilder(
            *model, tflite::ops::builtin::BuiltinOpResolver())(&interpreter),
        kTfLiteOk);
    ASSERT_NE(interpreter, nullptr);

    LiteRtEnvironment environment = nullptr;
    ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &environment),
              kLiteRtStatusOk);
    absl::Cleanup env_cleanup = [&]() {
      if (environment) LiteRtDestroyEnvironment(environment);
    };

    auto get_tensor_id = [&interpreter](const TfLiteOpaqueTensor* target_tensor)
        -> litert::internal::TfLiteTensorIdentifier {
      auto tensor_id = litert::internal::GetTensorIdentifier(
          *interpreter, reinterpret_cast<const TfLiteTensor*>(target_tensor));
      if (!tensor_id) return {-1, -1};
      return *tensor_id;
    };
    LiteRtExternalLiteRtBufferContextT buffer_context(environment,
                                                      get_tensor_id);
    interpreter->SetExternalContext(kTfLiteLiteRtBufferContext,
                                    &buffer_context);

    auto options = MlDriftClDelegateDefaultOptionsPtr();
    options->wait_type = kGpuDelegateWaitTypeDoNotWait;
    options->force_host_ptr_io_buffers = true;
    options->runtime_context = LrtGetRuntimeContext();

    auto delegate = CreateMlDriftClDelegate(std::move(options), environment);
    ASSERT_NE(delegate, nullptr);
    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(std::move(delegate)),
              kTfLiteOk);
    ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    // Register HostMemory on INPUT 0.
    const TfLiteTensor* in_tensor =
        interpreter->tensor(interpreter->inputs()[0]);
    LiteRtRankedTensorType tensor_type;
    tensor_type.layout.rank = in_tensor->dims->size;
    for (int d = 0; d < in_tensor->dims->size; ++d) {
      tensor_type.layout.dimensions[d] = in_tensor->dims->data[d];
    }
    tensor_type.element_type = kLiteRtElementTypeFloat32;
    auto in_req_or = buffer_context.GetBufferRequirements(in_tensor);
    ASSERT_TRUE(in_req_or.HasValue());
    size_t in_req_alignment = 0;
    ASSERT_EQ(LiteRtGetTensorBufferRequirementsAlignment(
                  const_cast<LiteRtTensorBufferRequirements>(*in_req_or),
                  &in_req_alignment),
              kLiteRtStatusOk);
    EXPECT_GE(in_req_alignment, 4096);

    void* in_host_ptr = nullptr;
    ASSERT_EQ(posix_memalign(&in_host_ptr, in_req_alignment, in_tensor->bytes),
              0);
    absl::Cleanup free_in = [in_host_ptr]() { free(in_host_ptr); };
    std::memset(in_host_ptr, 0, in_tensor->bytes);

    LiteRtTensorBuffer host_tb = nullptr;
    ASSERT_EQ(LiteRtCreateTensorBufferFromHostMemory(
                  &tensor_type, in_host_ptr, in_tensor->bytes,
                  /*deallocator=*/nullptr, &host_tb),
              kLiteRtStatusOk);
    ASSERT_EQ(buffer_context.RegisterTensorBuffer(
                  in_tensor,
                  LiteRtTensorBufferPtr(host_tb, LiteRtTensorBufferDeleter{})),
              kLiteRtStatusOk);

    ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

    ASSERT_FALSE(interpreter->execution_plan().empty());
    int node_idx = interpreter->execution_plan()[0];
    auto* node_reg = interpreter->node_and_registration(node_idx);
    ASSERT_NE(node_reg, nullptr);
    auto* kernel =
        reinterpret_cast<DelegateKernelLiteRt*>(node_reg->first.user_data);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->has_host_memory_buffers());
  }

  // --- Subtest B: Output HostMemory buffer tracking (kills Mutant #5) ---
  {
    const FileToc* fp = simple_add_create();
    auto model = tflite::FlatBufferModel::BuildFromBuffer(fp->data, fp->size);
    ASSERT_TRUE(model);
    std::unique_ptr<tflite::Interpreter> interpreter;
    ASSERT_EQ(
        tflite::InterpreterBuilder(
            *model, tflite::ops::builtin::BuiltinOpResolver())(&interpreter),
        kTfLiteOk);
    ASSERT_NE(interpreter, nullptr);

    LiteRtEnvironment environment = nullptr;
    ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &environment),
              kLiteRtStatusOk);
    absl::Cleanup env_cleanup = [&]() {
      if (environment) LiteRtDestroyEnvironment(environment);
    };

    auto get_tensor_id = [&interpreter](const TfLiteOpaqueTensor* target_tensor)
        -> litert::internal::TfLiteTensorIdentifier {
      auto tensor_id = litert::internal::GetTensorIdentifier(
          *interpreter, reinterpret_cast<const TfLiteTensor*>(target_tensor));
      if (!tensor_id) return {-1, -1};
      return *tensor_id;
    };
    LiteRtExternalLiteRtBufferContextT buffer_context(environment,
                                                      get_tensor_id);
    interpreter->SetExternalContext(kTfLiteLiteRtBufferContext,
                                    &buffer_context);

    auto options = MlDriftClDelegateDefaultOptionsPtr();
    options->wait_type = kGpuDelegateWaitTypeDoNotWait;
    options->force_host_ptr_io_buffers = true;
    options->runtime_context = LrtGetRuntimeContext();

    auto delegate = CreateMlDriftClDelegate(std::move(options), environment);
    ASSERT_NE(delegate, nullptr);
    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(std::move(delegate)),
              kTfLiteOk);
    ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    // Register HostMemory on OUTPUT 0 (none on inputs).
    const TfLiteTensor* out_tensor =
        interpreter->tensor(interpreter->outputs()[0]);
    LiteRtRankedTensorType tensor_type;
    tensor_type.layout.rank = out_tensor->dims->size;
    for (int d = 0; d < out_tensor->dims->size; ++d) {
      tensor_type.layout.dimensions[d] = out_tensor->dims->data[d];
    }
    tensor_type.element_type = kLiteRtElementTypeFloat32;
    auto out_req_or = buffer_context.GetBufferRequirements(out_tensor);
    ASSERT_TRUE(out_req_or.HasValue());
    size_t out_req_alignment = 0;
    ASSERT_EQ(LiteRtGetTensorBufferRequirementsAlignment(
                  const_cast<LiteRtTensorBufferRequirements>(*out_req_or),
                  &out_req_alignment),
              kLiteRtStatusOk);
    EXPECT_GE(out_req_alignment, 4096);

    void* out_host_ptr = nullptr;
    ASSERT_EQ(
        posix_memalign(&out_host_ptr, out_req_alignment, out_tensor->bytes), 0);
    absl::Cleanup free_out = [out_host_ptr]() { free(out_host_ptr); };
    std::memset(out_host_ptr, 0, out_tensor->bytes);

    LiteRtTensorBuffer host_tb = nullptr;
    ASSERT_EQ(LiteRtCreateTensorBufferFromHostMemory(
                  &tensor_type, out_host_ptr, out_tensor->bytes,
                  /*deallocator=*/nullptr, &host_tb),
              kLiteRtStatusOk);
    ASSERT_EQ(buffer_context.RegisterTensorBuffer(
                  out_tensor,
                  LiteRtTensorBufferPtr(host_tb, LiteRtTensorBufferDeleter{})),
              kLiteRtStatusOk);

    ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

    ASSERT_FALSE(interpreter->execution_plan().empty());
    int node_idx = interpreter->execution_plan()[0];
    auto* node_reg = interpreter->node_and_registration(node_idx);
    ASSERT_NE(node_reg, nullptr);
    auto* kernel =
        reinterpret_cast<DelegateKernelLiteRt*>(node_reg->first.user_data);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->has_host_memory_buffers());
  }

  // --- Subtest C: Async execution mode with HostMemory buffers waits for
  // completion ---
  {
    const FileToc* fp = simple_add_create();
    auto model = tflite::FlatBufferModel::BuildFromBuffer(fp->data, fp->size);
    ASSERT_TRUE(model);
    std::unique_ptr<tflite::Interpreter> interpreter;
    ASSERT_EQ(
        tflite::InterpreterBuilder(
            *model, tflite::ops::builtin::BuiltinOpResolver())(&interpreter),
        kTfLiteOk);
    ASSERT_NE(interpreter, nullptr);

    LiteRtEnvironment environment = nullptr;
    ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &environment),
              kLiteRtStatusOk);
    absl::Cleanup env_cleanup = [&]() {
      if (environment) LiteRtDestroyEnvironment(environment);
    };

    auto get_tensor_id = [&interpreter](const TfLiteOpaqueTensor* target_tensor)
        -> litert::internal::TfLiteTensorIdentifier {
      auto tensor_id = litert::internal::GetTensorIdentifier(
          *interpreter, reinterpret_cast<const TfLiteTensor*>(target_tensor));
      if (!tensor_id) return {-1, -1};
      return *tensor_id;
    };
    LiteRtExternalLiteRtBufferContextT buffer_context(environment,
                                                      get_tensor_id);
    buffer_context.SetAsyncExecutionMode(true);
    interpreter->SetExternalContext(kTfLiteLiteRtBufferContext,
                                    &buffer_context);

    auto options = MlDriftClDelegateDefaultOptionsPtr();
    options->force_host_ptr_io_buffers = true;
    options->runtime_context = LrtGetRuntimeContext();

    auto delegate = CreateMlDriftClDelegate(std::move(options), environment);
    ASSERT_NE(delegate, nullptr);
    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(std::move(delegate)),
              kTfLiteOk);
    ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    const TfLiteTensor* in_tensor =
        interpreter->tensor(interpreter->inputs()[0]);
    auto in_req_or = buffer_context.GetBufferRequirements(in_tensor);
    ASSERT_TRUE(in_req_or.HasValue());
    size_t in_req_alignment = 0;
    ASSERT_EQ(LiteRtGetTensorBufferRequirementsAlignment(
                  const_cast<LiteRtTensorBufferRequirements>(*in_req_or),
                  &in_req_alignment),
              kLiteRtStatusOk);
    EXPECT_GE(in_req_alignment, 4096);

    LiteRtRankedTensorType tensor_type;
    tensor_type.layout.rank = in_tensor->dims->size;
    for (int d = 0; d < in_tensor->dims->size; ++d) {
      tensor_type.layout.dimensions[d] = in_tensor->dims->data[d];
    }
    tensor_type.element_type = kLiteRtElementTypeFloat32;
    void* in_host_ptr = nullptr;
    ASSERT_EQ(posix_memalign(&in_host_ptr, in_req_alignment, in_tensor->bytes),
              0);
    absl::Cleanup free_in = [in_host_ptr]() { free(in_host_ptr); };
    std::memset(in_host_ptr, 0, in_tensor->bytes);

    LiteRtTensorBuffer host_tb = nullptr;
    ASSERT_EQ(LiteRtCreateTensorBufferFromHostMemory(
                  &tensor_type, in_host_ptr, in_tensor->bytes,
                  /*deallocator=*/nullptr, &host_tb),
              kLiteRtStatusOk);
    ASSERT_EQ(buffer_context.RegisterTensorBuffer(
                  in_tensor,
                  LiteRtTensorBufferPtr(host_tb, LiteRtTensorBufferDeleter{})),
              kLiteRtStatusOk);

    ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

    ASSERT_FALSE(interpreter->execution_plan().empty());
    int node_idx = interpreter->execution_plan()[0];
    auto* node_reg = interpreter->node_and_registration(node_idx);
    ASSERT_NE(node_reg, nullptr);
    auto* kernel =
        reinterpret_cast<DelegateKernelLiteRt*>(node_reg->first.user_data);
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->has_host_memory_buffers());
  }
}

TEST(DelegateOpenClTest,
     RegisterBufferRequirementsFallsBackWhenAlignmentApiNull) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }

  const FileToc* fp = simple_add_create();
  auto model = tflite::FlatBufferModel::BuildFromBuffer(fp->data, fp->size);
  ASSERT_TRUE(model);
  std::unique_ptr<tflite::Interpreter> interpreter;
  ASSERT_EQ(
      tflite::InterpreterBuilder(
          *model, tflite::ops::builtin::BuiltinOpResolver())(&interpreter),
      kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);

  LiteRtEnvironment environment = nullptr;
  ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &environment), kLiteRtStatusOk);
  absl::Cleanup env_cleanup = [&]() {
    if (environment) LiteRtDestroyEnvironment(environment);
  };

  auto get_tensor_id = [&interpreter](const TfLiteOpaqueTensor* target_tensor)
      -> litert::internal::TfLiteTensorIdentifier {
    auto tensor_id = litert::internal::GetTensorIdentifier(
        *interpreter, reinterpret_cast<const TfLiteTensor*>(target_tensor));
    if (!tensor_id) return {-1, -1};
    return *tensor_id;
  };
  LiteRtExternalLiteRtBufferContextT buffer_context(environment, get_tensor_id);
  interpreter->SetExternalContext(kTfLiteLiteRtBufferContext, &buffer_context);

  LiteRtRuntimeContext mock_runtime_context = *LrtGetRuntimeContext();
  mock_runtime_context.create_tensor_buffer_requirements_with_alignment =
      nullptr;

  auto options = MlDriftClDelegateDefaultOptionsPtr();
  options->force_host_ptr_io_buffers = true;
  options->runtime_context = &mock_runtime_context;

  auto delegate = CreateMlDriftClDelegate(std::move(options), environment);
  ASSERT_NE(delegate, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(std::move(delegate)),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  const TfLiteTensor* in_tensor = interpreter->tensor(interpreter->inputs()[0]);
  auto in_req_or = buffer_context.GetBufferRequirements(in_tensor);
  ASSERT_TRUE(in_req_or.HasValue());
  size_t in_req_alignment = 0;
  ASSERT_EQ(LiteRtGetTensorBufferRequirementsAlignment(
                const_cast<LiteRtTensorBufferRequirements>(*in_req_or),
                &in_req_alignment),
            kLiteRtStatusOk);
  // When create_tensor_buffer_requirements_with_alignment is null, it falls
  // back to create_tensor_buffer_requirements which defaults to 64 bytes.
  EXPECT_EQ(in_req_alignment, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT);
}

TEST(DelegateOpenClTest,
     RegisterBufferRequirementsFallsBackWhenRuntimeContextStructTruncated) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }

  const FileToc* fp = simple_add_create();
  auto model = tflite::FlatBufferModel::BuildFromBuffer(fp->data, fp->size);
  ASSERT_TRUE(model);
  std::unique_ptr<tflite::Interpreter> interpreter;
  ASSERT_EQ(
      tflite::InterpreterBuilder(
          *model, tflite::ops::builtin::BuiltinOpResolver())(&interpreter),
      kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);

  LiteRtEnvironment environment = nullptr;
  ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &environment), kLiteRtStatusOk);
  absl::Cleanup env_cleanup = [&]() {
    if (environment) LiteRtDestroyEnvironment(environment);
  };

  auto get_tensor_id = [&interpreter](const TfLiteOpaqueTensor* target_tensor)
      -> litert::internal::TfLiteTensorIdentifier {
    auto tensor_id = litert::internal::GetTensorIdentifier(
        *interpreter, reinterpret_cast<const TfLiteTensor*>(target_tensor));
    if (!tensor_id) return {-1, -1};
    return *tensor_id;
  };
  LiteRtExternalLiteRtBufferContextT buffer_context(environment, get_tensor_id);
  interpreter->SetExternalContext(kTfLiteLiteRtBufferContext, &buffer_context);

  // Simulate an older runtime (e.g. v1.2) where struct_size ends before
  // create_tensor_buffer_requirements_with_alignment, but the function pointer
  // slot in memory is non-null. LITERT_ABI_HAS_API should guard against
  // reading/calling the slot and safely fall back to
  // create_tensor_buffer_requirements.
  LiteRtRuntimeContext mock_runtime_context = *LrtGetRuntimeContext();
  mock_runtime_context.abi_header.struct_size = offsetof(
      LiteRtRuntimeContext, create_tensor_buffer_requirements_with_alignment);
  mock_runtime_context.abi_header.minor_version = 2;
  ASSERT_NE(
      mock_runtime_context.create_tensor_buffer_requirements_with_alignment,
      nullptr);

  auto options = MlDriftClDelegateDefaultOptionsPtr();
  options->force_host_ptr_io_buffers = true;
  options->runtime_context = &mock_runtime_context;

  auto delegate = CreateMlDriftClDelegate(std::move(options), environment);
  ASSERT_NE(delegate, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(std::move(delegate)),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  const TfLiteTensor* in_tensor = interpreter->tensor(interpreter->inputs()[0]);
  auto in_req_or = buffer_context.GetBufferRequirements(in_tensor);
  ASSERT_TRUE(in_req_or.HasValue());
  size_t in_req_alignment = 0;
  ASSERT_EQ(LiteRtGetTensorBufferRequirementsAlignment(
                const_cast<LiteRtTensorBufferRequirements>(*in_req_or),
                &in_req_alignment),
            kLiteRtStatusOk);
  // When struct_size is truncated (v1.2 ABI), it falls back to
  // create_tensor_buffer_requirements which defaults to 64 bytes.
  EXPECT_EQ(in_req_alignment, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT);
}

class MockNotFoundPostDispatchInferenceContext
    : public GpuInferenceContextOpenClLitert {
 public:
  using GpuInferenceContextOpenClLitert::GpuInferenceContextOpenClLitert;
  absl::StatusOr<GpuEventHandle> GetPostDispatchEvent(
      bool is_async_execution_mode) override {
    return absl::NotFoundError("Mock: async not supported");
  }
};

class MockTrackingGpuBackendOpenClLitert : public GpuBackendOpenClLitert {
 public:
  using GpuBackendOpenClLitert::GpuBackendOpenClLitert;
  int wait_for_completion_calls_ = 0;
  absl::Status WaitForCompletion() override {
    ++wait_for_completion_calls_;
    return absl::OkStatus();
  }
};

class TestDelegateKernelLiteRt : public DelegateKernelLiteRt {
 public:
  using DelegateKernelLiteRt::set_has_host_memory_buffers;

  TestDelegateKernelLiteRt(const LiteRtRuntimeContext* runtime_context,
                           GpuBackend* backend,
                           std::unique_ptr<GpuInferenceContext> ctx)
      : DelegateKernelLiteRt(runtime_context) {
    backend_ = backend;
    ctx_ = std::move(ctx);
  }
};

TEST(DelegateOpenClTest, AsyncModeWithHostMemoryWaitsWhenPostDispatchNotFound) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }
  ::ml_drift::cl::Environment env;
  if (!::ml_drift::cl::CreateEnvironment(&env).ok()) {
    GTEST_SKIP() << "Unable to create OpenCL environment";
  }

  MockTrackingGpuBackendOpenClLitert backend(
      &env, /*egl_display=*/nullptr, SimpleCache(), LrtGetRuntimeContext());
  auto mock_ctx =
      std::make_unique<MockNotFoundPostDispatchInferenceContext>(&backend);

  TestDelegateKernelLiteRt kernel(LrtGetRuntimeContext(), &backend,
                                  std::move(mock_ctx));

  // Case 1: has_host_memory_buffers is true -> WaitForCompletion MUST be
  // called.
  kernel.set_has_host_memory_buffers(true);
  EXPECT_TRUE(kernel
                  .HandleOutputEvents(/*context=*/nullptr,
                                      /*is_async_execution_mode=*/true)
                  .ok());
  EXPECT_EQ(backend.wait_for_completion_calls_, 1);

  // Case 2: has_host_memory_buffers is false -> WaitForCompletion MUST NOT be
  // called.
  kernel.set_has_host_memory_buffers(false);
  EXPECT_TRUE(kernel
                  .HandleOutputEvents(/*context=*/nullptr,
                                      /*is_async_execution_mode=*/true)
                  .ok());
  EXPECT_EQ(backend.wait_for_completion_calls_, 1);
}

TEST(DelegateOpenClTest, HostMemorySyncHostMemoryToDeviceAndToHost) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }
  ::ml_drift::cl::Environment env;
  if (!::ml_drift::cl::CreateEnvironment(&env).ok()) {
    GTEST_SKIP() << "Unable to create OpenCL environment";
  }

  GpuBackendOpenClLitert backend(&env, /*egl_display=*/nullptr, SimpleCache(),
                                 LrtGetRuntimeContext());
  backend.SetForceHostPtrIOBuffer(true);

  const size_t alignment =
      std::max(backend.GetHostPtrAlignment(),
               static_cast<size_t>(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT));
  constexpr size_t kBufferSize = 64;
  void* host_ptr = nullptr;
  ASSERT_EQ(posix_memalign(&host_ptr, alignment, kBufferSize), 0);
  absl::Cleanup free_ptr = [host_ptr]() { free(host_ptr); };

  LiteRtRankedTensorType tensor_type;
  tensor_type.layout.rank = 1;
  tensor_type.layout.dimensions[0] = 16;
  tensor_type.element_type = kLiteRtElementTypeFloat32;

  LiteRtTensorBuffer tensor_buffer = nullptr;
  ASSERT_EQ(LiteRtCreateTensorBufferFromHostMemory(
                &tensor_type, host_ptr, kBufferSize, /*deallocator=*/nullptr,
                &tensor_buffer),
            kLiteRtStatusOk);
  GpuTensorBufferPtr tb_ptr(tensor_buffer,
                            GpuTensorBufferDeleter{LiteRtDestroyTensorBuffer});

  auto handle_or = backend.GetGpuMemoryAllocated(tb_ptr);
  ASSERT_TRUE(handle_or.ok());

  const GpuIOBuffer* wrapped =
      backend.GetWrappedHostBufferForTesting(tb_ptr.get());
  ASSERT_NE(wrapped, nullptr);
  EXPECT_TRUE(wrapped->IsMapped());

  // Sync to device unmaps the buffer.
  EXPECT_TRUE(backend.SyncHostMemoryToDevice(tb_ptr).ok());
  EXPECT_FALSE(wrapped->IsMapped());

  // Sync to host remaps the buffer, WaitForCompletion + Finish completes sync.
  EXPECT_TRUE(backend.SyncHostMemoryToHost(tb_ptr).ok());
  EXPECT_TRUE(wrapped->IsMapped());
  EXPECT_TRUE(backend.WaitForCompletion().ok());
  EXPECT_TRUE(backend.FinishHostMemorySyncToHost(tb_ptr).ok());

  // PrepareHostMemoryForDeviceWrite unmaps before GPU write without host copy.
  EXPECT_TRUE(backend.PrepareHostMemoryForDeviceWrite(tb_ptr).ok());
  EXPECT_FALSE(wrapped->IsMapped());

  backend.ClearWrappedHostBuffers();
  EXPECT_EQ(backend.num_wrapped_host_buffers(), 0);
  EXPECT_EQ(backend.GetWrappedHostBufferForTesting(tb_ptr.get()), nullptr);
}

class StubGpuIOBuffer : public GpuIOBuffer {
 public:
  absl::Status Read(absl::Span<uint8_t> data) override {
    return absl::OkStatus();
  }
  absl::Status Write(absl::Span<const uint8_t> data) override {
    return absl::OkStatus();
  }
  size_t GetSize() const override { return 0; }
};

TEST(DelegateOpenClTest, RequiresExplicitHostSyncReflectsEnvironmentPresence) {
  GpuBackendOpenCl null_backend(/*env=*/nullptr);
  EXPECT_FALSE(null_backend.RequiresExplicitHostSync());

  ::ml_drift::cl::Environment dummy_env;
  GpuBackendOpenCl env_backend(&dummy_env);
  EXPECT_TRUE(env_backend.RequiresExplicitHostSync());
}

TEST(DelegateOpenClTest, GpuBufferRequirementsAndBaseDefaults) {
  GpuBackend::GpuBufferRequirements reqs;
  EXPECT_EQ(reqs.alignment, 0);

  StubGpuIOBuffer stub_buffer;
  EXPECT_FALSE(stub_buffer.IsMapped());
  EXPECT_TRUE(stub_buffer.SyncToDevice().ok());
  EXPECT_TRUE(stub_buffer.UnmapIfMapped().ok());
  EXPECT_TRUE(stub_buffer.SyncToHost(/*wait_for_completion=*/false).ok());
  EXPECT_TRUE(stub_buffer.SyncToHost(/*wait_for_completion=*/true).ok());
  EXPECT_TRUE(stub_buffer.SyncToHost().ok());
  EXPECT_TRUE(stub_buffer.FinishPendingHostSync().ok());

  GpuBackendOpenCl null_backend(/*env=*/nullptr);
  GpuBackend* base_backend = &null_backend;
  GpuTensorBufferPtr empty_tb;
  EXPECT_FALSE(base_backend->RequiresExplicitHostSync());
  EXPECT_TRUE(base_backend->SyncHostMemoryToDevice(empty_tb).ok());
  EXPECT_TRUE(base_backend->PrepareHostMemoryForDeviceWrite(empty_tb).ok());
  EXPECT_TRUE(base_backend->SyncHostMemoryToHost(empty_tb).ok());
  EXPECT_TRUE(base_backend->FinishHostMemorySyncToHost(empty_tb).ok());
  base_backend->ClearWrappedHostBuffers();
}

TEST(DelegateOpenClTest, GpuIOBufferOpenClEmptyAndNullQueueState) {
  auto buf_or = GpuIOBufferOpenCl::Create(
      /*env=*/nullptr, ::ml_drift::cl::Buffer(), /*is_host_ptr=*/true,
      /*host_ptr=*/nullptr, /*owns_host_ptr=*/false,
      /*requires_explicit_host_sync=*/true);
  ASSERT_TRUE(buf_or.ok());
  auto buf = *std::move(buf_or);
  EXPECT_TRUE(buf->IsHostPtr());
  EXPECT_FALSE(buf->IsMapped());
  EXPECT_EQ(buf->GetSize(), 0);
  EXPECT_TRUE(buf->SyncToDevice().ok());
  EXPECT_TRUE(buf->UnmapIfMapped().ok());
  EXPECT_TRUE(buf->SyncToHost(/*wait_for_completion=*/false).ok());
  EXPECT_TRUE(buf->SyncToHost(/*wait_for_completion=*/true).ok());
  EXPECT_TRUE(buf->SyncToHost().ok());
  EXPECT_TRUE(buf->FinishPendingHostSync().ok());
}

TEST(DelegateOpenClTest, HostIOBufferUnmapIfMappedAndAsyncSyncToHost) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }
  ::ml_drift::cl::Environment env;
  if (!::ml_drift::cl::CreateEnvironment(&env).ok()) {
    GTEST_SKIP() << "Unable to create OpenCL environment";
  }
  GpuBackendOpenCl backend(&env);
  backend.SetForceHostPtrIOBuffer(true);
  ASSERT_TRUE(backend.RequiresExplicitHostSync());

  const size_t alignment = backend.GetHostPtrAlignment();
  ASSERT_GT(alignment, 0);
  void* host_ptr = nullptr;
  ASSERT_EQ(posix_memalign(&host_ptr, alignment, 64), 0);
  absl::Cleanup free_ptr = [host_ptr]() { free(host_ptr); };
  auto ext_bytes = absl::MakeSpan(static_cast<uint8_t*>(host_ptr), 64);
  for (size_t i = 0; i < ext_bytes.size(); ++i) {
    ext_bytes[i] = static_cast<uint8_t>(i + 10);
  }

  // Wrapping pre-populated external input preserves dirty CPU contents across
  // initial map.
  auto in_res = backend.WrapExternalHostBuffer(::ml_drift::DataType::kFloat32,
                                               64, /*input=*/true, host_ptr);
  ASSERT_TRUE(in_res.ok());
  auto in_buf = *std::move(in_res);
  EXPECT_TRUE(in_buf->IsMapped());

  std::vector<uint8_t> readback(64, 0);
  EXPECT_TRUE(in_buf->SyncToDevice().ok());
  EXPECT_FALSE(in_buf->IsMapped());
  EXPECT_TRUE(in_buf->Read(absl::MakeSpan(readback)).ok());
  EXPECT_TRUE(in_buf->IsMapped());
  for (size_t i = 0; i < readback.size(); ++i) {
    EXPECT_EQ(readback[i], static_cast<uint8_t>(i + 10));
  }

  // Cold unmapped SyncToDevice fallback preserves subsequent host updates.
  EXPECT_TRUE(in_buf->UnmapIfMapped().ok());
  EXPECT_FALSE(in_buf->IsMapped());
  ext_bytes[0] = 77;
  ext_bytes[63] = 88;
  EXPECT_TRUE(in_buf->SyncToDevice().ok());
  EXPECT_FALSE(in_buf->IsMapped());
  EXPECT_TRUE(in_buf->Read(absl::MakeSpan(readback)).ok());
  EXPECT_EQ(readback[0], 77);
  EXPECT_EQ(readback[63], 88);

  // Output buffer UnmapIfMapped + async SyncToHost(false) +
  // FinishPendingHostSync.
  auto out_res = backend.CreateInternalHostBuffer(
      ::ml_drift::DataType::kFloat32, /*size=*/64, /*input=*/false);
  ASSERT_TRUE(out_res.ok());
  auto out_buf = *std::move(out_res);
  EXPECT_TRUE(out_buf->IsMapped());
  EXPECT_TRUE(out_buf->UnmapIfMapped().ok());
  EXPECT_FALSE(out_buf->IsMapped());
  EXPECT_TRUE(out_buf->UnmapIfMapped().ok());
  EXPECT_FALSE(out_buf->IsMapped());

  std::vector<uint8_t> gpu_payload(64, 0x5A);
  auto* opencl_out = static_cast<GpuIOBufferOpenCl*>(out_buf.get());
  ASSERT_TRUE(env.queue()
                  ->EnqueueWriteBuffer(opencl_out->cl_buffer().GetMemoryPtr(),
                                       gpu_payload.size(), gpu_payload.data(),
                                       /*async=*/true)
                  .ok());
  EXPECT_TRUE(out_buf->SyncToHost(/*wait_for_completion=*/false).ok());
  EXPECT_TRUE(out_buf->IsMapped());
  EXPECT_TRUE(env.queue()->WaitForCompletion().ok());
  EXPECT_TRUE(out_buf->FinishPendingHostSync().ok());

  auto out_bytes =
      absl::MakeSpan(static_cast<const uint8_t*>(out_buf->GetHostPtr()), 64);
  for (uint8_t b : out_bytes) {
    EXPECT_EQ(b, 0x5A);
  }

  // Blocking SyncToHost(true) after async SyncToHost(false) waits and syncs.
  EXPECT_TRUE(out_buf->UnmapIfMapped().ok());
  std::vector<uint8_t> gpu_payload2(64, 0x3C);
  ASSERT_TRUE(env.queue()
                  ->EnqueueWriteBuffer(opencl_out->cl_buffer().GetMemoryPtr(),
                                       gpu_payload2.size(), gpu_payload2.data(),
                                       /*async=*/true)
                  .ok());
  EXPECT_TRUE(out_buf->SyncToHost(/*wait_for_completion=*/false).ok());
  EXPECT_TRUE(out_buf->SyncToHost(/*wait_for_completion=*/true).ok());
  for (uint8_t b : out_bytes) {
    EXPECT_EQ(b, 0x3C);
  }

  // UnmapIfMapped, Write, and SyncToDevice safely drain pending_host_sync_.
  EXPECT_TRUE(out_buf->UnmapIfMapped().ok());
  EXPECT_TRUE(out_buf->SyncToHost(/*wait_for_completion=*/false).ok());
  EXPECT_TRUE(out_buf->UnmapIfMapped().ok());
  EXPECT_FALSE(out_buf->IsMapped());

  EXPECT_TRUE(out_buf->SyncToHost(/*wait_for_completion=*/false).ok());
  std::vector<uint8_t> gpu_payload3(64, 0x7E);
  EXPECT_TRUE(out_buf->Write(gpu_payload3).ok());
  EXPECT_FALSE(out_buf->IsMapped());
  EXPECT_TRUE(out_buf->Read(absl::MakeSpan(readback)).ok());
  EXPECT_EQ(readback, gpu_payload3);

  EXPECT_TRUE(out_buf->UnmapIfMapped().ok());
  EXPECT_TRUE(out_buf->SyncToHost(/*wait_for_completion=*/false).ok());
  EXPECT_TRUE(out_buf->SyncToDevice().ok());
  EXPECT_FALSE(out_buf->IsMapped());
}

TEST(DelegateOpenClTest, HostIOBufferConvertersWithHostPtrBuffers) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }
  ::ml_drift::cl::Environment env;
  if (!::ml_drift::cl::CreateEnvironment(&env).ok()) {
    GTEST_SKIP() << "Unable to create OpenCL environment";
  }
  GpuBackendOpenCl backend(&env);
  backend.SetForceHostPtrIOBuffer(true);

  const ::ml_drift::BHWC shape(1, 1, 2, 4);
  const ::ml_drift::TensorDescriptor tensor_desc =
      ::ml_drift::CreateBhwcTensorDescriptor(
          ::ml_drift::DataType::kFloat32,
          ::ml_drift::TensorStorageType::kBuffer, shape);
  ::ml_drift::cl::Tensor tensor;
  ASSERT_TRUE(
      ::ml_drift::cl::CreateTensor(env.context(), tensor_desc, &tensor).ok());

  ::ml_drift::BufferDescriptor buf_desc;
  buf_desc.element_type = ::ml_drift::DataType::kFloat32;
  buf_desc.element_size = 1;
  buf_desc.memory_type = ::ml_drift::MemoryType::kGlobal;

  auto b2t_or = backend.CreateBuffer2TensorConverter(buf_desc, tensor_desc);
  ASSERT_TRUE(b2t_or.ok());
  auto b2t = *std::move(b2t_or);

  auto t2b_or = backend.CreateTensor2BufferConverter(tensor_desc, buf_desc);
  ASSERT_TRUE(t2b_or.ok());
  auto t2b = *std::move(t2b_or);

  const std::vector<float> kInputValues = {1.0f, 2.0f, 3.0f, 4.0f,
                                           5.0f, 6.0f, 7.0f, 8.0f};
  const size_t byte_size = kInputValues.size() * sizeof(float);

  auto in_or = backend.CreateInternalHostBuffer(::ml_drift::DataType::kFloat32,
                                                byte_size, /*input=*/true);
  ASSERT_TRUE(in_or.ok());
  auto in_buf = *std::move(in_or);
  ASSERT_TRUE(in_buf->IsMapped());
  std::memcpy(in_buf->GetHostPtr(), kInputValues.data(), byte_size);

  auto out_or = backend.CreateInternalHostBuffer(::ml_drift::DataType::kFloat32,
                                                 byte_size, /*input=*/false);
  ASSERT_TRUE(out_or.ok());
  auto out_buf = *std::move(out_or);
  ASSERT_TRUE(out_buf->IsMapped());

  EXPECT_TRUE(b2t->Convert(*in_buf, tensor).ok());
  EXPECT_FALSE(in_buf->IsMapped());

  EXPECT_TRUE(t2b->Convert(tensor, *out_buf).ok());
  EXPECT_FALSE(out_buf->IsMapped());

  EXPECT_TRUE(out_buf->SyncToHost(/*wait_for_completion=*/false).ok());
  EXPECT_TRUE(env.queue()->WaitForCompletion().ok());
  EXPECT_TRUE(out_buf->FinishPendingHostSync().ok());
  EXPECT_TRUE(out_buf->IsMapped());

  std::vector<float> output_values(kInputValues.size(), 0.0f);
  std::memcpy(output_values.data(), out_buf->GetHostPtr(), byte_size);
  EXPECT_EQ(output_values, kInputValues);
}

TEST(DelegateOpenClTest, DelegateHostMemoryMultiInvokeWithRecreatedBuffers) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!::ml_drift::cl::LoadOpenCL().ok()) {
    GTEST_SKIP() << "OpenCL not loaded for ml_drift";
  }

  const FileToc* fp = simple_add_create();
  auto model = tflite::FlatBufferModel::BuildFromBuffer(fp->data, fp->size);
  ASSERT_TRUE(model);
  std::unique_ptr<tflite::Interpreter> interpreter;
  ASSERT_EQ(
      tflite::InterpreterBuilder(
          *model, tflite::ops::builtin::BuiltinOpResolver())(&interpreter),
      kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);

  LiteRtEnvironment environment = nullptr;
  ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &environment), kLiteRtStatusOk);
  absl::Cleanup env_cleanup = [&]() {
    if (environment) LiteRtDestroyEnvironment(environment);
  };

  auto get_tensor_id = [&interpreter](const TfLiteOpaqueTensor* target_tensor)
      -> litert::internal::TfLiteTensorIdentifier {
    auto tensor_id = litert::internal::GetTensorIdentifier(
        *interpreter, reinterpret_cast<const TfLiteTensor*>(target_tensor));
    if (!tensor_id) return {-1, -1};
    return *tensor_id;
  };
  LiteRtExternalLiteRtBufferContextT buffer_context(environment, get_tensor_id);
  interpreter->SetExternalContext(kTfLiteLiteRtBufferContext, &buffer_context);

  auto options = MlDriftClDelegateDefaultOptionsPtr();
  options->wait_type = kGpuDelegateWaitTypeDoNotWait;
  options->force_host_ptr_io_buffers = true;
  options->runtime_context = LrtGetRuntimeContext();

  auto delegate = CreateMlDriftClDelegate(std::move(options), environment);
  ASSERT_NE(delegate, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(std::move(delegate)),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_FALSE(interpreter->execution_plan().empty());
  int node_idx = interpreter->execution_plan()[0];
  auto* node_reg = interpreter->node_and_registration(node_idx);
  ASSERT_NE(node_reg, nullptr);
  auto* kernel =
      reinterpret_cast<DelegateKernelLiteRt*>(node_reg->first.user_data);
  ASSERT_NE(kernel, nullptr);
  auto* opencl_backend =
      static_cast<GpuBackendOpenClLitert*>(kernel->backend());
  ASSERT_NE(opencl_backend, nullptr);

  TfLiteTensor* in0 = interpreter->tensor(interpreter->inputs()[0]);
  TfLiteTensor* in1 = interpreter->tensor(interpreter->inputs()[1]);
  TfLiteTensor* out0 = interpreter->tensor(interpreter->outputs()[0]);
  // Simulate CompiledModel::RegisterBuffer() marking tensors backed by
  // buffer_context_ as kTfLiteNonCpu so Upload/DownloadIntermediateCpuTensors
  // does not overwrite them with stale TFLite arena data.
  for (TfLiteTensor* t : {in0, in1, out0}) {
    t->allocation_type = kTfLiteNonCpu;
    t->data.data = nullptr;
  }
  auto make_type = [](const TfLiteTensor* t) {
    LiteRtRankedTensorType type;
    type.layout.rank = t->dims->size;
    for (int d = 0; d < t->dims->size; ++d) {
      type.layout.dimensions[d] = t->dims->data[d];
    }
    type.element_type = kLiteRtElementTypeFloat32;
    return type;
  };

  auto req_or = buffer_context.GetBufferRequirements(in0);
  ASSERT_TRUE(req_or.HasValue());
  size_t alignment = 0;
  ASSERT_EQ(
      LiteRtGetTensorBufferRequirementsAlignment(
          const_cast<LiteRtTensorBufferRequirements>(*req_or), &alignment),
      kLiteRtStatusOk);
  EXPECT_GE(alignment, 4096);

  for (int iter = 0; iter < 2; ++iter) {
    void* p0 = nullptr;
    void* p1 = nullptr;
    void* po = nullptr;
    ASSERT_EQ(posix_memalign(&p0, alignment, in0->bytes), 0);
    ASSERT_EQ(posix_memalign(&p1, alignment, in1->bytes), 0);
    ASSERT_EQ(posix_memalign(&po, alignment, out0->bytes), 0);
    absl::Cleanup free_p = [&]() {
      free(p0);
      free(p1);
      free(po);
    };

    const float v0 = 1.5f + iter;
    const float v1 = 2.25f + iter;
    auto* f0 = static_cast<float*>(p0);
    auto* f1 = static_cast<float*>(p1);
    auto* fo = static_cast<float*>(po);
    const size_t n = out0->bytes / sizeof(float);
    for (size_t i = 0; i < n; ++i) {
      f0[i] = v0 + static_cast<float>(i);
      f1[i] = v1;
      fo[i] = -999.0f;
    }

    auto t0_type = make_type(in0);
    auto t1_type = make_type(in1);
    auto to_type = make_type(out0);
    LiteRtTensorBuffer tb0 = nullptr, tb1 = nullptr, tbo = nullptr;
    ASSERT_EQ(LiteRtCreateTensorBufferFromHostMemory(
                  &t0_type, p0, in0->bytes, /*deallocator=*/nullptr, &tb0),
              kLiteRtStatusOk);
    ASSERT_EQ(LiteRtCreateTensorBufferFromHostMemory(
                  &t1_type, p1, in1->bytes, /*deallocator=*/nullptr, &tb1),
              kLiteRtStatusOk);
    ASSERT_EQ(LiteRtCreateTensorBufferFromHostMemory(
                  &to_type, po, out0->bytes, /*deallocator=*/nullptr, &tbo),
              kLiteRtStatusOk);

    ASSERT_EQ(buffer_context.RegisterTensorBuffer(
                  in0, LiteRtTensorBufferPtr(tb0, LiteRtTensorBufferDeleter{})),
              kLiteRtStatusOk);
    ASSERT_EQ(buffer_context.RegisterTensorBuffer(
                  in1, LiteRtTensorBufferPtr(tb1, LiteRtTensorBufferDeleter{})),
              kLiteRtStatusOk);
    ASSERT_EQ(
        buffer_context.RegisterTensorBuffer(
            out0, LiteRtTensorBufferPtr(tbo, LiteRtTensorBufferDeleter{})),
        kLiteRtStatusOk);

    ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
    EXPECT_EQ(opencl_backend->num_wrapped_host_buffers(), 0);
    for (size_t i = 0; i < n; ++i) {
      EXPECT_NEAR(fo[i], (v0 + static_cast<float>(i)) + v1, 1e-4f);
    }
  }
}

std::vector<uint8_t> BuildQuantizedAddModelBuffer() {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<flatbuffers::Offset<tflite::OperatorCode>> opcodes = {
      tflite::CreateOperatorCode(
          builder,
          /*deprecated_builtin_code=*/tflite::BuiltinOperator_ADD,
          /*custom_code=*/0, /*version=*/1,
          /*builtin_code=*/tflite::BuiltinOperator_ADD),
  };
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
      tflite::CreateBuffer(builder),
  };

  auto make_quant = [&](float scale, int64_t zero_point) {
    return tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0, builder.CreateVector<float>({scale}),
        builder.CreateVector<int64_t>({zero_point}));
  };

  const std::vector<int32_t> shape = {1, 1, 1, 8};
  auto make_q_tensor = [&](const char* name, float scale, int64_t zero_point) {
    return tflite::CreateTensor(builder, builder.CreateVector(shape),
                                tflite::TensorType_UINT8,
                                /*buffer=*/0, builder.CreateString(name),
                                make_quant(scale, zero_point));
  };

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
      make_q_tensor("in0", /*scale=*/0.1f, /*zero_point=*/10),
      make_q_tensor("in1", /*scale=*/0.05f, /*zero_point=*/20),
      make_q_tensor("out", /*scale=*/0.1f, /*zero_point=*/5),
  };

  auto add_opts =
      tflite::CreateAddOptions(builder, tflite::ActivationFunctionType_NONE)
          .Union();
  std::vector<flatbuffers::Offset<tflite::Operator>> ops = {
      tflite::CreateOperator(builder, /*opcode_index=*/0,
                             builder.CreateVector(std::vector<int32_t>{0, 1}),
                             builder.CreateVector(std::vector<int32_t>{2}),
                             tflite::BuiltinOptions_AddOptions, add_opts),
  };

  auto subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors),
      builder.CreateVector(std::vector<int32_t>{0, 1}),
      builder.CreateVector(std::vector<int32_t>{2}), builder.CreateVector(ops),
      builder.CreateString("main"));
  auto model = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(opcodes),
      builder.CreateVector(
          std::vector<flatbuffers::Offset<tflite::SubGraph>>{subgraph}),
      builder.CreateString("Quantized Add test"),
      builder.CreateVector(buffers));
  builder.Finish(model, tflite::ModelIdentifier());
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

TEST(DelegateOpenClTest, QuantizedUInt8AddMatchesCpuAndHostMemoryOnOpenCl) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!IsOpenClSupported()) {
    GTEST_SKIP() << "OpenCL is not supported on this platform";
  }

  const std::vector<uint8_t> model_buf = BuildQuantizedAddModelBuffer();
  auto tflite_model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(model_buf.data()), model_buf.size());
  ASSERT_NE(tflite_model, nullptr);

  const std::vector<uint8_t> in0_vals = {10, 11, 15, 20, 25, 30, 35, 40};
  const std::vector<uint8_t> in1_vals = {20, 24, 28, 32, 36, 40, 44, 48};

  // 1. CPU reference execution.
  std::unique_ptr<tflite::Interpreter> cpu_interp;
  ASSERT_EQ(tflite::InterpreterBuilder(
                *tflite_model,
                tflite::ops::builtin::BuiltinOpResolver())(&cpu_interp),
            kTfLiteOk);
  ASSERT_NE(cpu_interp, nullptr);
  ASSERT_EQ(cpu_interp->AllocateTensors(), kTfLiteOk);
  std::memcpy(cpu_interp->typed_input_tensor<uint8_t>(0), in0_vals.data(),
              in0_vals.size());
  std::memcpy(cpu_interp->typed_input_tensor<uint8_t>(1), in1_vals.data(),
              in1_vals.size());
  ASSERT_EQ(cpu_interp->Invoke(), kTfLiteOk);
  const uint8_t* cpu_out_ptr = cpu_interp->typed_output_tensor<uint8_t>(0);
  ASSERT_NE(cpu_out_ptr, nullptr);
  const std::vector<uint8_t> cpu_out(cpu_out_ptr,
                                     cpu_out_ptr + in0_vals.size());

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  for (bool external_tensors_mode : {false, true}) {
    SCOPED_TRACE(absl::StrCat("external_tensors_mode=", external_tensors_mode));
    auto litert_options = litert::Options::Create();
    ASSERT_TRUE(litert_options);
    ASSERT_TRUE(
        litert_options->SetHardwareAccelerators(litert::HwAccelerators::kGpu));
    auto gpu_options = litert_options->GetOptions<litert::GpuOptions>();
    ASSERT_TRUE(gpu_options);
    ASSERT_TRUE(gpu_options->SetBackend(litert::GpuOptions::Backend::kOpenCl));
    ASSERT_TRUE(
        gpu_options->SetPrecision(litert::GpuOptions::Precision::kFp32));
    ASSERT_EQ(gpu_options->SetSerializeProgramCache(false), kLiteRtStatusOk);
    ASSERT_EQ(gpu_options->EnableExternalTensorsMode(external_tensors_mode),
              kLiteRtStatusOk);

    auto compiled_model = litert::CompiledModel::Create(
        *env, litert::BufferRef<uint8_t>(model_buf.data(), model_buf.size()),
        *litert_options);
    ASSERT_TRUE(compiled_model) << compiled_model.Error().Message();
    auto fully_accelerated = compiled_model->IsFullyAccelerated();
    ASSERT_TRUE(fully_accelerated);
    EXPECT_TRUE(*fully_accelerated);

    auto input_buffers = compiled_model->CreateInputBuffers();
    ASSERT_TRUE(input_buffers);
    ASSERT_EQ(input_buffers->size(), 2);
    auto output_buffers = compiled_model->CreateOutputBuffers();
    ASSERT_TRUE(output_buffers);
    ASSERT_EQ(output_buffers->size(), 1);

    // Boundary quantized tensors are not in GPU external/buffer requirements,
    // so CompiledModel creates HostMemory custom allocations directly.
    auto in_buf_type = (*input_buffers)[0].BufferType();
    ASSERT_TRUE(in_buf_type);
    EXPECT_EQ(*in_buf_type, litert::TensorBufferType::kHostMemory);
    auto out_buf_type = (*output_buffers)[0].BufferType();
    ASSERT_TRUE(out_buf_type);
    EXPECT_EQ(*out_buf_type, litert::TensorBufferType::kHostMemory);

    ASSERT_TRUE(
        (*input_buffers)[0].Write<uint8_t>(absl::MakeConstSpan(in0_vals)));
    ASSERT_TRUE(
        (*input_buffers)[1].Write<uint8_t>(absl::MakeConstSpan(in1_vals)));

    ASSERT_TRUE(compiled_model->Run(*input_buffers, *output_buffers));

    std::vector<uint8_t> gpu_out(in0_vals.size(), 0);
    ASSERT_TRUE((*output_buffers)[0].Read<uint8_t>(absl::MakeSpan(gpu_out)));
    for (size_t i = 0; i < cpu_out.size(); ++i) {
      EXPECT_LE(
          std::abs(static_cast<int>(gpu_out[i]) - static_cast<int>(cpu_out[i])),
          1)
          << "at index " << i << ": gpu=" << static_cast<int>(gpu_out[i])
          << ", cpu=" << static_cast<int>(cpu_out[i]);
    }
  }
}

}  // namespace
}  // namespace litert::ml_drift
