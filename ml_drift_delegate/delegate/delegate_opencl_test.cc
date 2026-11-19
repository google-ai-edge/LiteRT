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
#include <cmath>
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

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "third_party/odml/infra/ml_drift_delegate/testdata/simple_add.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/runtime/external_litert_buffer_context.h"
#include "litert/runtime/tensor_identifier.h"
#include "litert/runtime/tfl_utils.h"
#include "ml_drift_delegate/delegate/gpu_backend_opencl.h"
#include "tflite/c/common.h"
#include "tflite/interpreter.h"
#include "tflite/interpreter_builder.h"
#include "tflite/kernels/register.h"
#include "tflite/model_builder.h"
#include "tflite/schema/schema_generated.h"
#include "tflite/version.h"

namespace litert::ml_drift {
namespace {

using ::absl_testing::IsOk;

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

  std::vector<uint8_t> write_data(64);
  for (size_t i = 0; i < 64; ++i) {
    write_data[i] = static_cast<uint8_t>(i + 1);
  }
  EXPECT_TRUE(buf->Write(write_data).ok());

  std::vector<uint8_t> read_data(64, 0);
  EXPECT_TRUE(buf->Read(absl::MakeSpan(read_data)).ok());
  EXPECT_EQ(write_data, read_data);

  uint8_t* host_bytes = static_cast<uint8_t*>(buf->GetHostPtr());
  ASSERT_NE(host_bytes, nullptr);
  host_bytes[0] = 42;
  host_bytes[1] = 99;
  EXPECT_TRUE(buf->SyncToDevice().ok());
  EXPECT_TRUE(buf->SyncToHost().ok());
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
  EXPECT_TRUE(wrap_buf->SyncToDevice().ok());
  EXPECT_TRUE(wrap_buf->SyncToHost().ok());

  auto dev_res = backend.CreateIOBufferWithSize(::ml_drift::DataType::kFloat32,
                                                /*size=*/64, /*input=*/true);
  ASSERT_TRUE(dev_res.ok());
  auto dev_buf = *std::move(dev_res);
  EXPECT_FALSE(dev_buf->IsHostPtr());
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
    auto data_vec = builder.CreateVector(
        raw_bytes, const_rhs_data->size() * sizeof(float));
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
    auto opts =
        tflite::CreateReshapeOptions(
            builder,
            builder.CreateVector(target_shape.data(), target_shape.size()))
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
    values[i] =
        offset + 0.1f * static_cast<float>((static_cast<int>(i * 7 + 3) % 19) -
                                           9);
  }
  return values;
}

void VerifyBatchMatMulMatchesCpuOnOpenCl(
    const litert::Environment& env, absl::Span<const int32_t> base_lhs_shape,
    absl::Span<const int32_t> base_rhs_shape,
    absl::Span<const int32_t> out_shape, bool adj_x, bool adj_y,
    bool const_rhs, bool use_ir_model) {
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

  const std::vector<uint8_t> model_buf = BuildBatchMatMulModelBuffer(
      lhs_shape, rhs_shape, out_shape, adj_x, adj_y,
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
  ASSERT_TRUE(
      gpu_options->SetPrecision(litert::GpuOptions::Precision::kFp32));
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
  ASSERT_TRUE(
      (*input_buffers)[0].Write<float>(absl::MakeConstSpan(lhs_data)));
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

std::vector<uint8_t> BuildInt4CacheUpdateModelBuffer(int batch_heads,
                                                     int tokens, int cache_size,
                                                     int head_size,
                                                     float scale_k,
                                                     float scale_v) {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<flatbuffers::Offset<tflite::OperatorCode>> opcodes;
  opcodes.push_back(tflite::CreateOperatorCode(
      builder,
      /*deprecated_builtin_code=*/
      tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES,
      /*custom_code=*/0, /*version=*/1,
      /*builtin_code=*/tflite::BuiltinOperator_STABLEHLO_COMPOSITE));
  auto opcodes_vec = builder.CreateVector(opcodes);

  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
  buffers.push_back(tflite::CreateBuffer(builder));

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  auto add_float_or_int_tensor = [&](absl::Span<const int32_t> shape,
                                     tflite::TensorType type,
                                     const char* name) -> int32_t {
    const int32_t idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(tflite::CreateTensor(
        builder, builder.CreateVector(shape.data(), shape.size()), type,
        /*buffer=*/0, builder.CreateString(name)));
    return idx;
  };
  auto add_int4_tensor = [&](absl::Span<const int32_t> shape, float scale,
                             const char* name) -> int32_t {
    const int32_t idx = static_cast<int32_t>(tensors.size());
    auto quant = tflite::CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0,
        builder.CreateVector(std::vector<float>{scale}),
        builder.CreateVector(std::vector<int64_t>{0}));
    tensors.push_back(tflite::CreateTensor(
        builder, builder.CreateVector(shape.data(), shape.size()),
        tflite::TensorType_INT4, /*buffer=*/0, builder.CreateString(name),
        quant));
    return idx;
  };

  const int32_t src_k_idx = add_float_or_int_tensor(
      {1, batch_heads, tokens, head_size}, tflite::TensorType_FLOAT32, "src_k");
  const int32_t src_v_idx = add_float_or_int_tensor(
      {1, batch_heads, tokens, head_size}, tflite::TensorType_FLOAT32, "src_v");
  const int32_t params_idx = add_float_or_int_tensor(
      {1, 1, 1, 7}, tflite::TensorType_INT32, "runtime_params");
  // Name the caches like LiteRT-LM KV caches, which LiteRT-LM binds to BUFFER
  // storage via the `kv_cache_` buffer storage tensor pattern.
  const int32_t out_k_idx = add_int4_tensor(
      {1, batch_heads, cache_size, head_size}, scale_k, "kv_cache_k");
  const int32_t out_v_idx = add_int4_tensor(
      {1, batch_heads, head_size, cache_size}, scale_v, "kv_cache_v");

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("kv_cache_batch_size", batch_heads);
    fbb.Int("cache_size", cache_size);
    fbb.Int("head_size", head_size);
    fbb.Float("scale_k", scale_k);
    fbb.Float("scale_v", scale_v);
  });
  fbb.Finish();
  const std::vector<uint8_t>& flex_attrs = fbb.GetBuffer();

  auto composite_opts =
      tflite::CreateStableHLOCompositeOptions(
          builder, builder.CreateString("odml.cache_update"),
          /*decomposition_subgraph_index=*/1,
          builder.CreateVector(flex_attrs.data(), flex_attrs.size()),
          tflite::CustomOptionsFormat_FLEXBUFFERS, /*version=*/1)
          .Union();
  const std::vector<int32_t> main_inputs = {src_k_idx, src_v_idx, params_idx};
  const std::vector<int32_t> main_outputs = {out_k_idx, out_v_idx};
  const std::vector<flatbuffers::Offset<tflite::Operator>> main_ops = {
      tflite::CreateOperator(
          builder, /*opcode_index=*/0, builder.CreateVector(main_inputs),
          builder.CreateVector(main_outputs), tflite::BuiltinOptions_NONE,
          /*builtin_options=*/0, /*custom_options=*/0,
          tflite::CustomOptionsFormat_FLEXBUFFERS,
          /*mutating_variable_inputs=*/0, /*intermediates=*/0,
          /*large_custom_options_offset=*/0, /*large_custom_options_size=*/0,
          tflite::BuiltinOptions2_StableHLOCompositeOptions, composite_opts)};

  auto main_subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors), builder.CreateVector(main_inputs),
      builder.CreateVector(main_outputs), builder.CreateVector(main_ops),
      builder.CreateString("main"));
  // Placeholder decomposition subgraph at index 1 with matching I/O tensors for
  // TFLite STABLEHLO_COMPOSITE Prepare().
  std::vector<flatbuffers::Offset<tflite::Tensor>> decomp_tensors;
  auto add_decomp_tensor = [&](absl::Span<const int32_t> shape,
                               tflite::TensorType type, const char* name,
                               float scale = 0.0f) -> int32_t {
    const int32_t idx = static_cast<int32_t>(decomp_tensors.size());
    flatbuffers::Offset<tflite::QuantizationParameters> quant = 0;
    if (scale > 0.0f) {
      quant = tflite::CreateQuantizationParameters(
          builder, /*min=*/0, /*max=*/0,
          builder.CreateVector(std::vector<float>{scale}),
          builder.CreateVector(std::vector<int64_t>{0}));
    }
    decomp_tensors.push_back(tflite::CreateTensor(
        builder, builder.CreateVector(shape.data(), shape.size()), type,
        /*buffer=*/0, builder.CreateString(name), quant));
    return idx;
  };
  const int32_t d_in0 = add_decomp_tensor(
      {1, batch_heads, tokens, head_size}, tflite::TensorType_FLOAT32, "d_k");
  const int32_t d_in1 = add_decomp_tensor(
      {1, batch_heads, tokens, head_size}, tflite::TensorType_FLOAT32, "d_v");
  const int32_t d_in2 =
      add_decomp_tensor({1, 1, 1, 7}, tflite::TensorType_INT32, "d_p");
  const int32_t d_out0 = add_decomp_tensor(
      {1, batch_heads, cache_size, head_size}, tflite::TensorType_INT4,
      "d_out_k", scale_k);
  const int32_t d_out1 = add_decomp_tensor(
      {1, batch_heads, head_size, cache_size}, tflite::TensorType_INT4,
      "d_out_v", scale_v);
  const std::vector<flatbuffers::Offset<tflite::Operator>> empty_ops;
  auto decomp_subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(decomp_tensors),
      builder.CreateVector(std::vector<int32_t>{d_in0, d_in1, d_in2}),
      builder.CreateVector(std::vector<int32_t>{d_out0, d_out1}),
      builder.CreateVector(empty_ops), builder.CreateString("decomp"));

  auto model = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, opcodes_vec,
      builder.CreateVector(std::vector<flatbuffers::Offset<tflite::SubGraph>>{
          main_subgraph, decomp_subgraph}),
      builder.CreateString("odml.cache_update int4 test"),
      builder.CreateVector(buffers));
  builder.Finish(model, tflite::ModelIdentifier());
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

TEST(DelegateOpenClTest, AddValuesToCacheInt4ExternalBuffersBothPipelines) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!IsOpenClSupported()) {
    GTEST_SKIP() << "OpenCL is not supported on this platform";
  }

  constexpr int kBatchHeads = 2;
  constexpr int kTokens = 2;
  constexpr int kCacheSize = 16;
  constexpr int kHeadSize = 16;
  constexpr float kScaleK = 0.25f;
  constexpr float kScaleV = 0.5f;
  constexpr size_t kPackedBytes = (kBatchHeads * kCacheSize * kHeadSize) / 2;

  const std::vector<uint8_t> model_buf = BuildInt4CacheUpdateModelBuffer(
      kBatchHeads, kTokens, kCacheSize, kHeadSize, kScaleK, kScaleV);

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  const size_t input_elems = kBatchHeads * kTokens * kHeadSize;
  std::vector<float> src_k(input_elems);
  std::vector<float> src_v(input_elems);
  for (size_t i = 0; i < input_elems; ++i) {
    const int centered = static_cast<int>(i % 21) - 10;
    src_k[i] = static_cast<float>(centered) * kScaleK;
    src_v[i] = static_cast<float>(-centered) * kScaleV;
  }
  constexpr int kStartIdx = 3;
  const std::vector<int32_t> params = {
      kStartIdx, kStartIdx + kTokens, kStartIdx + kTokens, kTokens, 0, 0, 0};

  auto quantize_nibble = [](float val, float scale) -> uint8_t {
    const float clamped =
        std::min(7.0f * scale, std::max(-8.0f * scale, val));
    const float q = std::min(
        15.0f, std::max(0.0f, std::nearbyint(clamped / scale) + 8.0f));
    return static_cast<uint8_t>(q);
  };

  for (bool use_ir_model : {false, true}) {
    SCOPED_TRACE(absl::StrCat("use_ir_model=", use_ir_model));
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
    ASSERT_EQ(gpu_options->EnableExternalTensorsMode(true), kLiteRtStatusOk);
    ASSERT_EQ(gpu_options->AddBufferStorageTensorPattern("kv_cache_"),
              kLiteRtStatusOk);
    ASSERT_EQ(gpu_options->SetUseIrModel(use_ir_model), kLiteRtStatusOk);

    auto compiled_model = litert::CompiledModel::Create(
        *env, litert::BufferRef<uint8_t>(model_buf.data(), model_buf.size()),
        *litert_options);
    ASSERT_TRUE(compiled_model) << compiled_model.Error().Message();
    auto fully_accelerated = compiled_model->IsFullyAccelerated();
    ASSERT_TRUE(fully_accelerated);
    EXPECT_TRUE(*fully_accelerated);

    auto input_buffers = compiled_model->CreateInputBuffers();
    ASSERT_TRUE(input_buffers);
    ASSERT_EQ(input_buffers->size(), 3);
    ASSERT_TRUE((*input_buffers)[0].Write<float>(absl::MakeConstSpan(src_k)));
    ASSERT_TRUE((*input_buffers)[1].Write<float>(absl::MakeConstSpan(src_v)));
    ASSERT_TRUE(
        (*input_buffers)[2].Write<int32_t>(absl::MakeConstSpan(params)));

    auto output_buffers = compiled_model->CreateOutputBuffers();
    ASSERT_TRUE(output_buffers);
    ASSERT_EQ(output_buffers->size(), 2);
    // Verify true 0.5-byte/element GPU buffer allocation!
    auto size_k = (*output_buffers)[0].Size();
    auto size_v = (*output_buffers)[1].Size();
    ASSERT_TRUE(size_k);
    ASSERT_TRUE(size_v);
    EXPECT_EQ(*size_k, kPackedBytes);
    EXPECT_EQ(*size_v, kPackedBytes);
    auto packed_size_k = (*output_buffers)[0].PackedSize();
    auto packed_size_v = (*output_buffers)[1].PackedSize();
    ASSERT_TRUE(packed_size_k);
    ASSERT_TRUE(packed_size_v);
    EXPECT_EQ(*packed_size_k, kPackedBytes);
    EXPECT_EQ(*packed_size_v, kPackedBytes);

    std::vector<uint8_t> init_cache(kPackedBytes, 0x3C);
    ASSERT_TRUE(
        (*output_buffers)[0].Write<uint8_t>(absl::MakeConstSpan(init_cache)));
    ASSERT_TRUE(
        (*output_buffers)[1].Write<uint8_t>(absl::MakeConstSpan(init_cache)));

    ASSERT_TRUE(compiled_model->Run(*input_buffers, *output_buffers));

    std::vector<uint8_t> actual_k(kPackedBytes);
    std::vector<uint8_t> actual_v(kPackedBytes);
    ASSERT_TRUE((*output_buffers)[0].Read<uint8_t>(absl::MakeSpan(actual_k)));
    ASSERT_TRUE((*output_buffers)[1].Read<uint8_t>(absl::MakeSpan(actual_v)));

    std::vector<uint8_t> expected_k = init_cache;
    std::vector<uint8_t> expected_v = init_cache;
    constexpr int kPackedHeadSlices = kHeadSize / 8;
    constexpr int kCacheSlices = kCacheSize / 4;
    for (int b = 0; b < kBatchHeads; ++b) {
      for (int t = 0; t < kTokens; ++t) {
        const int tok = kStartIdx + t;
        const int tok_slice = tok / 4;
        const int tok_in_slice = tok % 4;
        for (int s = 0; s < kPackedHeadSlices; ++s) {
          const int k_vec4 =
              ((b * kPackedHeadSlices + s) * kCacheSlices + tok_slice) * 4 +
              tok_in_slice;
          const int v_vec4 =
              ((b * kCacheSlices + tok_slice) * kPackedHeadSlices + s) * 4 +
              tok_in_slice;
          for (int comp = 0; comp < 4; ++comp) {
            const size_t src_lo =
                (b * kTokens + t) * kHeadSize + (s * 8 + comp);
            const size_t src_hi =
                (b * kTokens + t) * kHeadSize + (s * 8 + 4 + comp);
            expected_k[k_vec4 * 4 + comp] = static_cast<uint8_t>(
                quantize_nibble(src_k[src_lo], kScaleK) |
                (quantize_nibble(src_k[src_hi], kScaleK) << 4));
            expected_v[v_vec4 * 4 + comp] = static_cast<uint8_t>(
                quantize_nibble(src_v[src_lo], kScaleV) |
                (quantize_nibble(src_v[src_hi], kScaleV) << 4));
          }
        }
      }
    }
    EXPECT_EQ(actual_k, expected_k);
    EXPECT_EQ(actual_v, expected_v);
  }
}

// Builds a model in which QUANTIZE ops produce the INT4 K/V slices of an
// `odml.cache_update`. `with_layout_ops` adds RESHAPE and TRANSPOSE ops between
// the QUANTIZE ops and the composite.
std::vector<uint8_t> BuildInt4QuantizedSlicesCacheUpdateModelBuffer(
    int batch_heads, int tokens, int cache_size, int head_size, float scale_k,
    float scale_v, bool with_layout_ops) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers;
  buffers.push_back(tflite::CreateBuffer(builder));

  std::vector<flatbuffers::Offset<tflite::OperatorCode>> opcodes;
  opcodes.push_back(tflite::CreateOperatorCode(
      builder, tflite::BuiltinOperator_QUANTIZE, /*custom_code=*/0,
      /*version=*/4, tflite::BuiltinOperator_QUANTIZE));
  opcodes.push_back(tflite::CreateOperatorCode(
      builder, tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES,
      /*custom_code=*/0, /*version=*/1,
      tflite::BuiltinOperator_STABLEHLO_COMPOSITE));
  opcodes.push_back(tflite::CreateOperatorCode(
      builder, tflite::BuiltinOperator_RESHAPE, /*custom_code=*/0,
      /*version=*/1, tflite::BuiltinOperator_RESHAPE));
  opcodes.push_back(tflite::CreateOperatorCode(
      builder, tflite::BuiltinOperator_TRANSPOSE, /*custom_code=*/0,
      /*version=*/1, tflite::BuiltinOperator_TRANSPOSE));
  auto opcodes_vec = builder.CreateVector(opcodes);

  auto make_quant = [&](float scale) {
    return tflite::CreateQuantizationParameters(
        builder, 0, 0, builder.CreateVector(std::vector<float>{scale}),
        builder.CreateVector(std::vector<int64_t>{0}));
  };

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  auto add_tensor =
      [&](const std::vector<int32_t>& shape, tflite::TensorType type,
          const char* name,
          flatbuffers::Offset<tflite::QuantizationParameters> quant = 0) {
        const int32_t idx = static_cast<int32_t>(tensors.size());
        tensors.push_back(
            tflite::CreateTensor(builder, builder.CreateVector(shape), type,
                                 /*buffer=*/0, builder.CreateString(name),
                                 quant));
        return idx;
      };

  const int32_t in_k_idx = add_tensor(
      {1, batch_heads, tokens, head_size}, tflite::TensorType_FLOAT32, "in_k");
  const int32_t in_v_idx = add_tensor(
      {1, batch_heads, tokens, head_size}, tflite::TensorType_FLOAT32, "in_v");
  const int32_t params_idx =
      add_tensor({1, 1, 1, 7}, tflite::TensorType_INT32, "runtime_params");
  // In-graph QUANTIZE ops produce the INT4 K/V slices.
  const int32_t q_k_idx =
      add_tensor({1, batch_heads, tokens, head_size}, tflite::TensorType_INT4,
                 "src_k_int4", make_quant(scale_k));
  const int32_t q_v_idx =
      add_tensor({1, batch_heads, tokens, head_size}, tflite::TensorType_INT4,
                 "src_v_int4", make_quant(scale_v));
  // With layout ops, each INT4 slice goes through a pair of ops that restores
  // its layout before `odml.cache_update` reads it: two RESHAPE ops for K and
  // two TRANSPOSE ops for V.
  int32_t src_k_idx = q_k_idx;
  int32_t src_v_idx = q_v_idx;
  std::vector<flatbuffers::Offset<tflite::Operator>> layout_ops;
  if (with_layout_ops) {
    const std::vector<int32_t> perm = {0, 1, 3, 2};
    buffers.push_back(tflite::CreateBuffer(
        builder,
        builder.CreateVector(reinterpret_cast<const uint8_t*>(perm.data()),
                             perm.size() * sizeof(int32_t))));
    const int32_t perm_idx = static_cast<int32_t>(tensors.size());
    tensors.push_back(tflite::CreateTensor(
        builder, builder.CreateVector(std::vector<int32_t>{4}),
        tflite::TensorType_INT32, static_cast<uint32_t>(buffers.size() - 1),
        builder.CreateString("perm")));
    const std::vector<int32_t> slice_shape = {1, batch_heads, tokens,
                                              head_size};
    const std::vector<int32_t> flat_shape = {1, batch_heads * tokens, 1,
                                             head_size};
    const int32_t flat_k_idx =
        add_tensor(flat_shape, tflite::TensorType_INT4, "src_k_flat",
                   make_quant(scale_k));
    src_k_idx = add_tensor(slice_shape, tflite::TensorType_INT4,
                           "src_k_restored", make_quant(scale_k));
    const int32_t swapped_v_idx =
        add_tensor({1, batch_heads, head_size, tokens},
                   tflite::TensorType_INT4, "src_v_swapped",
                   make_quant(scale_v));
    src_v_idx = add_tensor(slice_shape, tflite::TensorType_INT4,
                           "src_v_restored", make_quant(scale_v));
    auto add_reshape = [&](int32_t input_idx, int32_t output_idx,
                           const std::vector<int32_t>& new_shape) {
      const auto options =
          tflite::CreateReshapeOptions(builder, builder.CreateVector(new_shape))
              .Union();
      const auto inputs =
          builder.CreateVector(std::vector<int32_t>{input_idx});
      const auto outputs =
          builder.CreateVector(std::vector<int32_t>{output_idx});
      layout_ops.push_back(tflite::CreateOperator(
          builder, /*opcode_index=*/2, inputs, outputs,
          tflite::BuiltinOptions_ReshapeOptions, options));
    };
    auto add_transpose = [&](int32_t input_idx, int32_t output_idx) {
      const auto options = tflite::CreateTransposeOptions(builder).Union();
      const auto inputs =
          builder.CreateVector(std::vector<int32_t>{input_idx, perm_idx});
      const auto outputs =
          builder.CreateVector(std::vector<int32_t>{output_idx});
      layout_ops.push_back(tflite::CreateOperator(
          builder, /*opcode_index=*/3, inputs, outputs,
          tflite::BuiltinOptions_TransposeOptions, options));
    };
    add_reshape(q_k_idx, flat_k_idx, flat_shape);
    add_reshape(flat_k_idx, src_k_idx, slice_shape);
    add_transpose(q_v_idx, swapped_v_idx);
    add_transpose(swapped_v_idx, src_v_idx);
  }
  const int32_t out_k_idx =
      add_tensor({1, batch_heads, cache_size, head_size},
                 tflite::TensorType_INT4, "kv_cache_k", make_quant(scale_k));
  const int32_t out_v_idx =
      add_tensor({1, batch_heads, head_size, cache_size},
                 tflite::TensorType_INT4, "kv_cache_v", make_quant(scale_v));

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("kv_cache_batch_size", batch_heads);
    fbb.Int("cache_size", cache_size);
    fbb.Int("head_size", head_size);
    fbb.Float("scale_k", scale_k);
    fbb.Float("scale_v", scale_v);
  });
  fbb.Finish();
  const auto& flex_attrs = fbb.GetBuffer();
  auto composite_opts =
      tflite::CreateStableHLOCompositeOptions(
          builder, builder.CreateString("odml.cache_update"),
          /*decomposition_subgraph_index=*/1,
          builder.CreateVector(flex_attrs.data(), flex_attrs.size()),
          tflite::CustomOptionsFormat_FLEXBUFFERS, /*version=*/1)
          .Union();

  std::vector<flatbuffers::Offset<tflite::Operator>> main_ops = {
      tflite::CreateOperator(
          builder, /*opcode_index=*/0,
          builder.CreateVector(std::vector<int32_t>{in_k_idx}),
          builder.CreateVector(std::vector<int32_t>{q_k_idx})),
      tflite::CreateOperator(
          builder, /*opcode_index=*/0,
          builder.CreateVector(std::vector<int32_t>{in_v_idx}),
          builder.CreateVector(std::vector<int32_t>{q_v_idx})),
      tflite::CreateOperator(
          builder, /*opcode_index=*/1,
          builder.CreateVector(std::vector<int32_t>{src_k_idx, src_v_idx,
                                                    params_idx}),
          builder.CreateVector(std::vector<int32_t>{out_k_idx, out_v_idx}),
          tflite::BuiltinOptions_NONE, 0, 0,
          tflite::CustomOptionsFormat_FLEXBUFFERS, 0, 0, 0, 0,
          tflite::BuiltinOptions2_StableHLOCompositeOptions, composite_opts),
  };
  // The layout ops run between the QUANTIZE ops and `odml.cache_update`.
  main_ops.insert(main_ops.begin() + 2, layout_ops.begin(), layout_ops.end());

  auto main_subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors),
      builder.CreateVector(
          std::vector<int32_t>{in_k_idx, in_v_idx, params_idx}),
      builder.CreateVector(std::vector<int32_t>{out_k_idx, out_v_idx}),
      builder.CreateVector(main_ops), builder.CreateString("main"));

  std::vector<flatbuffers::Offset<tflite::Tensor>> decomp_tensors;
  auto add_dt = [&](const std::vector<int32_t>& shape, tflite::TensorType type,
                    float scale = 0.0f) {
    const int32_t id = static_cast<int32_t>(decomp_tensors.size());
    decomp_tensors.push_back(tflite::CreateTensor(
        builder, builder.CreateVector(shape), type, 0,
        builder.CreateString("d"), scale > 0.0f ? make_quant(scale) : 0));
    return id;
  };
  const int32_t d0 = add_dt({1, batch_heads, tokens, head_size},
                            tflite::TensorType_INT4, scale_k);
  const int32_t d1 = add_dt({1, batch_heads, tokens, head_size},
                            tflite::TensorType_INT4, scale_v);
  const int32_t d2 = add_dt({1, 1, 1, 7}, tflite::TensorType_INT32);
  const int32_t do0 = add_dt({1, batch_heads, cache_size, head_size},
                             tflite::TensorType_INT4, scale_k);
  const int32_t do1 = add_dt({1, batch_heads, head_size, cache_size},
                             tflite::TensorType_INT4, scale_v);
  const std::vector<flatbuffers::Offset<tflite::Operator>> empty_ops;
  auto decomp_subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(decomp_tensors),
      builder.CreateVector(std::vector<int32_t>{d0, d1, d2}),
      builder.CreateVector(std::vector<int32_t>{do0, do1}),
      builder.CreateVector(empty_ops), builder.CreateString("decomp"));

  auto model = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, opcodes_vec,
      builder.CreateVector(std::vector<flatbuffers::Offset<tflite::SubGraph>>{
          main_subgraph, decomp_subgraph}),
      builder.CreateString("int4 quantized_kv_slices cache_update test"),
      builder.CreateVector(buffers));
  builder.Finish(model, tflite::ModelIdentifier());
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

TEST(DelegateOpenClTest,
     AddValuesToCacheInt4QuantizedSlicesBothPipelines) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!IsOpenClSupported()) {
    GTEST_SKIP() << "OpenCL is not supported on this platform";
  }

  constexpr int kBatchHeads = 2;
  constexpr int kTokens = 2;
  constexpr int kCacheSize = 16;
  constexpr int kHeadSize = 16;
  constexpr float kScaleK = 0.25f;
  constexpr float kScaleV = 0.5f;
  constexpr size_t kPackedBytes = (kBatchHeads * kCacheSize * kHeadSize) / 2;

  auto env = litert::Environment::Create({});
  ASSERT_TRUE(env);

  const size_t input_elems = kBatchHeads * kTokens * kHeadSize;
  std::vector<float> src_k(input_elems);
  std::vector<float> src_v(input_elems);
  for (size_t i = 0; i < input_elems; ++i) {
    const int centered = static_cast<int>(i % 15) - 7;
    src_k[i] = static_cast<float>(centered) * kScaleK;
    src_v[i] = static_cast<float>(-centered) * kScaleV;
  }
  constexpr int kStartIdx = 2;
  const std::vector<int32_t> params = {
      kStartIdx, kStartIdx + kTokens, kStartIdx + kTokens, kTokens, 0, 0, 0};

  auto quantize_nibble = [](float val, float scale) -> uint8_t {
    const float clamped =
        std::min(7.0f * scale, std::max(-8.0f * scale, val));
    const float q = std::min(
        15.0f, std::max(0.0f, std::nearbyint(clamped / scale) + 8.0f));
    return static_cast<uint8_t>(q);
  };

  for (const auto [use_ir_model, with_layout_ops] :
       std::vector<std::pair<bool, bool>>{
           {false, false}, {true, false}, {false, true}, {true, true}}) {
    SCOPED_TRACE(absl::StrCat("use_ir_model=", use_ir_model,
                              " with_layout_ops=", with_layout_ops));
    const std::vector<uint8_t> model_buf =
        BuildInt4QuantizedSlicesCacheUpdateModelBuffer(
            kBatchHeads, kTokens, kCacheSize, kHeadSize, kScaleK, kScaleV,
            with_layout_ops);
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
    ASSERT_EQ(gpu_options->EnableExternalTensorsMode(true), kLiteRtStatusOk);
    ASSERT_EQ(gpu_options->AddBufferStorageTensorPattern("kv_cache_"),
              kLiteRtStatusOk);
    ASSERT_EQ(gpu_options->SetUseIrModel(use_ir_model), kLiteRtStatusOk);

    auto compiled_model = litert::CompiledModel::Create(
        *env, litert::BufferRef<uint8_t>(model_buf.data(), model_buf.size()),
        *litert_options);
    ASSERT_TRUE(compiled_model) << compiled_model.Error().Message();
    auto fully_accelerated = compiled_model->IsFullyAccelerated();
    ASSERT_TRUE(fully_accelerated);
    EXPECT_TRUE(*fully_accelerated);

    auto input_buffers = compiled_model->CreateInputBuffers();
    ASSERT_TRUE(input_buffers);
    ASSERT_TRUE((*input_buffers)[0].Write<float>(absl::MakeConstSpan(src_k)));
    ASSERT_TRUE((*input_buffers)[1].Write<float>(absl::MakeConstSpan(src_v)));
    ASSERT_TRUE(
        (*input_buffers)[2].Write<int32_t>(absl::MakeConstSpan(params)));

    auto output_buffers = compiled_model->CreateOutputBuffers();
    ASSERT_TRUE(output_buffers);
    std::vector<uint8_t> init_cache(kPackedBytes, 0x29);
    ASSERT_TRUE(
        (*output_buffers)[0].Write<uint8_t>(absl::MakeConstSpan(init_cache)));
    ASSERT_TRUE(
        (*output_buffers)[1].Write<uint8_t>(absl::MakeConstSpan(init_cache)));

    ASSERT_TRUE(compiled_model->Run(*input_buffers, *output_buffers));

    std::vector<uint8_t> actual_k(kPackedBytes);
    std::vector<uint8_t> actual_v(kPackedBytes);
    ASSERT_TRUE((*output_buffers)[0].Read<uint8_t>(absl::MakeSpan(actual_k)));
    ASSERT_TRUE((*output_buffers)[1].Read<uint8_t>(absl::MakeSpan(actual_v)));

    std::vector<uint8_t> expected_k = init_cache;
    std::vector<uint8_t> expected_v = init_cache;
    constexpr int kPackedHeadSlices = kHeadSize / 8;
    constexpr int kCacheSlices = kCacheSize / 4;
    for (int b = 0; b < kBatchHeads; ++b) {
      for (int t = 0; t < kTokens; ++t) {
        const int tok = kStartIdx + t;
        const int tok_slice = tok / 4;
        const int tok_in_slice = tok % 4;
        for (int s = 0; s < kPackedHeadSlices; ++s) {
          const int k_vec4 =
              ((b * kPackedHeadSlices + s) * kCacheSlices + tok_slice) * 4 +
              tok_in_slice;
          const int v_vec4 =
              ((b * kCacheSlices + tok_slice) * kPackedHeadSlices + s) * 4 +
              tok_in_slice;
          for (int comp = 0; comp < 4; ++comp) {
            const size_t src_lo =
                (b * kTokens + t) * kHeadSize + (s * 8 + comp);
            const size_t src_hi =
                (b * kTokens + t) * kHeadSize + (s * 8 + 4 + comp);
            expected_k[k_vec4 * 4 + comp] = static_cast<uint8_t>(
                quantize_nibble(src_k[src_lo], kScaleK) |
                (quantize_nibble(src_k[src_hi], kScaleK) << 4));
            expected_v[v_vec4 * 4 + comp] = static_cast<uint8_t>(
                quantize_nibble(src_v[src_lo], kScaleV) |
                (quantize_nibble(src_v[src_hi], kScaleV) << 4));
          }
        }
      }
    }
    EXPECT_EQ(actual_k, expected_k);
    EXPECT_EQ(actual_v, expected_v);
  }
}

std::vector<uint8_t> BuildQuantizedAddModelBuffer(
    tflite::TensorType tensor_type = tflite::TensorType_UINT8,
    int64_t in0_zp = 10, int64_t in1_zp = 20, int64_t out_zp = 5) {
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
                                tensor_type,
                                /*buffer=*/0, builder.CreateString(name),
                                make_quant(scale, zero_point));
  };

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors = {
      make_q_tensor("in0", /*scale=*/0.1f, in0_zp),
      make_q_tensor("in1", /*scale=*/0.05f, in1_zp),
      make_q_tensor("out", /*scale=*/0.1f, out_zp),
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

template <typename QuantT>
struct QuantizedAddInputPair {
  std::vector<QuantT> in0;
  std::vector<QuantT> in1;
};

template <typename QuantT>
struct QuantizedAddCase {
  std::vector<QuantT> in0;
  std::vector<QuantT> in1;
  std::vector<QuantT> expected;
};

template <typename QuantT>
absl::Status CompareQuantizedOutputs(absl::Span<const QuantT> actual,
                                     absl::Span<const QuantT> expected,
                                     absl::string_view label) {
  if (actual.size() != expected.size()) {
    return absl::InternalError(
        absl::StrCat(label, " size mismatch: actual=", actual.size(),
                     ", expected=", expected.size()));
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (std::abs(static_cast<int>(actual[i]) - static_cast<int>(expected[i])) >
        1) {
      return absl::FailedPreconditionError(
          absl::StrCat(label, " mismatch at index ", i,
                       ": gpu=", static_cast<int>(actual[i]),
                       ", cpu=", static_cast<int>(expected[i])));
    }
  }
  return absl::OkStatus();
}

template <typename QuantT>
absl::Status RunSingleQuantizedAddConfigOnOpenCl(
    litert::Environment& env, absl::Span<const uint8_t> model_buf,
    litert::GpuOptions::Precision precision, bool external_tensors_mode,
    absl::Span<const QuantizedAddCase<QuantT>> cases) {
  if (cases.empty()) {
    return absl::InvalidArgumentError(
        "QuantizedAddCase span must not be empty.");
  }
  auto litert_options = litert::Options::Create();
  if (!litert_options ||
      !litert_options->SetHardwareAccelerators(litert::HwAccelerators::kGpu)) {
    return absl::InternalError("Failed to configure LiteRT GPU options.");
  }
  auto gpu_options = litert_options->GetOptions<litert::GpuOptions>();
  if (!gpu_options ||
      !gpu_options->SetBackend(litert::GpuOptions::Backend::kOpenCl) ||
      !gpu_options->SetPrecision(precision) ||
      gpu_options->SetSerializeProgramCache(false) != kLiteRtStatusOk ||
      gpu_options->EnableExternalTensorsMode(external_tensors_mode) !=
          kLiteRtStatusOk) {
    return absl::InternalError("Failed to configure OpenCL GpuOptions.");
  }

  auto compiled_model = litert::CompiledModel::Create(
      env, litert::BufferRef<uint8_t>(model_buf.data(), model_buf.size()),
      *litert_options);
  if (!compiled_model) {
    return absl::InternalError(absl::StrCat("CompiledModel::Create failed: ",
                                            compiled_model.Error().Message()));
  }
  auto fully_accelerated = compiled_model->IsFullyAccelerated();
  if (!fully_accelerated || !*fully_accelerated) {
    return absl::FailedPreconditionError("Model is not fully GPU accelerated.");
  }

  auto input_buffers = compiled_model->CreateInputBuffers();
  if (!input_buffers || input_buffers->size() != 2) {
    return absl::InternalError("Unexpected input buffer count.");
  }
  auto output_buffers = compiled_model->CreateOutputBuffers();
  if (!output_buffers || output_buffers->size() != 1) {
    return absl::InternalError("Unexpected output buffer count.");
  }

  // Boundary quantized tensors are not in GPU external/buffer requirements,
  // so CompiledModel creates HostMemory custom allocations directly.
  auto in_buf_type = (*input_buffers)[0].BufferType();
  auto out_buf_type = (*output_buffers)[0].BufferType();
  if (!in_buf_type || *in_buf_type != litert::TensorBufferType::kHostMemory ||
      !out_buf_type || *out_buf_type != litert::TensorBufferType::kHostMemory) {
    return absl::FailedPreconditionError(
        "Expected HostMemory boundary buffers for quantized I/O.");
  }

  for (size_t run_idx = 0; run_idx < cases.size(); ++run_idx) {
    const QuantizedAddCase<QuantT>& test_case = cases[run_idx];
    const std::string run_label = absl::StrCat("run ", run_idx + 1);
    if (!(*input_buffers)[0].Write<QuantT>(
            absl::MakeConstSpan(test_case.in0)) ||
        !(*input_buffers)[1].Write<QuantT>(
            absl::MakeConstSpan(test_case.in1)) ||
        !compiled_model->Run(*input_buffers, *output_buffers)) {
      return absl::InternalError(absl::StrCat(run_label, " execution failed."));
    }
    std::vector<QuantT> gpu_out(test_case.expected.size(), 0);
    if (!(*output_buffers)[0].Read<QuantT>(absl::MakeSpan(gpu_out))) {
      return absl::InternalError(
          absl::StrCat(run_label, " output Read failed."));
    }
    ABSL_RETURN_IF_ERROR(CompareQuantizedOutputs<QuantT>(
        gpu_out, test_case.expected, run_label));
  }

  if (!external_tensors_mode) {
    // Also verify quantized outputs when the caller provides
    // OpenClBufferPacked output buffers.
    const QuantizedAddCase<QuantT>& last_case = cases.back();
    auto packed_type = (*output_buffers)[0].TensorType();
    auto packed_bytes = (*output_buffers)[0].Size();
    if (!packed_type || !packed_bytes) {
      return absl::InternalError("Failed to query output tensor type/size.");
    }
    auto cl_out = litert::TensorBuffer::CreateManaged(
        env, litert::TensorBufferType::kOpenClBufferPacked, *packed_type,
        *packed_bytes);
    if (!cl_out) {
      return absl::InternalError("Failed to create OpenClBufferPacked output.");
    }
    std::vector<litert::TensorBuffer> packed_outputs;
    packed_outputs.push_back(std::move(*cl_out));
    if (!compiled_model->Run(*input_buffers, packed_outputs)) {
      return absl::InternalError("OpenClBufferPacked boundary Run failed.");
    }
    std::vector<QuantT> gpu_out(last_case.expected.size(), 0);
    if (!packed_outputs[0].Read<QuantT>(absl::MakeSpan(gpu_out))) {
      return absl::InternalError("OpenClBufferPacked boundary Read failed.");
    }
    ABSL_RETURN_IF_ERROR(CompareQuantizedOutputs<QuantT>(
        gpu_out, last_case.expected, "OpenClBufferPacked boundary"));
  }
  return absl::OkStatus();
}

template <typename QuantT>
absl::Status VerifyQuantizedAddMatchesCpuOnOpenCl(
    tflite::TensorType tensor_type, int64_t in0_zp, int64_t in1_zp,
    int64_t out_zp, absl::Span<const QuantizedAddInputPair<QuantT>> inputs) {
  if (inputs.empty()) {
    return absl::InvalidArgumentError("Input pairs must not be empty.");
  }
  const std::vector<uint8_t> model_buf =
      BuildQuantizedAddModelBuffer(tensor_type, in0_zp, in1_zp, out_zp);
  auto tflite_model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(model_buf.data()), model_buf.size());
  if (tflite_model == nullptr) {
    return absl::InternalError("Failed to build FlatBufferModel.");
  }

  // 1. CPU reference execution across all input pairs.
  std::unique_ptr<tflite::Interpreter> cpu_interp;
  if (tflite::InterpreterBuilder(*tflite_model,
                                 tflite::ops::builtin::BuiltinOpResolver())(
          &cpu_interp) != kTfLiteOk ||
      cpu_interp == nullptr || cpu_interp->AllocateTensors() != kTfLiteOk) {
    return absl::InternalError("Failed to initialize CPU reference model.");
  }

  auto run_cpu = [&](const QuantizedAddInputPair<QuantT>& pair)
      -> absl::StatusOr<std::vector<QuantT>> {
    QuantT* in0 = cpu_interp->typed_input_tensor<QuantT>(0);
    QuantT* in1 = cpu_interp->typed_input_tensor<QuantT>(1);
    if (in0 == nullptr || in1 == nullptr) {
      return absl::InternalError("CPU reference input tensor is null.");
    }
    std::memcpy(in0, pair.in0.data(), pair.in0.size() * sizeof(QuantT));
    std::memcpy(in1, pair.in1.data(), pair.in1.size() * sizeof(QuantT));
    if (cpu_interp->Invoke() != kTfLiteOk) {
      return absl::InternalError("CPU reference Invoke() failed.");
    }
    const QuantT* ptr = cpu_interp->typed_output_tensor<QuantT>(0);
    if (ptr == nullptr) {
      return absl::InternalError("CPU reference output tensor is null.");
    }
    return std::vector<QuantT>(ptr, ptr + pair.in0.size());
  };

  std::vector<QuantizedAddCase<QuantT>> cases;
  cases.reserve(inputs.size());
  for (const QuantizedAddInputPair<QuantT>& pair : inputs) {
    ABSL_ASSIGN_OR_RETURN(std::vector<QuantT> expected, run_cpu(pair));
    cases.push_back(QuantizedAddCase<QuantT>{
        .in0 = pair.in0,
        .in1 = pair.in1,
        .expected = std::move(expected),
    });
  }

  auto env = litert::Environment::Create({});
  if (!env) {
    return absl::InternalError("Failed to create LiteRT Environment.");
  }

  std::vector<litert::GpuOptions::Precision> precisions = {
      litert::GpuOptions::Precision::kFp32,
      litert::GpuOptions::Precision::kDefault,
  };
  ::ml_drift::cl::Environment cl_env;
  if (::ml_drift::cl::CreateEnvironment(&cl_env).ok() &&
      cl_env.IsSupported(::ml_drift::CalculationsPrecision::kF16)) {
    precisions.push_back(litert::GpuOptions::Precision::kFp16);
  }

  std::vector<std::string> failures;
  for (const litert::GpuOptions::Precision precision : precisions) {
    const absl::string_view precision_name =
        precision == litert::GpuOptions::Precision::kFp32
            ? "fp32"
            : (precision == litert::GpuOptions::Precision::kFp16 ? "fp16"
                                                                 : "default");
    for (const bool external_tensors_mode : {false, true}) {
      const absl::Status status = RunSingleQuantizedAddConfigOnOpenCl<QuantT>(
          *env, model_buf, precision, external_tensors_mode, cases);
      if (!status.ok()) {
        failures.push_back(absl::StrCat(
            "[precision=", precision_name,
            ",external_tensors_mode=", external_tensors_mode, "] ", status));
      }
    }
  }
  if (!failures.empty()) {
    return absl::FailedPreconditionError(absl::StrJoin(failures, "; "));
  }
  return absl::OkStatus();
}

TEST(DelegateOpenClTest, QuantizedUInt8AddMatchesCpuOnOpenCl) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!IsOpenClSupported()) {
    GTEST_SKIP() << "OpenCL is not supported on this platform";
  }

  const std::vector<QuantizedAddInputPair<uint8_t>> inputs = {
      {
          .in0 = {10, 11, 15, 20, 25, 30, 35, 40},
          .in1 = {20, 24, 28, 32, 36, 40, 44, 48},
      },
      {
          .in0 = {45, 50, 55, 60, 65, 70, 75, 80},
          .in1 = {12, 16, 20, 24, 28, 32, 36, 40},
      },
  };
  EXPECT_THAT(VerifyQuantizedAddMatchesCpuOnOpenCl<uint8_t>(
                  tflite::TensorType_UINT8, /*in0_zp=*/10, /*in1_zp=*/20,
                  /*out_zp=*/5, inputs),
              IsOk());
}

TEST(DelegateOpenClTest, QuantizedInt8AddMatchesCpuOnOpenCl) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!IsOpenClSupported()) {
    GTEST_SKIP() << "OpenCL is not supported on this platform";
  }

  const std::vector<QuantizedAddInputPair<int8_t>> inputs = {
      {
          .in0 = {-40, -25, -10, 0, 15, 30, 45, 60},
          .in1 = {-20, -8, 4, 12, 20, 28, 36, 44},
      },
      {
          .in0 = {-50, -30, -15, 5, 25, 40, 55, 70},
          .in1 = {10, -12, 18, -4, 22, 30, -16, 8},
      },
  };
  EXPECT_THAT(VerifyQuantizedAddMatchesCpuOnOpenCl<int8_t>(
                  tflite::TensorType_INT8, /*in0_zp=*/-10, /*in1_zp=*/12,
                  /*out_zp=*/-5, inputs),
              IsOk());
}

TEST(DelegateOpenClTest, QuantizedDelegateOmitsArenaTemporariesAndGuardsShape) {
#if defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
  GTEST_SKIP() << "GPU tests are not supported in MSAN";
#endif
  if (!IsOpenClSupported()) {
    GTEST_SKIP() << "OpenCL is not supported on this platform";
  }

  const std::vector<uint8_t> model_buf = BuildQuantizedAddModelBuffer(
      tflite::TensorType_INT8, /*in0_zp=*/-10, /*in1_zp=*/12, /*out_zp=*/-5);
  auto tflite_model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(model_buf.data()), model_buf.size());
  ASSERT_NE(tflite_model, nullptr);

  const std::vector<int8_t> in0_vals = {-40, -25, -10, 0, 15, 30, 45, 60};
  const std::vector<int8_t> in1_vals = {-20, -8, 4, 12, 20, 28, 36, 44};

  std::unique_ptr<tflite::Interpreter> cpu_interp;
  ASSERT_EQ(tflite::InterpreterBuilder(
                *tflite_model,
                tflite::ops::builtin::BuiltinOpResolver())(&cpu_interp),
            kTfLiteOk);
  ASSERT_NE(cpu_interp, nullptr);
  ASSERT_EQ(cpu_interp->AllocateTensors(), kTfLiteOk);
  std::memcpy(cpu_interp->typed_input_tensor<int8_t>(0), in0_vals.data(),
              in0_vals.size());
  std::memcpy(cpu_interp->typed_input_tensor<int8_t>(1), in1_vals.data(),
              in1_vals.size());
  ASSERT_EQ(cpu_interp->Invoke(), kTfLiteOk);
  const int8_t* cpu_out_ptr = cpu_interp->typed_output_tensor<int8_t>(0);
  ASSERT_NE(cpu_out_ptr, nullptr);
  const std::vector<int8_t> cpu_out(cpu_out_ptr, cpu_out_ptr + in0_vals.size());

  std::unique_ptr<tflite::Interpreter> interpreter;
  ASSERT_EQ(tflite::InterpreterBuilder(
                *tflite_model,
                tflite::ops::builtin::BuiltinOpResolver())(&interpreter),
            kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->tensors_size(), 3);

  LiteRtEnvironment environment = nullptr;
  ASSERT_EQ(LiteRtCreateEnvironment(0, nullptr, &environment), kLiteRtStatusOk);
  absl::Cleanup env_cleanup = [&]() {
    if (environment != nullptr) {
      LiteRtDestroyEnvironment(environment);
    }
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
  options->serialize_program_cache = false;
  options->runtime_context = LrtGetRuntimeContext();

  auto delegate = CreateMlDriftClDelegate(std::move(options), environment);
  ASSERT_NE(delegate, nullptr);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(std::move(delegate)),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

  // Verify the delegated node registers zero arena temporaries.
  const std::vector<int>& plan = interpreter->execution_plan();
  ASSERT_FALSE(plan.empty());
  for (int node_index : plan) {
    const auto* node_and_reg = interpreter->node_and_registration(node_index);
    ASSERT_NE(node_and_reg, nullptr);
    const TfLiteNode& node = node_and_reg->first;
    EXPECT_TRUE(node.temporaries == nullptr || node.temporaries->size == 0);
  }

  // Verify all synthetic fp32 intermediate tensors remain unallocated in the
  // TFLite arena (`data.raw == nullptr`).
  ASSERT_EQ(interpreter->tensors_size(), 6);
  for (int t_idx = 3; t_idx < interpreter->tensors_size(); ++t_idx) {
    const TfLiteTensor* tensor = interpreter->tensor(t_idx);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->type, kTfLiteFloat32);
    EXPECT_EQ(tensor->data.raw, nullptr) << "tensor index " << t_idx;
    EXPECT_EQ(tensor->bytes, in0_vals.size() * sizeof(float));
  }

  std::memcpy(interpreter->typed_input_tensor<int8_t>(0), in0_vals.data(),
              in0_vals.size());
  std::memcpy(interpreter->typed_input_tensor<int8_t>(1), in1_vals.data(),
              in1_vals.size());
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  // Synthetic fp32 tensors must remain unallocated after Invoke().
  for (int t_idx = 3; t_idx < interpreter->tensors_size(); ++t_idx) {
    EXPECT_EQ(interpreter->tensor(t_idx)->data.raw, nullptr);
  }

  const int8_t* gpu_out = interpreter->typed_output_tensor<int8_t>(0);
  ASSERT_NE(gpu_out, nullptr);
  for (size_t i = 0; i < cpu_out.size(); ++i) {
    EXPECT_LE(
        std::abs(static_cast<int>(gpu_out[i]) - static_cast<int>(cpu_out[i])),
        1);
  }

  // Verify that an unexpected post-init dimension mismatch on a quantized I/O
  // tensor is rejected cleanly instead of overflowing the locked TensorBuffer.
  TfLiteTensor* orig_in0 = interpreter->tensor(0);
  ASSERT_NE(orig_in0, nullptr);
  ASSERT_NE(orig_in0->dims, nullptr);
  ASSERT_EQ(orig_in0->dims->size, 4);
  const int saved_dim = orig_in0->dims->data[3];
  {
    absl::Cleanup restore_dim = [&]() { orig_in0->dims->data[3] = saved_dim; };
    orig_in0->dims->data[3] = saved_dim * 2;
    EXPECT_EQ(interpreter->Invoke(), kTfLiteError);
  }

  // Verify unsupported sub-8-bit quantized types fail cleanly before locking.
  const TfLiteType saved_type = orig_in0->type;
  {
    absl::Cleanup restore_type = [&]() { orig_in0->type = saved_type; };
    orig_in0->type = kTfLiteInt4;
    EXPECT_EQ(interpreter->Invoke(), kTfLiteError);
  }

  // Verify a non-empty quantized tensor with null data fails cleanly instead
  // of silently succeeding with uninitialized GPU buffer contents.
  char* saved_raw = orig_in0->data.raw;
  {
    absl::Cleanup restore_raw = [&]() { orig_in0->data.raw = saved_raw; };
    orig_in0->data.raw = nullptr;
    EXPECT_EQ(interpreter->Invoke(), kTfLiteError);
  }
}

}  // namespace
}  // namespace litert::ml_drift
