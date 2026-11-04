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
#include "ml_drift/cl/environment.h"  // from @ml_drift
#include "ml_drift/cl/opencl_wrapper.h"  // from @ml_drift
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "third_party/odml/infra/ml_drift_delegate/testdata/simple_add.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
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

}  // namespace
}  // namespace litert::ml_drift
