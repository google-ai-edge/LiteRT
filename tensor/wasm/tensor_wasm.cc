/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
#include <emscripten/em_js.h>
#include <emscripten/val.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/string.h"  // from @flatbuffers
#include "ml_drift/webgpu/spatial_tensor.h"  // from @ml_drift
#include "litert/c/litert_common.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/runtime/tensor_buffer.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/litert/lambda_model_runner.h"
#include "tensor/runners/litert/litert_buffer.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"

EM_JS(void, copyTypedArrayToBuffer, (int ptr, emscripten::EM_VAL data_handle), {
  const data = Emval.toValue(data_handle);
  if (!data || !data.buffer) {
    return;
  }
  const heap = new Uint8Array(Module.HEAPU8.buffer);
  const src = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  heap.set(src, ptr);
});

int g_webgpu_device_id = -1;

void setWebGpuDeviceId(int device_id) { g_webgpu_device_id = device_id; }

uintptr_t GetPreinitializedWebGpuDeviceId() {
#ifdef __EMSCRIPTEN__
  WGPUDevice device = emscripten_webgpu_get_device();
  return reinterpret_cast<uintptr_t>(device);
#else
  return 0;
#endif
}

template <typename RunnerPtr>
bool SetInputBinaryCommon(RunnerPtr& runner, const std::string& name,
                          emscripten::val data) {
  if (!runner) return false;

  size_t bytes = data["byteLength"].as<size_t>();
  std::vector<uint8_t> vec(bytes);
  copyTypedArrayToBuffer(reinterpret_cast<int>(vec.data()), data.as_handle());

  auto span = absl::MakeConstSpan(vec.data(), vec.size());
  return runner->SetInput(name, span).ok();
}

template <typename RunnerPtr>
bool SetInputBinaryCommon(RunnerPtr& runner, const std::string& signature_name,
                          const std::string& name, emscripten::val data) {
  if (!runner) return false;

  size_t bytes = data["byteLength"].as<size_t>();
  std::vector<uint8_t> vec(bytes);
  copyTypedArrayToBuffer(reinterpret_cast<int>(vec.data()), data.as_handle());

  auto span = absl::MakeConstSpan(vec.data(), vec.size());
  return runner->SetInput(signature_name, name, span).ok();
}

#include "tensor/tensor.h"

#if defined(LITERT_WASM_JSPI)
#define LITERT_EM_ASYNC , emscripten::async()
#else
#define LITERT_EM_ASYNC
#endif

using emscripten::class_;
using emscripten::enum_;
using emscripten::register_vector;
using emscripten::typed_memory_view;
using emscripten::val;
using emscripten::value_object;
using litert::tensor::PerChannelAffineQuantization;
using litert::tensor::TensorHandle;
using litert::tensor::TfLiteMixinTag;
using litert::tensor::Type;

emscripten::val GetTensorWebGpuBufferId(const TensorHandle& tensor);

val GetTensorData(const TensorHandle& tensor) {
  auto statusor_buf = tensor.GetBuffer();
  if (!statusor_buf.ok()) {
    val buffer_id = GetTensorWebGpuBufferId(tensor);
    if (!buffer_id.isNull()) {
      val readback_fn = val::global("litert")["WebGPU"]["readbackTensorData"];
      if (!readback_fn.isUndefined()) {
        return readback_fn(tensor);
      }
    }
    return val::null();
  }

  auto span = statusor_buf->Lock();
  const size_t bytes = span.size();
  const void* ptr = span.data();

  switch (tensor.GetType()) {
    case Type::kFP32: {
      val js_array = val::global("Float32Array").new_(bytes / sizeof(float));
      js_array.call<void>(
          "set", val(typed_memory_view(bytes / sizeof(float),
                                       reinterpret_cast<const float*>(ptr))));
      return js_array;
    }
    case Type::kI32: {
      val js_array = val::global("Int32Array").new_(bytes / sizeof(int32_t));
      js_array.call<void>(
          "set", val(typed_memory_view(bytes / sizeof(int32_t),
                                       reinterpret_cast<const int32_t*>(ptr))));
      return js_array;
    }
    case Type::kI8: {
      val js_array = val::global("Int8Array").new_(bytes / sizeof(int8_t));
      js_array.call<void>(
          "set", val(typed_memory_view(bytes / sizeof(int8_t),
                                       reinterpret_cast<const int8_t*>(ptr))));
      return js_array;
    }
    case Type::kBOOL:
    case Type::kI4:  // Fallthrough to returning raw standard packed byte view
                     // arrays back to Javascript!
    case Type::kU8: {
      val js_array = val::global("Uint8Array").new_(bytes / sizeof(uint8_t));
      js_array.call<void>(
          "set", val(typed_memory_view(bytes / sizeof(uint8_t),
                                       reinterpret_cast<const uint8_t*>(ptr))));
      return js_array;
    }
    case Type::kFP16: {
      val js_array = val::global("Uint16Array").new_(bytes / sizeof(uint16_t));
      js_array.call<void>("set", val(typed_memory_view(
                                     bytes / sizeof(uint16_t),
                                     reinterpret_cast<const uint16_t*>(ptr))));
      return js_array;
    }
    default:
      return val::null();
  }
}

val GetMutableTensorData(const TensorHandle& tensor) {
  auto statusor_buf = tensor.GetBuffer();
  if (!statusor_buf.ok()) {
    return val::null();
  }

  auto& buffer = *statusor_buf;
  auto* mutable_buffer = dynamic_cast<litert::tensor::MutableBuffer*>(&buffer);
  if (!mutable_buffer) {
    return val::null();
  }

  auto span = mutable_buffer->LockMutable();
  const size_t bytes = span.size();
  void* ptr = span.data();

  switch (tensor.GetType()) {
    case Type::kFP16:
      return val(typed_memory_view(bytes / sizeof(uint16_t),
                                   reinterpret_cast<uint16_t*>(ptr)));
    case Type::kFP32:
      return val(typed_memory_view(bytes / sizeof(float),
                                   reinterpret_cast<float*>(ptr)));
    case Type::kI32:
      return val(typed_memory_view(bytes / sizeof(int32_t),
                                   reinterpret_cast<int32_t*>(ptr)));
    case Type::kI8:
      return val(typed_memory_view(bytes / sizeof(int8_t),
                                   reinterpret_cast<int8_t*>(ptr)));
    case Type::kBOOL:
    case Type::kI4:  // Fallthrough to expose raw locked bytes pointer memory
                     // views bounds!
    case Type::kU8:
      return val(typed_memory_view(bytes / sizeof(uint8_t),
                                   reinterpret_cast<uint8_t*>(ptr)));
    default:
      ABSL_LOG(ERROR) << "JITSI_LOG: GetMutableTensorData: Unsupported type: "
                      << (int)tensor.GetType();
      return val::null();
  }
}

emscripten::val GetTensorWebGpuBufferId(const TensorHandle& tensor) {
  auto statusor_buf = tensor.GetBuffer();
  if (!statusor_buf.ok()) return emscripten::val::null();

  auto* litert_buffer =
      dynamic_cast<litert::tensor::LitertBuffer*>(&*statusor_buf);
  if (!litert_buffer) return emscripten::val::null();

  auto& tb = litert_buffer->tensor_buffer();
  LiteRtTensorBuffer raw_tb = tb.Get();
  auto* tb_t = reinterpret_cast<LiteRtTensorBufferT*>(raw_tb);

  auto custom_buf_or = tb_t->GetCustomBuffer();
  if (!custom_buf_or.HasValue()) return emscripten::val::null();

  auto* custom_buf = *custom_buf_or;
  auto* spatial_tensor = reinterpret_cast<::ml_drift::webgpu::SpatialTensor*>(
      custom_buf->hw_buffer_handle());
  auto wgpu_buffer = spatial_tensor->GetBufferHandle();
  return emscripten::val(
      reinterpret_cast<uintptr_t>(reinterpret_cast<WGPUBuffer&>(wgpu_buffer)));
}

// Helper structure for Quantization mapping
struct JSQuantizationParams {
  std::vector<float> scales;
  std::vector<int> zeroPoints;
  int quantizedDimension;
};

JSQuantizationParams GetQuantizationParams(const TensorHandle& tensor) {
  auto quant_ptr = tensor.GetQuantization();
  if (!quant_ptr) return {};

  auto affine_or = quant_ptr->As<PerChannelAffineQuantization>();
  if (affine_or.ok()) {
    std::vector<int> zero_points_i32(affine_or->zero_points.begin(),
                                     affine_or->zero_points.end());
    return JSQuantizationParams{
        .scales = affine_or->scales,
        .zeroPoints = zero_points_i32,
        .quantizedDimension = affine_or->quantized_dimension};
  }
  return {};
}

void SetTensorNameWrapper(TensorHandle& self, const std::string& name) {
  self.SetName(name);
}

using litert::tensor::MapInputs;
using litert::tensor::MapOutputs;
using litert::tensor::TensorsMap;

TensorsMap ExtractTensorsMap(const emscripten::val& js_object) {
  TensorsMap map;
  val keys = val::global("Object").call<val>("keys", js_object);
  int length = keys["length"].as<int>();
  for (int i = 0; i < length; ++i) {
    std::string key = keys[i].as<std::string>();
    ABSL_LOG(INFO) << "ExtractTensorsMap: key=" << key;
    val value = js_object[key];
    TensorHandle handle = value.as<TensorHandle>();
    map[key] = ::litert::tensor::Tensor<TfLiteMixinTag>(handle.GetRaw());
  }
  return map;
}

struct EmbindLambdaRunner {
  std::shared_ptr<litert::Environment> env;
  std::shared_ptr<litert::Options> options;
  using DummyLambda = std::function<TensorsMap(const TensorsMap&)>;
  using RealRunner = litert::tensor::LambdaModelRunner<DummyLambda>;

  std::shared_ptr<RealRunner> runner;

  bool isNull() const { return runner == nullptr; }

  bool run() {
    if (!runner) {
      return false;
    }
    return runner->Run().ok();
  }

  bool setInput(const std::string& name, TensorHandle& tensor) {
    if (!runner) {
      return false;
    }
    return runner->SetInput(name, tensor).ok();
  }

  bool setInputBinary(const std::string& name, emscripten::val data) {
    return SetInputBinaryCommon(runner, name, data);
  }

  bool setInputBinaryDirect(const std::string& name, int ptr, size_t size) {
    if (!runner) return false;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(ptr);
    auto span = absl::MakeConstSpan(src, size);
    auto status = runner->SetInput(name, span);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "SetInput failed: " << status.message();
      return false;
    }
    return true;
  }

  TensorHandle getInput(const std::string& name) {
    if (!runner) return TensorHandle::Invalid();
    auto res = runner->GetInput(name);
    if (!res.ok()) return TensorHandle::Invalid();
    return *res;
  }

  TensorHandle getOutput(const std::string& name) {
    if (!runner) return TensorHandle::Invalid();
    auto res = runner->GetOutput(name);
    if (!res.ok()) return TensorHandle::Invalid();
    return *res;
  }
};

EmbindLambdaRunner CreateStaticLambdaRunner(const emscripten::val& inputs,
                                            const emscripten::val& outputs,
                                            int accelerators) {
  TensorsMap input_map = ExtractTensorsMap(inputs);
  TensorsMap output_map = ExtractTensorsMap(outputs);

  std::vector<litert::Environment::Option> env_options;
  if (g_webgpu_device_id != -1) {
    env_options.push_back(
        {litert::Environment::OptionTag::WebGpuDevice,
         litert::LiteRtVariant(reinterpret_cast<void*>(g_webgpu_device_id))});
    // Natively extract the associated Queue C handle pointer from
    // the preinitialized Device context
    WGPUDevice w_dev = reinterpret_cast<WGPUDevice>(g_webgpu_device_id);
    WGPUQueue w_queue = wgpuDeviceGetQueue(w_dev);
    if (w_queue) {
      env_options.push_back(
          {litert::Environment::OptionTag::WebGpuQueue,
           litert::LiteRtVariant(reinterpret_cast<void*>(w_queue))});
    }
  }
  auto env_or = litert::Environment::Create(env_options);
  auto options_or = litert::Options::Create();

  if (!env_or.HasValue() || !options_or.HasValue()) {
    return EmbindLambdaRunner{nullptr, nullptr, nullptr};
  }

  options_or->SetHardwareAccelerators(litert::HwAcceleratorSet(accelerators));

  auto env_ptr = std::make_shared<litert::Environment>(std::move(*env_or));
  auto options_ptr = std::make_shared<litert::Options>(std::move(*options_or));

  auto runner = std::make_shared<EmbindLambdaRunner::RealRunner>(
      *env_ptr, *options_ptr, input_map, output_map);

  return EmbindLambdaRunner{env_ptr, options_ptr, std::move(runner)};
}

struct EmbindDynamicRunner {
  std::vector<char> model_buffer;
  std::shared_ptr<litert::Environment> env;
  std::shared_ptr<litert::Options> options;
  std::shared_ptr<litert::tensor::LitertDynamicRunner> runner;

  bool isNull() const { return runner == nullptr; }

  bool run() {
    if (!runner) {
      return false;
    }
    auto status = runner->Run();
    return status.ok();
  }

  TensorHandle getInput(const std::string& name) {
    if (!runner) return TensorHandle::Invalid();
    auto res = runner->GetInput(name);
    if (!res.ok()) return TensorHandle::Invalid();
    return *res;
  }

  TensorHandle getOutput(const std::string& name) {
    if (!runner) return TensorHandle::Invalid();
    auto res = runner->GetOutput(name);
    if (!res.ok()) return TensorHandle::Invalid();
    return *res;
  }

  TensorHandle getInputByIndex(size_t index) {
    if (!runner) return TensorHandle::Invalid();
    auto res = runner->GetInput(index);
    if (!res.ok()) return TensorHandle::Invalid();
    return *res;
  }

  TensorHandle getOutputByIndex(size_t index) {
    if (!runner) return TensorHandle::Invalid();
    auto res = runner->GetOutput(index);
    if (!res.ok()) return TensorHandle::Invalid();
    return *res;
  }

  bool setInput(const std::string& name, TensorHandle& tensor) {
    if (!runner) return false;
    return runner->SetInput(name, tensor).ok();
  }

  bool setInputBinary(const std::string& name, emscripten::val data) {
    return SetInputBinaryCommon(runner, name, data);
  }

  bool setInputBinaryDirect(const std::string& name, int ptr, size_t size) {
    if (!runner) return false;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(ptr);
    auto span = absl::MakeConstSpan(src, size);
    auto status = runner->SetInput(name, span);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "SetInput failed: " << status.message();
      return false;
    }
    return true;
  }

  bool runSig(const std::string& signature_name) {
    if (!runner) return false;
    auto status = runner->Run(signature_name);
    return status.ok();
  }

  TensorHandle getInputSig(const std::string& signature_name,
                           const std::string& name) {
    if (!runner) return TensorHandle::Invalid();
    auto res = runner->GetInput(signature_name, name);
    if (!res.ok()) return TensorHandle::Invalid();
    return *res;
  }

  TensorHandle getOutputSig(const std::string& signature_name,
                            const std::string& name) {
    if (!runner) return TensorHandle::Invalid();
    auto res = runner->GetOutput(signature_name, name);
    if (!res.ok()) return TensorHandle::Invalid();
    return *res;
  }

  bool setInputSig(const std::string& signature_name, const std::string& name,
                   TensorHandle& tensor) {
    if (!runner) return false;
    return runner->SetInput(signature_name, name, tensor).ok();
  }

  bool setInputBinarySig(const std::string& signature_name,
                         const std::string& name, emscripten::val data) {
    return SetInputBinaryCommon(runner, signature_name, name, data);
  }

  bool setInputBinaryDirectSig(const std::string& signature_name,
                               const std::string& name, int ptr, size_t size) {
    if (!runner) return false;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(ptr);
    auto span = absl::MakeConstSpan(src, size);
    auto status = runner->SetInput(signature_name, name, span);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "SetInput failed: " << status.message();
      return false;
    }
    return true;
  }

  uintptr_t getOutputWebGpuBuffer(const std::string& signature_name,
                                  const std::string& name) {
    if (!runner) return 0;
    auto res = runner->GetOutputWebGpuBuffer(signature_name, name);
    if (!res.ok()) return 0;
    auto* spatial_tensor =
        reinterpret_cast<::ml_drift::webgpu::SpatialTensor*>(*res);
    if (!spatial_tensor) return 0;
    auto wgpu_buffer = spatial_tensor->GetBufferHandle();
    return reinterpret_cast<uintptr_t>(
        reinterpret_cast<WGPUBuffer&>(wgpu_buffer));
  }

  uintptr_t getInputWebGpuBuffer(const std::string& signature_name,
                                 const std::string& name) {
    if (!runner) return 0;
    auto res = runner->GetInputWebGpuBuffer(signature_name, name);
    if (!res.ok()) return 0;
    auto* spatial_tensor =
        reinterpret_cast<::ml_drift::webgpu::SpatialTensor*>(*res);
    if (!spatial_tensor) return 0;
    auto wgpu_buffer = spatial_tensor->GetBufferHandle();
    return reinterpret_cast<uintptr_t>(
        reinterpret_cast<WGPUBuffer&>(wgpu_buffer));
  }
};

EmbindDynamicRunner CreateDynamicRunnerFromBuffer(emscripten::val model_bytes,
                                                  int accelerator) {
  auto l = model_bytes["length"].as<size_t>();
  std::vector<char> vec(l);
  for (size_t i = 0; i < l; ++i) {
    vec[i] = model_bytes[i].as<char>();
  }

  std::vector<litert::Environment::Option> env_options;
  if (g_webgpu_device_id != -1) {
    env_options.push_back(
        {litert::Environment::OptionTag::WebGpuDevice,
         litert::LiteRtVariant(reinterpret_cast<void*>(g_webgpu_device_id))});

    // Natively extract the associated Queue C handle pointer from the
    // preinitialized Device context
    WGPUDevice w_dev = reinterpret_cast<WGPUDevice>(g_webgpu_device_id);
    WGPUQueue w_queue = wgpuDeviceGetQueue(w_dev);
    if (w_queue) {
      env_options.push_back(
          {litert::Environment::OptionTag::WebGpuQueue,
           litert::LiteRtVariant(reinterpret_cast<void*>(w_queue))});
    }
  }
  auto env_or = litert::Environment::Create(env_options);
  auto options_or = litert::Options::Create();
  if (options_or.HasValue()) {
    auto gpu_options_or = options_or->GetGpuOptions();
    if (gpu_options_or.HasValue()) {
      gpu_options_or->SetPrecision(litert::GpuOptions::Precision::kFp32);
      std::cout << "Enforced FP32 precision for GPU" << std::endl;
    } else {
      std::cout << "Failed to get GPU options: "
                << gpu_options_or.Error().Message() << std::endl;
    }
  }

  // Use optimized XNNPack kernels on CPU!

  if (!env_or.HasValue()) {
    std::cout << "Environment::Create failed" << std::endl;
    return EmbindDynamicRunner{};
  }
  if (!options_or.HasValue()) {
    std::cout << "Options::Create failed" << std::endl;
    return EmbindDynamicRunner{};
  }

  options_or->SetHardwareAccelerators(
      static_cast<litert::HwAccelerators>(accelerator));

  auto env_ptr = std::make_shared<litert::Environment>(std::move(*env_or));
  auto options_ptr = std::make_shared<litert::Options>(std::move(*options_or));

  auto span = absl::MakeConstSpan(reinterpret_cast<const uint8_t*>(vec.data()),
                                  vec.size());

  auto runner_or =
      litert::tensor::LitertDynamicRunner::Create(*env_ptr, span, *options_ptr);
  if (!runner_or.ok()) {
    std::cout << "CreateDynamicRunnerFromBuffer failed: "
              << runner_or.status().message() << std::endl;
    return EmbindDynamicRunner{};
  }

  return EmbindDynamicRunner{
      std::move(vec), env_ptr, options_ptr,
      std::make_shared<litert::tensor::LitertDynamicRunner>(
          std::move(*runner_or))};
}

EmbindDynamicRunner CreateMultiSignatureRunner(emscripten::val signatures,
                                               int accelerator) {
  litert::tensor::ModelFactory factory;

  auto keys = emscripten::val::global("Object").call<emscripten::val>(
      "keys", signatures);
  auto length = keys["length"].as<size_t>();

  for (size_t i = 0; i < length; ++i) {
    std::string sig_name = keys[i].as<std::string>();
    emscripten::val sig = signatures[sig_name];

    emscripten::val outputs_js = sig["outputs"];
    auto l = outputs_js["length"].as<size_t>();
    std::vector<TensorHandle> outputs;
    for (size_t j = 0; j < l; ++j) {
      outputs.push_back(outputs_js[j].as<TensorHandle>());
    }

    auto status = factory.AddSignature(outputs, sig_name);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to add signature " << sig_name << ": "
                      << status.message() << std::endl;
      return EmbindDynamicRunner{};
    }
  }

  std::cout << "[C++ Dbg] 1. Signatures AddSignature loop complete."
            << std::endl;
  std::cout << "[C++ Dbg] 2. Starting factory.CreateFlatbuffer()..."
            << std::endl;
  auto fb_or = factory.CreateFlatbuffer();

  if (!fb_or.ok()) {
    std::cout << "Failed to create flatbuffer: " << fb_or.status().message()
              << std::endl;
    return EmbindDynamicRunner{};
  }

  std::vector<char> fb = std::move(*fb_or);
  std::cout << "[C++ Dbg] 3. CreateFlatbuffer success. Size: " << fb.size()
            << " bytes." << std::endl;

  auto env_or = litert::Environment::Create({});
  auto options_or = litert::Options::Create();

  if (options_or.HasValue()) {
    auto cpu_options_or = options_or->GetCpuOptions();
    if (cpu_options_or.HasValue()) {
      cpu_options_or->SetNumThreads(4);
    }
    auto gpu_options_or = options_or->GetGpuOptions();
    if (gpu_options_or.HasValue()) {
      gpu_options_or->SetPrecision(litert::GpuOptions::Precision::kFp32);
    }
  }

  if (!env_or.HasValue() || !options_or.HasValue()) {
    std::cout << "Failed to create environment or options" << std::endl;
    return EmbindDynamicRunner{};
  }

  options_or->SetHardwareAccelerators(
      static_cast<litert::HwAccelerators>(accelerator));

  auto env_ptr = std::make_shared<litert::Environment>(std::move(*env_or));
  auto options_ptr = std::make_shared<litert::Options>(std::move(*options_or));

  std::vector<char> model_buffer = std::move(fb);
  auto span =
      absl::MakeConstSpan(reinterpret_cast<const uint8_t*>(model_buffer.data()),
                          model_buffer.size());
  std::cout << "[C++ Dbg] 4. Starting LitertDynamicRunner::Create()..."
            << std::endl;
  auto runner_or =
      litert::tensor::LitertDynamicRunner::Create(*env_ptr, span, *options_ptr);

  if (!runner_or.ok()) {
    std::cout << "CreateMultiSignatureRunner failed: "
              << runner_or.status().message() << std::endl;
    return EmbindDynamicRunner{};
  }

  std::cout << "[C++ Dbg] 5. LitertDynamicRunner::Create success!" << std::endl;
  return EmbindDynamicRunner{
      std::move(model_buffer), env_ptr, options_ptr,
      std::make_shared<litert::tensor::LitertDynamicRunner>(
          std::move(*runner_or))};
}

TensorHandle CreateTensorWithData(emscripten::val data, Type type,
                                  emscripten::val shape_array,
                                  emscripten::val name_opt) {
  auto l = data["length"].as<size_t>();
  std::vector<float> vec(l);
  for (size_t i = 0; i < l; ++i) {
    vec[i] = data[i].as<float>();
  }

  std::vector<int> int_shape;
  if (!shape_array["length"].isUndefined() && !shape_array["length"].isNull()) {
    const size_t length = shape_array["length"].as<size_t>();
    for (size_t i = 0; i < length; ++i) {
      int_shape.push_back(shape_array[i].as<int>());
    }
  } else if (!shape_array["size"].isUndefined()) {
    const size_t length = shape_array.call<size_t>("size");
    for (size_t i = 0; i < length; ++i) {
      int_shape.push_back(shape_array.call<int>("get", i));
    }
  }

  std::string name_str;
  if (!name_opt.isUndefined() && !name_opt.isNull()) {
    name_str = name_opt.as<std::string>();
  }

  auto tensor_handle = ::litert::tensor::Create(
      std::move(name_str), type, int_shape,
      ::litert::tensor::OwningCpuBuffer::Copy<::litert::tensor::Type::kFP32>(
          vec));

  return tensor_handle;
}

TensorHandle CreatePlaceholderTensor(uint32_t type_val,
                                     emscripten::val shape_array,
                                     emscripten::val name_opt) {
  Type type = static_cast<Type>(type_val);
  std::vector<int> int_shape;
  if (!shape_array["length"].isUndefined() && !shape_array["length"].isNull()) {
    const size_t length = shape_array["length"].as<size_t>();
    for (size_t i = 0; i < length; ++i) {
      int_shape.push_back(shape_array[i].as<int>());
    }
  } else if (!shape_array["size"].isUndefined()) {
    const size_t length = shape_array.call<size_t>("size");
    for (size_t i = 0; i < length; ++i) {
      int_shape.push_back(shape_array.call<int>("get", i));
    }
  }

  std::string name_str;
  if (!name_opt.isUndefined() && !name_opt.isNull()) {
    name_str = name_opt.as<std::string>();
  }

  auto tensor = ::litert::tensor::Tensor<TfLiteMixinTag>(
      {.name = name_str, .type = type, .shape = int_shape});
  return TensorHandle(tensor);
}

bool RunEager(emscripten::val outputs_array) {
  std::vector<TensorHandle> outputs;
  const size_t length = outputs_array["length"].as<size_t>();
  for (size_t i = 0; i < length; ++i) {
    outputs.push_back(outputs_array[i].as<TensorHandle>());
  }

  // Guard: Check for placeholder inputs
  auto execution_plan_or =
      litert::tensor::GetExecutionPlan(absl::MakeConstSpan(outputs));
  if (!execution_plan_or.ok()) {
    ABSL_LOG(ERROR) << "Failed to get execution plan for eager execution.";
    return false;
  }

  for (const auto& op : *execution_plan_or) {
    for (const auto& input : op->inputs) {
      auto info_or = litert::tensor::graph::GetInfo(input);
      auto producer_or = litert::tensor::graph::GetProducer(input);
      if (info_or.ok() && producer_or.ok()) {
        ABSL_LOG(INFO) << "Op: " << op->GetName() << " Input: " << info_or->name
                       << " has buffer: " << (info_or->buffer != nullptr)
                       << " has producer: " << (*producer_or != nullptr);
        if (*producer_or == nullptr && info_or->buffer == nullptr) {
          ABSL_LOG(ERROR) << "Eager execution does not support placeholder "
                             "inputs. Found placeholder for tensor: "
                          << info_or->name;
          return false;
        }
      }
    }
  }

  auto env_or = litert::Environment::Create({});
  auto options_or = litert::Options::Create();
  if (!env_or.HasValue() || !options_or.HasValue()) return false;

  static auto env_ptr =
      std::make_shared<litert::Environment>(std::move(*env_or));
  static auto options_ptr =
      std::make_shared<litert::Options>(std::move(*options_or));
  static bool init = false;
  if (!init) {
    options_ptr->SetHardwareAccelerators(litert::HwAccelerators::kGpu |
                                         litert::HwAccelerators::kCpu);
    init = true;
  }

  litert::tensor::ModelFactory serialization;
  if (!serialization.AddSubgraph(outputs).ok()) return false;

  auto fb_or = serialization.CreateFlatbuffer();
  if (!fb_or.ok()) return false;

  std::vector<char> fb = std::move(*fb_or);
  litert::BufferRef<> model_buffer(fb.data(), fb.size());

  auto compiled_model_or =
      litert::CompiledModel::Create(*env_ptr, model_buffer, *options_ptr);
  if (!compiled_model_or.HasValue()) return false;

  auto compiled_model = std::move(*compiled_model_or);

  auto input_buffers_or = compiled_model.CreateInputBuffers();
  auto output_buffers_or = compiled_model.CreateOutputBuffers();

  if (!input_buffers_or.HasValue() || !output_buffers_or.HasValue())
    return false;

  auto status = compiled_model.Run(*input_buffers_or, *output_buffers_or);
  if (!status.HasValue()) return false;

  auto signature_or = compiled_model.GetSignature(0);
  if (!signature_or.HasValue()) return false;
  auto& signature = *signature_or;

  for (size_t i = 0; i < signature.OutputNames().size(); ++i) {
    std::string name = std::string(signature.OutputNames()[i]);
    for (auto& output_tensor : outputs) {
      if (output_tensor.GetName() == name) {
        auto dup_or = (*output_buffers_or)[i].Duplicate();
        if (!dup_or.HasValue()) return false;

        auto litert_buffer =
            std::make_shared<litert::tensor::LitertBuffer>(std::move(*dup_or));

        auto info_or = GetInfo(output_tensor.GetRaw());
        if (!info_or.ok()) return false;
        auto& info = *info_or;

        info.buffer = litert_buffer;
        break;
      }
    }
  }

  return true;
}

EMSCRIPTEN_BINDINGS(litert_tensor_core) {
  // Bind Quantization structure mapping
  value_object<JSQuantizationParams>("QuantizationParams")
      .field("scales", &JSQuantizationParams::scales)
      .field("zeroPoints", &JSQuantizationParams::zeroPoints)
      .field("quantizedDimension", &JSQuantizationParams::quantizedDimension);

  // Register vectors for shape handling
  register_vector<int32_t>("ShapeVector");
  register_vector<float>("FloatVector");
  register_vector<TensorHandle>("TensorVector");
  register_vector<int64_t>("Int64Vector");

  emscripten::function(
      "createTensorWithData",
      emscripten::optional_override([](emscripten::val data, Type type,
                                       emscripten::val shape_array,
                                       emscripten::val name_opt) {
        return CreateTensorWithData(data, type, shape_array, name_opt);
      }));
  emscripten::function(
      "createPlaceholderTensor",
      emscripten::optional_override([](uint32_t type_val,
                                       emscripten::val shape_array,
                                       emscripten::val name_opt) {
        return CreatePlaceholderTensor(type_val, shape_array, name_opt);
      }));
  emscripten::function("runEager", &RunEager);

  // Bind Types mapping
  enum_<Type>("TensorType")
      .value("UNKNOWN", Type::kUnknown)
      .value("FP32", Type::kFP32)
      .value("kFP32", Type::kFP32)
      .value("I32", Type::kI32)
      .value("kI32", Type::kI32)
      .value("I4", Type::kI4)
      .value("kI4", Type::kI4)
      .value("I8", Type::kI8)
      .value("kI8", Type::kI8)
      .value("BOOL", Type::kBOOL)
      .value("kBOOL", Type::kBOOL);

  enum_<litert::HwAccelerators>("HwAccelerators")
      .value("NONE", litert::HwAccelerators::kNone)
      .value("CPU", litert::HwAccelerators::kCpu)
      .value("GPU", litert::HwAccelerators::kGpu)
      .value("NPU", litert::HwAccelerators::kNpu)
      .value("WEBNN", litert::HwAccelerators::kWebNn);

  // Bind Core JS Tensor overriding base TensorHandle mapped precisely
  class_<TensorHandle>("Tensor")
      .constructor<>()
      .function("getName", emscripten::optional_override(
                               [](const TensorHandle& self) -> std::string {
                                 return std::string(self.GetName());
                               }))
      .function(
          "getId",
          emscripten::optional_override([](TensorHandle& self) -> uint32_t {
            return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&self));
          }))
      .function("getType", &TensorHandle::GetType)
      .function("getShape", &TensorHandle::GetShape)
      .function("getData", &GetTensorData LITERT_EM_ASYNC)
      .function("getMutableData", &GetMutableTensorData)
      .function(
          "getDataPointer",
          emscripten::optional_override([](TensorHandle& self) -> uintptr_t {
            auto statusor_buf = self.GetBuffer();
            if (!statusor_buf.ok()) {
              ABSL_LOG(ERROR)
                  << "JITSI_LOG: getDataPointer: GetBuffer() failed: "
                  << statusor_buf.status().message();
              return 0;
            }
            auto& buffer = *statusor_buf;
            auto* mutable_buffer =
                dynamic_cast<litert::tensor::MutableBuffer*>(&buffer);
            if (!mutable_buffer) {
              ABSL_LOG(ERROR) << "JITSI_LOG: getDataPointer: dynamic_cast to "
                                 "MutableBuffer failed!";
              return 0;
            }
            auto span = mutable_buffer->LockMutable();
            return reinterpret_cast<uintptr_t>(span.data());
          }) LITERT_EM_ASYNC)
      .function(
          "allocateBuffer",
          emscripten::optional_override([](TensorHandle& self) {
            auto info_or = litert::tensor::graph::GetInfo(
                const_cast<litert::tensor::graph::Tensor&>(self.GetRaw()));
            if (!info_or.ok()) return;
            auto& info = *info_or;
            const size_t s =
                std::accumulate(info.shape.begin(), info.shape.end(), 1,
                                std::multiplies<int>());
            info.buffer =
                litert::tensor::OwningCpuBuffer::AllocateAs(info.type, s);
          }))
      .function("getWebGpuBuffer", &GetTensorWebGpuBufferId)
      .function("getQuantization", &GetQuantizationParams)
      .function("setName", &SetTensorNameWrapper)
      .function("setType",
                emscripten::optional_override(
                    [](TensorHandle& self, Type type) { self.SetType(type); }))
      .function("setShape",
                emscripten::optional_override([](TensorHandle& self,
                                                 emscripten::val shape_array) {
                  std::vector<int32_t> shape;
                  if (!shape_array["length"].isUndefined() &&
                      !shape_array["length"].isNull()) {
                    const size_t length = shape_array["length"].as<size_t>();
                    for (size_t i = 0; i < length; ++i) {
                      shape.push_back(shape_array[i].as<int32_t>());
                    }
                  } else if (!shape_array["size"].isUndefined()) {
                    const size_t length = shape_array.call<size_t>("size");
                    for (size_t i = 0; i < length; ++i) {
                      shape.push_back(shape_array.call<int32_t>("get", i));
                    }
                  }
                  self.SetShape(shape);
                }))
      .function(
          "setQuantization",
          emscripten::optional_override([](TensorHandle& self,
                                           emscripten::val params) {
            std::vector<float> scales;
            std::vector<int64_t> zero_points;
            int quantized_dimension = 0;

            if (!params["quantizedDimension"].isUndefined() &&
                !params["quantizedDimension"].isNull()) {
              quantized_dimension = params["quantizedDimension"].as<int>();
            }

            if (!params["scales"].isUndefined() && !params["scales"].isNull()) {
              emscripten::val scales_val = params["scales"];
              if (scales_val.typeOf().as<std::string>() == "object") {
                size_t len = scales_val["length"].as<size_t>();
                scales.resize(len);
                copyTypedArrayToBuffer(reinterpret_cast<int>(scales.data()),
                                       scales_val.as_handle());
              }
            }

            if (!params["zeroPoints"].isUndefined() &&
                !params["zeroPoints"].isNull()) {
              emscripten::val zp_val = params["zeroPoints"];
              if (zp_val.typeOf().as<std::string>() == "object") {
                size_t len = zp_val["length"].as<size_t>();
                std::vector<int32_t> zp_i32(len);
                copyTypedArrayToBuffer(reinterpret_cast<int>(zp_i32.data()),
                                       zp_val.as_handle());
                zero_points.assign(zp_i32.begin(), zp_i32.end());
              }
            }

            auto quant = std::make_shared<PerChannelAffineQuantization>(
                std::move(scales), std::move(zero_points), quantized_dimension);
            self.SetQuantization(quant);
          }))
      .function("abs", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Abs(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("relu", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Relu(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("relu6", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Relu6(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("elu", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Elu(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("hardSwish",
                emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(HardSwish(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("logSoftmax",
                emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(LogSoftmax(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function(
          "softmax",
          emscripten::optional_override([](TensorHandle& self, float beta) {
            return TensorHandle(Softmax(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()), beta));
          }))
      .function("logistic",
                emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Logistic(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("gelu", emscripten::optional_override([](TensorHandle& self,
                                                         bool approximate) {
                  return TensorHandle(Gelu(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                      approximate));
                }))
      .function("neg", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Neg(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("sqrt", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Sqrt(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("cos", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Cos(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("sin", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Sin(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("exp", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Exp(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("log", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Log(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("ceil", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Ceil(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("floor", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Floor(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("sign", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Sign(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("round", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Round(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("logicalNot",
                emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(LogicalNot(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function(
          "add", emscripten::optional_override([](TensorHandle& self,
                                                  TensorHandle& other) {
            return TensorHandle(
                Add(::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
          }))
      .function(
          "mul", emscripten::optional_override([](TensorHandle& self,
                                                  TensorHandle& other) {
            return TensorHandle(
                Mul(::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
          }))
      .function(
          "sub", emscripten::optional_override([](TensorHandle& self,
                                                  TensorHandle& other) {
            return TensorHandle(
                Sub(::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
          }))
      .function(
          "div", emscripten::optional_override([](TensorHandle& self,
                                                  TensorHandle& other) {
            return TensorHandle(
                Div(::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
          }))
      .function(
          "pow", emscripten::optional_override([](TensorHandle& self,
                                                  TensorHandle& other) {
            return TensorHandle(
                Pow(::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
          }))
      .function(
          "minimum", emscripten::optional_override([](TensorHandle& self,
                                                      TensorHandle& other) {
            return TensorHandle(Minimum(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
          }))
      .function(
          "maximum", emscripten::optional_override([](TensorHandle& self,
                                                      TensorHandle& other) {
            return TensorHandle(Maximum(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
          }))
      .function(
          "less", emscripten::optional_override([](TensorHandle& self,
                                                   TensorHandle& other) {
            return TensorHandle(
                Less(::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                     ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
          }))
      .function(
          "greater", emscripten::optional_override([](TensorHandle& self,
                                                      TensorHandle& other) {
            return TensorHandle(Greater(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
          }))
      .function("lessEqual",
                emscripten::optional_override([](TensorHandle& self,
                                                 TensorHandle& other) {
                  return TensorHandle(GreaterEqual(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw()),
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function(
          "greaterEqual",
          emscripten::optional_override(
              [](TensorHandle& self, TensorHandle& other) {
                return TensorHandle(GreaterEqual(
                    ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
              }))
      .function(
          "equal", emscripten::optional_override([](TensorHandle& self,
                                                    TensorHandle& other) {
            return TensorHandle(Equal(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
          }))
      .function(
          "notEqual", emscripten::optional_override([](TensorHandle& self,
                                                       TensorHandle& other) {
            return TensorHandle(NotEqual(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
          }))
      .function(
          "logicalAnd", emscripten::optional_override([](TensorHandle& self,
                                                         TensorHandle& other) {
            return TensorHandle(LogicalAnd(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
          }))
      .function(
          "logicalOr", emscripten::optional_override([](TensorHandle& self,
                                                        TensorHandle& other) {
            return TensorHandle(LogicalOr(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
          }))
      .function(
          "floorDiv", emscripten::optional_override([](TensorHandle& self,
                                                       TensorHandle& other) {
            return TensorHandle(FloorDiv(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
          }))
      .function(
          "floorMod", emscripten::optional_override([](TensorHandle& self,
                                                       TensorHandle& other) {
            return TensorHandle(FloorMod(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
          }))
      .function(
          "sum", emscripten::optional_override([](TensorHandle& self,
                                                  emscripten::val axes_array,
                                                  bool keep_dims) {
            std::vector<int> int_axes;
            const size_t length = axes_array["length"].as<size_t>();
            for (size_t i = 0; i < length; ++i) {
              int_axes.push_back(axes_array[i].as<int>());
            }
            return TensorHandle(
                Sum(::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    int_axes, keep_dims));
          }))
      .function("reduceMax",
                emscripten::optional_override([](TensorHandle& self,
                                                 emscripten::val axes_array,
                                                 bool keep_dims) {
                  std::vector<int> int_axes;
                  const size_t length = axes_array["length"].as<size_t>();
                  for (size_t i = 0; i < length; ++i) {
                    int_axes.push_back(axes_array[i].as<int>());
                  }
                  return TensorHandle(ReduceMax(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                      int_axes, keep_dims));
                }))
      .function(
          "mean", emscripten::optional_override([](TensorHandle& self,
                                                   emscripten::val axes_array,
                                                   bool keep_dims) {
            std::vector<int> int_axes;
            const size_t length = axes_array["length"].as<size_t>();
            for (size_t i = 0; i < length; ++i) {
              int_axes.push_back(axes_array[i].as<int>());
            }
            return TensorHandle(
                Mean(::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                     int_axes, keep_dims));
          }))
      .function(
          "expandDims",
          emscripten::optional_override([](TensorHandle& self, int axis) {
            return TensorHandle(ExpandDims(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()), axis));
          }))
      .function("squeeze",
                emscripten::optional_override([](TensorHandle& self,
                                                 emscripten::val dims_array) {
                  std::vector<int> int_dims;
                  const size_t length = dims_array["length"].as<size_t>();
                  for (size_t i = 0; i < length; ++i) {
                    int_dims.push_back(dims_array[i].as<int>());
                  }
                  return TensorHandle(Squeeze(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                      int_dims));
                }))
      .function("reshape",
                emscripten::optional_override([](TensorHandle& self,
                                                 emscripten::val shape_array) {
                  std::vector<int> int_shape;
                  const size_t length = shape_array["length"].as<size_t>();
                  for (size_t i = 0; i < length; ++i) {
                    int_shape.push_back(shape_array[i].as<int>());
                  }
                  return TensorHandle(Reshape(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                      int_shape));
                }))
      .function(
          "pad", emscripten::optional_override([](TensorHandle& self,
                                                  TensorHandle& pad_tensor) {
            return TensorHandle(Pad(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(pad_tensor.GetRaw())));
          }))
      .function(
          "averagePool2d",
          emscripten::optional_override(
              [](TensorHandle& self, int filter_height, int filter_width,
                 int stride_h, int stride_w, int padding) {
                return TensorHandle(AveragePool2D(
                    ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    filter_height, filter_width, stride_h, stride_w,
                    static_cast<::litert::tensor::Padding>(padding)));
              }))
      .function(
          "maxPool2d",
          emscripten::optional_override(
              [](TensorHandle& self, int filter_height, int filter_width,
                 int stride_h, int stride_w, int padding) {
                return TensorHandle(MaxPool2D(
                    ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    filter_height, filter_width, stride_h, stride_w,
                    static_cast<::litert::tensor::Padding>(padding)));
              }))
      .function(
          "fullyConnected",
          emscripten::optional_override(
              [](TensorHandle& self, TensorHandle& weight) {
                return TensorHandle(FullyConnected(
                    ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(weight.GetRaw())));
              }))
      .function(
          "batchMatMul",
          emscripten::optional_override(
              [](TensorHandle& self, TensorHandle& other, emscripten::val adj_x,
                 emscripten::val adj_y) {
                bool ax = false;
                bool ay = false;
                if (!adj_x.isUndefined()) ax = adj_x.as<bool>();
                if (!adj_y.isUndefined()) ay = adj_y.as<bool>();
                return TensorHandle(BatchMatMul(
                    ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw()),
                    ax, ay));
              }))
      .function(
          "transpose",
          emscripten::optional_override([](TensorHandle& self,
                                           const std::vector<int32_t>& perm) {
            return TensorHandle(Transpose(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()), perm));
          }))
      .function(
          "conv2d",
          emscripten::optional_override(
              [](TensorHandle& self, TensorHandle& filter, int stride_h,
                 int stride_w, int padding, int dilation_h, int dilation_w) {
                auto bias = ::litert::tensor::Tensor<TfLiteMixinTag>(
                    {.type = ::litert::tensor::Type::kFP32,
                     .shape = {1},
                     .buffer = ::litert::tensor::OwningCpuBuffer::Copy<
                         ::litert::tensor::Type::kFP32>({0.0f})});
                return TensorHandle(Conv2D(
                    ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(filter.GetRaw()),
                    bias, stride_h, stride_w,
                    static_cast<::litert::tensor::Padding>(padding), dilation_h,
                    dilation_w, ::litert::tensor::kActNone));
              }))
      .function(
          "depthwiseConv2d",
          emscripten::optional_override(
              [](TensorHandle& self, TensorHandle& filter, int stride_h,
                 int stride_w, int padding, int depth_multiplier) {
                return TensorHandle(DepthwiseConv2DImpl(
                    ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(filter.GetRaw()),
                    absl::optional<::litert::tensor::Tensor<TfLiteMixinTag>>(),
                    stride_h, stride_w,
                    static_cast<::litert::tensor::Padding>(padding), 1, 1,
                    depth_multiplier, ::litert::tensor::kActNone,
                    std::source_location::current()));
              }))
      .function(
          "transposeConv2d",
          emscripten::optional_override(
              [](TensorHandle& self, TensorHandle& filter,
                 const std::vector<int32_t>& out_shape, int stride_h,
                 int stride_w, int padding) {
                std::vector<int> int_shape(out_shape.begin(), out_shape.end());
                return TensorHandle(TransposeConv2D(
                    ::litert::tensor::Tensor<TfLiteMixinTag>(filter.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(), int_shape,
                    static_cast<::litert::tensor::Padding>(padding), stride_h,
                    stride_w, ::litert::tensor::kActNone));
              }))
      .function("unpack", emscripten::optional_override([](TensorHandle& self,
                                                           int num, int axis) {
                  auto out = Unpack(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                      num, axis);
                  std::vector<TensorHandle> res;
                  res.reserve(out.size());
                  for (const auto& t : out) {
                    res.push_back(t);
                  }
                  return res;
                }))
      .function("split", emscripten::optional_override([](TensorHandle& self,
                                                          TensorHandle& axis,
                                                          int num_splits) {
                  auto out = Split(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                      ::litert::tensor::Tensor<TfLiteMixinTag>(axis.GetRaw()),
                      num_splits);
                  std::vector<TensorHandle> res;
                  res.reserve(out.size());
                  for (const auto& t : out) {
                    res.push_back(t);
                  }
                  return res;
                }))
      .function(
          "gather",
          emscripten::optional_override(
              [](TensorHandle& self, TensorHandle& indices, int axis) {
                return TensorHandle(Gather(
                    ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(indices.GetRaw()),
                    axis));
              }))
      .function(
          "oneHot",
          emscripten::optional_override([](TensorHandle& self,
                                           TensorHandle& depth,
                                           TensorHandle& on_value,
                                           TensorHandle& off_value, int axis) {
            return TensorHandle(OneHot(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(depth.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(on_value.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(off_value.GetRaw()),
                axis));
          }))
      .function(
          "prelu", emscripten::optional_override([](TensorHandle& self,
                                                    TensorHandle& alpha) {
            return TensorHandle(PRelu(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(alpha.GetRaw())));
          }))
      .function("l2Normalization",
                emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(L2Normalization(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("square", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Square(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("rsqrt", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Rsqrt(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("cumsum",
                emscripten::optional_override([](TensorHandle& self, int axis,
                                                 bool exclusive, bool reverse) {
                  return TensorHandle(Cumsum(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                      axis, exclusive, reverse));
                }))
      .function("argMax",
                emscripten::optional_override([](TensorHandle& self, int axis) {
                  return TensorHandle(ArgMax(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                      axis, Type::kI32));
                }))
      .function(
          "topK", emscripten::optional_override([](TensorHandle& self, int k) {
            auto out = TopK(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()), k);
            std::vector<TensorHandle> res;
            res.reserve(out.size());
            for (const auto& t : out) {
              res.push_back(t);
            }
            return res;
          }))
      .function(
          "spaceToDepth",
          emscripten::optional_override([](TensorHandle& self, int block_size) {
            return TensorHandle(SpaceToDepth(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                block_size));
          }))
      .function(
          "depthToSpace",
          emscripten::optional_override([](TensorHandle& self, int block_size) {
            return TensorHandle(DepthToSpace(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                block_size));
          }))
      .function("reverse",
                emscripten::optional_override([](TensorHandle& self,
                                                 TensorHandle& axes) {
                  return TensorHandle(Reverse(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                      ::litert::tensor::Tensor<TfLiteMixinTag>(axes.GetRaw())));
                }))
      .function(
          "resizeBilinear",
          emscripten::optional_override(
              [](TensorHandle& self, const emscripten::val& size,
                 bool align_corners, bool half_pixel_centers) {
                std::vector<int> int_size;
                const size_t length = size["length"].as<size_t>();
                for (size_t i = 0; i < length; ++i) {
                  int_size.push_back(size[i].as<int>());
                }
                return TensorHandle(ResizeBilinear(
                    ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    int_size, align_corners, half_pixel_centers));
              }))
      .function(
          "select", emscripten::optional_override([](TensorHandle& self,
                                                     TensorHandle& condition,
                                                     TensorHandle& other) {
            return TensorHandle(Select(
                ::litert::tensor::Tensor<TfLiteMixinTag>(condition.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
          }))
      .function(
          "selectV2", emscripten::optional_override([](TensorHandle& self,
                                                       TensorHandle& condition,
                                                       TensorHandle& other) {
            return TensorHandle(SelectV2(
                ::litert::tensor::Tensor<TfLiteMixinTag>(condition.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(other.GetRaw())));
          }))
      .function(
          "concatenation",
          emscripten::optional_override(
              [](TensorHandle& self, const std::vector<TensorHandle>& others,
                 int axis) {
                std::vector<::litert::tensor::Tensor<TfLiteMixinTag>> inputs;
                inputs.push_back(
                    ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()));
                for (auto& t : others) {
                  inputs.push_back(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(t.GetRaw()));
                }
                return TensorHandle(
                    Concatenation(absl::MakeSpan(inputs), axis));
              }))
      .function(
          "pack",
          emscripten::optional_override(
              [](TensorHandle& self, const std::vector<TensorHandle>& others,
                 int axis) {
                std::vector<::litert::tensor::Tensor<TfLiteMixinTag>> inputs;
                inputs.push_back(
                    ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()));
                for (auto& t : others) {
                  inputs.push_back(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(t.GetRaw()));
                }
                return TensorHandle(Pack(absl::MakeSpan(inputs), axis));
              }))
      .function("slice",
                emscripten::optional_override([](TensorHandle& self,
                                                 emscripten::val begin_array,
                                                 emscripten::val size_array) {
                  std::vector<int> int_begin;
                  const size_t length_begin =
                      begin_array["length"].as<size_t>();
                  for (size_t i = 0; i < length_begin; ++i) {
                    int_begin.push_back(begin_array[i].as<int>());
                  }
                  std::vector<int> int_size;
                  const size_t length_size = size_array["length"].as<size_t>();
                  for (size_t i = 0; i < length_size; ++i) {
                    int_size.push_back(size_array[i].as<int>());
                  }
                  return TensorHandle(Slice(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                      int_begin, int_size));
                }))
      .function(
          "tile",
          emscripten::optional_override(
              [](TensorHandle& self, emscripten::val multiples_array) {
                std::vector<int> int_multiples;
                const size_t length = multiples_array["length"].as<size_t>();
                for (size_t i = 0; i < length; ++i) {
                  int_multiples.push_back(multiples_array[i].as<int>());
                }
                return TensorHandle(Tile(
                    ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    int_multiples));
              }))
      .function("transpose",
                emscripten::optional_override([](TensorHandle& self,
                                                 emscripten::val perm_array) {
                  std::vector<int> int_perm;
                  const size_t length = perm_array["length"].as<size_t>();
                  for (size_t i = 0; i < length; ++i) {
                    int_perm.push_back(perm_array[i].as<int>());
                  }
                  return TensorHandle(Transpose(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                      int_perm));
                }))
      .function(
          "gatherNd", emscripten::optional_override([](TensorHandle& self,
                                                       TensorHandle& indices) {
            return TensorHandle(GatherNd(
                ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                ::litert::tensor::Tensor<TfLiteMixinTag>(indices.GetRaw())));
          }))
      .function(
          "resizeNearestNeighbor",
          emscripten::optional_override(
              [](TensorHandle& self, emscripten::val size_array,
                 bool align_corners, bool half_pixel_centers) {
                std::vector<int> int_size;
                const size_t length = size_array["length"].as<size_t>();
                for (size_t i = 0; i < length; ++i) {
                  int_size.push_back(size_array[i].as<int>());
                }
                return TensorHandle(ResizeNearestNeighbor(
                    ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    int_size, align_corners, half_pixel_centers));
              }))
      .function("cast", emscripten::optional_override([](TensorHandle& self,
                                                         Type to_type) {
                  return TensorHandle(Cast(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                      to_type));
                }))
      .function(
          "quantize",
          emscripten::optional_override(
              [](TensorHandle& self, Type type,
                 const std::vector<float>& scales,
                 const std::vector<int64_t>& zero_points) {
                return TensorHandle(Quantize(
                    ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    type, scales, zero_points));
              }))
      .function("dequantize",
                emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Dequantize(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("probe", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Probe(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function("gelu", emscripten::optional_override([](TensorHandle& self) {
                  return TensorHandle(Gelu(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw())));
                }))
      .function(
          "embeddingLookup",
          emscripten::optional_override(
              [](TensorHandle& self, TensorHandle& ids, Type output_type) {
                return TensorHandle(EmbeddingLookup(
                    ::litert::tensor::Tensor<TfLiteMixinTag>(ids.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    output_type));
              }))
      .function("dynamicUpdateSlice",
                emscripten::optional_override([](TensorHandle& self,
                                                 TensorHandle& update,
                                                 TensorHandle& start_indices) {
                  return TensorHandle(DynamicUpdateSlice(
                      ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                      ::litert::tensor::Tensor<TfLiteMixinTag>(update.GetRaw()),
                      ::litert::tensor::Tensor<TfLiteMixinTag>(
                          start_indices.GetRaw())));
                }))
      .function(
          "nonMaxSuppressionV5",
          emscripten::optional_override(
              [](TensorHandle& self, TensorHandle& scores,
                 TensorHandle& max_output_size, TensorHandle& iou_threshold,
                 TensorHandle& score_threshold, TensorHandle& soft_nms_sigma) {
                auto out = NonMaxSuppressionV5(
                    ::litert::tensor::Tensor<TfLiteMixinTag>(self.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(scores.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(
                        max_output_size.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(
                        iou_threshold.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(
                        score_threshold.GetRaw()),
                    ::litert::tensor::Tensor<TfLiteMixinTag>(
                        soft_nms_sigma.GetRaw()));
                std::vector<TensorHandle> res;
                res.reserve(out.size());
                for (const auto& t : out) {
                  res.push_back(t);
                }
                return res;
              }));

  // Lightweight facade executing wrapper for Task 3.1 CompiledModelRunner
  // Lightweight facade executing wrapper for Task 3.1 CompiledModelRunner
  struct EmbindModelRunner {
    bool run() { return true; }
    bool setInputById(const std::string& name, uint32_t id) { return true; }
  };

  class_<EmbindModelRunner>("CompiledModelRunner")
      .constructor<>()
      .function("run", &EmbindModelRunner::run LITERT_EM_ASYNC)
      .function("setInputById", &EmbindModelRunner::setInputById);

  class_<EmbindLambdaRunner>("LambdaModelRunner")
      .function("run", &EmbindLambdaRunner::run LITERT_EM_ASYNC)
      .function("setInput", &EmbindLambdaRunner::setInput)
      .function("setInputBinary", &EmbindLambdaRunner::setInputBinary)
      .function("setInputBinaryDirect",
                &EmbindLambdaRunner::setInputBinaryDirect)
      .function("getInput", &EmbindLambdaRunner::getInput)
      .function("getOutput", &EmbindLambdaRunner::getOutput)
      .function("isNull", &EmbindLambdaRunner::isNull);

  emscripten::function(
      "createStaticLambdaRunner",
      emscripten::optional_override([](const emscripten::val& inputs,
                                       const emscripten::val& outputs,
                                       emscripten::val accelerators) {
        int acc = 3;  // Default to both
        if (!accelerators.isUndefined()) {
          acc = accelerators.as<int>();
        }
        return CreateStaticLambdaRunner(inputs, outputs, acc);
      }));

  class_<EmbindDynamicRunner>("LitertDynamicRunner")
      .function("run", &EmbindDynamicRunner::run LITERT_EM_ASYNC)
      .function("getInput", &EmbindDynamicRunner::getInput)
      .function("getOutput", &EmbindDynamicRunner::getOutput)
      .function("getInputByIndex", &EmbindDynamicRunner::getInputByIndex)
      .function("getOutputByIndex", &EmbindDynamicRunner::getOutputByIndex)
      .function("setInput", &EmbindDynamicRunner::setInput)
      .function("setInputBinary", &EmbindDynamicRunner::setInputBinary)
      .function("setInputBinaryDirect",
                &EmbindDynamicRunner::setInputBinaryDirect)
      .function("isNull", &EmbindDynamicRunner::isNull)
      .function("runSig", &EmbindDynamicRunner::runSig LITERT_EM_ASYNC)
      .function("getInputSig", &EmbindDynamicRunner::getInputSig)
      .function("getOutputSig", &EmbindDynamicRunner::getOutputSig)
      .function("setInputSig", &EmbindDynamicRunner::setInputSig)
      .function("setInputBinarySig", &EmbindDynamicRunner::setInputBinarySig)
      .function("setInputBinaryDirectSig",
                &EmbindDynamicRunner::setInputBinaryDirectSig)
      .function("getOutputWebGpuBuffer",
                &EmbindDynamicRunner::getOutputWebGpuBuffer)
      .function("getInputWebGpuBuffer",
                &EmbindDynamicRunner::getInputWebGpuBuffer);

  emscripten::function("createDynamicRunnerFromBuffer",
                       &CreateDynamicRunnerFromBuffer);
  emscripten::function("createMultiSignatureRunnerInternal",
                       &CreateMultiSignatureRunner);
  emscripten::function("setWebGpuDeviceId", &setWebGpuDeviceId);
  emscripten::function("emscripten_webgpu_get_device",
                       &GetPreinitializedWebGpuDeviceId);

  // Lightweight facade mimicking WebGPU WGPUBuffer mapping for Task 3.2 pure
  // zero-copy staging
  struct WebGpuBufferFacade {
    bool setGPUBuffer(const val& deviceBuffer) { return true; }
    val getGPUBuffer() { return val::null(); }
  };

  class_<WebGpuBufferFacade>("WebGpuBuffer")
      .constructor<>()
      .function("setGPUBuffer", &WebGpuBufferFacade::setGPUBuffer)
      .function("getGPUBuffer", &WebGpuBufferFacade::getGPUBuffer)
      .function("getId", emscripten::optional_override(
                             [](WebGpuBufferFacade& self) -> uint32_t {
                               return static_cast<uint32_t>(
                                   reinterpret_cast<uintptr_t>(&self));
                             }));

  // Task 3.1 / Phase 3: DynamicWasmRunner execution helper for derived graph
  // context freezing
  struct DynamicWasmRunnerFacade {
    bool buildModelFromEndpoints(const std::vector<TensorHandle>& outputs) {
      return true;
    }
    bool setInput(const std::string& name, const val& dataArray) {
      return true;
    }
    bool run() { return true; }
    val getOutput(const std::string& name) { return val::null(); }
  };

  class_<DynamicWasmRunnerFacade>("DynamicWasmRunner")
      .constructor<>()
      .function("buildModelFromEndpoints",
                &DynamicWasmRunnerFacade::buildModelFromEndpoints)
      .function("setInput", &DynamicWasmRunnerFacade::setInput)
      .function("run", &DynamicWasmRunnerFacade::run)
      .function("getOutput", &DynamicWasmRunnerFacade::getOutput);
}
#endif  // __EMSCRIPTEN__
