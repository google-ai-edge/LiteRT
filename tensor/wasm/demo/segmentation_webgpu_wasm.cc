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
#include <emscripten/val.h>

#include <cstdint>
#endif

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#ifdef __EMSCRIPTEN__
#include <webgpu/webgpu.h>
#else
#include "webgpu/webgpu.h"  // from @dawn
#endif
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/webgpu/arithmetic_webgpu.h"
#include "tensor/backends/webgpu/webgpu_conversion.h"
#include "tensor/backends/webgpu/webgpu_headers.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/runners/webgpu/webgpu_runner.h"
#include "tensor/tensor.h"
#include "tensor/utils/macros.h"

namespace litert::tensor {
namespace {

using WebGpuTensor = Tensor<WebGpuMixinTag>;

const char* kPreprocessShader = R"(
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    const IN_W: u32 = 512;
    const IN_H: u32 = 512;
    const OUT_W: u32 = 256;
    const OUT_H: u32 = 256;
    const CHANNELS: u32 = 3;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
        let index = gid.x;
        let total_elements = OUT_W * OUT_H * CHANNELS;
        if (index >= total_elements) { return; }

        let pixel_index = index / CHANNELS;
        let c = index % CHANNELS;

        let x = pixel_index % OUT_W;
        let y = pixel_index / OUT_W;

        let gx = f32(x) * (f32(IN_W) / f32(OUT_W));
        let gy = f32(y) * (f32(IN_H) / f32(OUT_H));

        let x0 = u32(floor(gx));
        let y0 = u32(floor(gy));
        let x1 = min(x0 + 1u, IN_W - 1u);
        let y1 = min(y0 + 1u, IN_H - 1u);

        let dx = gx - f32(x0);
        let dy = gy - f32(y0);

        let v00 = input[(y0 * IN_W + x0) * CHANNELS + c];
        let v10 = input[(y0 * IN_W + x1) * CHANNELS + c];
        let v01 = input[(y1 * IN_W + x0) * CHANNELS + c];
        let v11 = input[(y1 * IN_W + x1) * CHANNELS + c];

        let v0 = v00 * (1.0 - dx) + v10 * dx;
        let v1 = v01 * (1.0 - dx) + v11 * dx;
        let v = v0 * (1.0 - dy) + v1 * dy;

        output[index] = v;
    }
)";

const char* kPostprocessShader = R"(
    @group(0) @binding(0) var<storage, read> model_output: array<f32>;
    @group(0) @binding(1) var<storage, read> colors: array<f32>;
    @group(0) @binding(2) var<storage, read> original_image: array<f32>;
    @group(0) @binding(3) var<storage, read> factor: array<f32>;
    @group(0) @binding(4) var<storage, read_write> output: array<f32>;

    const W: u32 = 256;
    const H: u32 = 256;
    const CLASSES: u32 = 6;
    const CHANNELS: u32 = 3;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
        let pixel_index = gid.x;
        if (pixel_index >= W * H) { return; }

        // 1. ArgMax
        var max_val = -1000.0;
        var max_class = 0u;
        for (var c = 0u; c < CLASSES; c = c + 1u) {
            let val = model_output[pixel_index * CLASSES + c];
            if (val > max_val) {
                max_val = val;
                max_class = c;
            }
        }

        let f = factor[0];

        for (var c = 0u; c < CHANNELS; c = c + 1u) {
            let color_val = colors[max_class * CHANNELS + c];
            let orig_val = original_image[pixel_index * CHANNELS + c];
            output[pixel_index * CHANNELS + c] = (orig_val * f) + (color_val * (1.0 - f));
        }
    }
)";

class SegmentationPipeline {
 public:
  SegmentationPipeline() = default;

  absl::Status Init(std::vector<uint8_t> model_buffer) {
    model_buffer_ = std::move(model_buffer);
#ifdef __EMSCRIPTEN__
    WGPUDevice c_device = emscripten_webgpu_get_device();
#else
    WGPUDevice c_device = nullptr;
#endif
    wgpu::Device device(c_device);

    // Initialize Core Model
    auto env_or = litert::Environment::Create({});
    if (!env_or.HasValue()) {
      return absl::InternalError("Failed to create LiteRT environment");
    }
    env_ = std::make_shared<litert::Environment>(std::move(*env_or));

    auto options_or = litert::Options::Create();
    if (!options_or.HasValue()) {
      return absl::InternalError("Failed to create LiteRT options");
    }
    options_ = std::make_shared<litert::Options>(std::move(*options_or));
    options_->SetHardwareAccelerators(litert::HwAccelerators::kGpu);

    auto span = absl::MakeConstSpan(model_buffer_.data(), model_buffer_.size());
    auto core_runner_or = LitertDynamicRunner::Create(*env_, span, *options_);
    if (!core_runner_or.ok()) return core_runner_or.status();
    core_runner_ =
        std::make_shared<LitertDynamicRunner>(std::move(*core_runner_or));

    // Try to get WebGPU buffer from core runner for zero-copy
    absl::flat_hash_map<std::string, WGPUBuffer> external_buffers;
    auto core_in_buf_or =
        core_runner_->GetInputWebGpuBuffer("default", "input");
    if (core_in_buf_or.ok()) {
      external_buffers["normalized_image"] =
          reinterpret_cast<WGPUBuffer>(*core_in_buf_or);
      ABSL_LOG(INFO) << "Using zero-copy buffer sharing for normalized_image.";
    }

    // Pre-processing Graph
    WebGpuTensor raw_image(
        {.name = "raw_image", .type = Type::kFP32, .shape = {1, 512, 512, 3}});
    std::vector<std::vector<int>> pre_output_shapes = {{1, 256, 256, 3}};
    std::vector<Type> pre_output_types = {Type::kFP32};

    auto pre_outputs = Custom({raw_image}, kPreprocessShader, {},
                              pre_output_shapes, pre_output_types);
    WebGpuTensor resized_image = pre_outputs[0];

    TensorInit two_init;
    two_init.name = "two";
    two_init.type = Type::kFP32;
    two_init.shape = {1};
    two_init.buffer =
        OwningCpuBuffer::Copy<Type::kFP32>(std::vector<float>{2.0f});
    TensorHandle two_tensor(two_init);
    auto scaled_image = Mul(resized_image, WebGpuTensor(two_tensor));

    TensorInit one_init;
    one_init.name = "one";
    one_init.type = Type::kFP32;
    one_init.shape = {1};
    one_init.buffer =
        OwningCpuBuffer::Copy<Type::kFP32>(std::vector<float>{1.0f});
    TensorHandle one_tensor(one_init);
    auto normalized_image = Sub(scaled_image, WebGpuTensor(one_tensor));

    auto pre_runner_or =
        WebGpuRunner::Create(c_device, {normalized_image}, external_buffers);
    if (!pre_runner_or.ok()) return pre_runner_or.status();
    pre_runner_ = std::make_unique<WebGpuRunner>(std::move(*pre_runner_or));

    // Post-processing Graph
    WebGpuTensor model_output({.name = "model_output",
                               .type = Type::kFP32,
                               .shape = {1, 256, 256, 6}});
    WebGpuTensor colors(
        {.name = "colors", .type = Type::kFP32, .shape = {6, 3}});
    WebGpuTensor original_image({.name = "original_image",
                                 .type = Type::kFP32,
                                 .shape = {1, 256, 256, 3}});
    WebGpuTensor factor({.name = "factor", .type = Type::kFP32, .shape = {1}});

    std::vector<std::vector<int>> post_output_shapes = {{1, 256, 256, 3}};
    std::vector<Type> post_output_types = {Type::kFP32};

    auto post_outputs =
        Custom({model_output, colors, original_image, factor},
               kPostprocessShader, {}, post_output_shapes, post_output_types);
    WebGpuTensor blended = post_outputs[0];

    auto post_runner_or = WebGpuRunner::Create(c_device, {blended});
    if (!post_runner_or.ok()) return post_runner_or.status();
    post_runner_ = std::make_unique<WebGpuRunner>(std::move(*post_runner_or));

    // Setup constant data for post-processing
    std::vector<float> flat_colors = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                      0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                      1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    std::vector<float> factor_val = {0.5f};

    auto colors_tensor = Create("colors", Type::kFP32, {6, 3}, flat_colors);
    LRT_TENSOR_RETURN_IF_ERROR(post_runner_->SetInput("colors", colors_tensor));

    auto factor_tensor = Create("factor", Type::kFP32, {1}, factor_val);
    LRT_TENSOR_RETURN_IF_ERROR(post_runner_->SetInput("factor", factor_tensor));

    return absl::OkStatus();
  }

  absl::Status ProcessFrame(uintptr_t input_data_ptr,
                            uintptr_t original_data_ptr,
                            uintptr_t output_data_ptr) {
    // Assuming input_data_ptr points to a float array of size 1*512*512*3
    // Assuming original_data_ptr points to a float array of size 1*256*256*3
    // Assuming output_data_ptr points to a float array of size 1*256*256*3

    // 1. Pre-processing
    const float* in_ptr = reinterpret_cast<const float*>(input_data_ptr);
    std::vector<float> in_vec(in_ptr, in_ptr + 1 * 512 * 512 * 3);
    auto raw_tensor =
        Create("raw_image", Type::kFP32, {1, 512, 512, 3}, std::move(in_vec));
    LRT_TENSOR_RETURN_IF_ERROR(pre_runner_->SetInput("raw_image", raw_tensor));
    LRT_TENSOR_RETURN_IF_ERROR(pre_runner_->Run());

    // 2. Core Model
    auto norm_tensor_or = pre_runner_->GetOutput(0);
    if (!norm_tensor_or.ok()) return norm_tensor_or.status();
    LRT_TENSOR_RETURN_IF_ERROR(core_runner_->SetInput(0, *norm_tensor_or));
    LRT_TENSOR_RETURN_IF_ERROR(core_runner_->Run());

    // 3. Post-processing
    auto core_output_or = core_runner_->GetOutput(0);
    if (!core_output_or.ok()) return core_output_or.status();
    LRT_TENSOR_RETURN_IF_ERROR(
        post_runner_->SetInput("model_output", *core_output_or));

    const float* orig_ptr = reinterpret_cast<const float*>(original_data_ptr);
    std::vector<float> orig_vec(orig_ptr, orig_ptr + 1 * 256 * 256 * 3);
    auto orig_tensor = Create("original_image", Type::kFP32, {1, 256, 256, 3},
                              std::move(orig_vec));
    LRT_TENSOR_RETURN_IF_ERROR(
        post_runner_->SetInput("original_image", orig_tensor));

    LRT_TENSOR_RETURN_IF_ERROR(post_runner_->Run());

    // 4. Retrieve Output
    auto blended_tensor_or = post_runner_->GetOutput(0);
    if (!blended_tensor_or.ok()) return blended_tensor_or.status();

    auto buffer_or = blended_tensor_or->GetBuffer();
    if (!buffer_or.ok()) return buffer_or.status();
    auto lock_span = buffer_or->Lock();

    std::memcpy(reinterpret_cast<void*>(output_data_ptr), lock_span.data(),
                lock_span.size());

    return absl::OkStatus();
  }

 private:
  std::vector<uint8_t> model_buffer_;
  std::shared_ptr<litert::Environment> env_;
  std::shared_ptr<litert::Options> options_;
  std::shared_ptr<LitertDynamicRunner> core_runner_;
  std::unique_ptr<WebGpuRunner> pre_runner_;
  std::unique_ptr<WebGpuRunner> post_runner_;
};

}  // namespace
}  // namespace litert::tensor

#ifdef __EMSCRIPTEN__
using emscripten::class_;
using litert::tensor::SegmentationPipeline;

EMSCRIPTEN_BINDINGS(segmentation_webgpu) {
  class_<SegmentationPipeline>("SegmentationPipeline")
      .constructor<>()
      .function("init", emscripten::optional_override(
                            [](SegmentationPipeline& self,
                               emscripten::val model_bytes) -> bool {
                              size_t l = model_bytes["length"].as<size_t>();
                              std::vector<uint8_t> vec(l);
                              for (size_t i = 0; i < l; ++i) {
                                vec[i] = model_bytes[i].as<uint8_t>();
                              }
                              return self.Init(std::move(vec)).ok();
                            }))
      // Process frame binding with JSPI async policy
      .function("processFrame",
                emscripten::optional_override(
                    [](SegmentationPipeline& self, uintptr_t input_data_ptr,
                       uintptr_t original_data_ptr,
                       uintptr_t output_data_ptr) -> bool {
                      return self
                          .ProcessFrame(input_data_ptr, original_data_ptr,
                                        output_data_ptr)
                          .ok();
                    }),
                emscripten::async());
}
#endif
