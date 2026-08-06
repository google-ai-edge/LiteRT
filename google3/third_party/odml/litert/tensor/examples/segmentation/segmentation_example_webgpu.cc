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
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "webgpu/webgpu.h"  // from @dawn
#include "ml_drift/webgpu/execution_environment.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_headers.h"  // from @ml_drift
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "third_party/odml/litert/tensor/arithmetic.h"
#include "third_party/odml/litert/tensor/backends/webgpu/arithmetic_webgpu.h"
#include "third_party/odml/litert/tensor/backends/webgpu/webgpu_conversion.h"
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/examples/segmentation/image_utils.h"
#include "third_party/odml/litert/tensor/runners/litert/litert_dynamic_runner.h"
#include "third_party/odml/litert/tensor/runners/webgpu/webgpu_runner.h"
#include "third_party/odml/litert/tensor/tensor.h"

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

void RunEndToEndPipeline(const std::string& core_model,
                         const std::string& image_file,
                         const std::string& output_dir) {
  ABSL_LOG(INFO) << "Loading image: " << image_file;
  int width, height, channels;
  unsigned char* img_data =
      ImageUtils::LoadImage(image_file, width, height, channels, 3);
  if (!img_data) {
    ABSL_LOG(ERROR) << "Failed to load image.";
    return;
  }

  // Resize on CPU to 512x512 as in the original example, to match shader
  // expectations.
  auto input_floats =
      ImageUtils::ResizeImageCpu(img_data, width, height, 3, 512, 512);

  // Initialize WebGPU environment
  auto webgpu_env = std::make_unique<ml_drift::webgpu::ExecutionEnvironment>(
#if defined(__APPLE__)
      wgpu::BackendType::Metal
#elif defined(_WIN32)
      wgpu::BackendType::D3D12
#elif defined(__EMSCRIPTEN__)
      wgpu::BackendType::WebGPU
#else
      wgpu::BackendType::Vulkan
#endif
  );
  ABSL_CHECK_OK(webgpu_env->Initialize());
  WGPUDevice device = webgpu_env->device().Get();

  // 2. Core Model (Initialize first to get its buffers)
  auto env_or = litert::Environment::Create({});
  auto env = std::move(*env_or);
  auto options_or = litert::Options::Create();
  auto options = std::move(*options_or);
  options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);

  auto core_runner_or = LitertDynamicRunner::Create(env, core_model, options);
  ABSL_CHECK_OK(core_runner_or.status());
  auto core_runner = std::move(*core_runner_or);

  // Try to get WebGPU buffer from core runner for zero-copy
  absl::flat_hash_map<std::string, WGPUBuffer> external_buffers;
  auto core_in_buf_or = core_runner.GetInputWebGpuBuffer("default", "input");
  if (core_in_buf_or.ok()) {
    external_buffers["normalized_image"] =
        reinterpret_cast<WGPUBuffer>(*core_in_buf_or);
    ABSL_LOG(INFO) << "Using zero-copy buffer sharing for normalized_image.";
  } else {
    ABSL_LOG(WARNING) << "Failed to get WebGPU buffer from core runner, "
                         "falling back to copy mode.";
  }

  // 1. Pre-processing Graph
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

  ABSL_LOG(INFO) << "Building pre-processing runner...";
  auto pre_runner_or =
      WebGpuRunner::Create(device, {normalized_image}, external_buffers);
  ABSL_CHECK_OK(pre_runner_or.status());
  auto pre_runner = std::move(*pre_runner_or);

  auto raw_tensor =
      Create("raw_image", Type::kFP32, {1, 512, 512, 3}, input_floats);
  ABSL_CHECK_OK(pre_runner.SetInput("raw_image", raw_tensor));

  ABSL_LOG(INFO) << "Running pre-processing runner...";
  ABSL_CHECK_OK(pre_runner.Run());
  ABSL_LOG(INFO) << "Pre-processing runner finished.";

  if (external_buffers.empty()) {
    // Fallback mode: read back and set input
    auto norm_tensor_or = pre_runner.GetOutput(0);
    ABSL_CHECK_OK(norm_tensor_or.status());
    auto norm_tensor = std::move(*norm_tensor_or);
    ABSL_CHECK_OK(core_runner.SetInput(0, norm_tensor));
  } else {
    ABSL_LOG(INFO) << "Skipping SetInput on core runner as buffer was shared.";
  }

  ABSL_CHECK_OK(core_runner.Run());

  auto core_output_or = core_runner.GetOutput(0);
  ABSL_CHECK_OK(core_output_or.status());
  TensorHandle core_output = std::move(*core_output_or);

  // 3. Post-processing Graph
  WebGpuTensor model_output(
      {.name = "model_output", .type = Type::kFP32, .shape = {1, 256, 256, 6}});
  WebGpuTensor colors({.name = "colors", .type = Type::kFP32, .shape = {6, 3}});
  WebGpuTensor original_image({.name = "original_image",
                               .type = Type::kFP32,
                               .shape = {1, 256, 256, 3}});
  WebGpuTensor factor({.name = "factor", .type = Type::kFP32, .shape = {1}});

  std::vector<std::vector<int>> post_output_shapes = {{1, 256, 256, 3}};
  std::vector<Type> post_output_types = {Type::kFP32};

  auto post_outputs =
      Custom({model_output, colors, original_image, factor}, kPostprocessShader,
             {}, post_output_shapes, post_output_types);
  WebGpuTensor blended = post_outputs[0];

  auto post_runner_or = WebGpuRunner::Create(device, {blended});
  ABSL_CHECK_OK(post_runner_or.status());
  auto post_runner = std::move(*post_runner_or);

  // Data for post-processing
  std::vector<float> flat_colors = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                    0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                    1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
  std::vector<float> factor_val = {0.5f};
  auto resized_original =
      ImageUtils::ResizeImageCpu(img_data, width, height, 3, 256, 256);

  ABSL_CHECK_OK(post_runner.SetInput("model_output", core_output));

  auto colors_tensor = Create("colors", Type::kFP32, {6, 3}, flat_colors);
  ABSL_CHECK_OK(post_runner.SetInput("colors", colors_tensor));

  auto orig_tensor =
      Create("original_image", Type::kFP32, {1, 256, 256, 3}, resized_original);
  ABSL_CHECK_OK(post_runner.SetInput("original_image", orig_tensor));

  auto factor_tensor = Create("factor", Type::kFP32, {1}, factor_val);
  ABSL_CHECK_OK(post_runner.SetInput("factor", factor_tensor));

  ABSL_CHECK_OK(post_runner.Run());

  auto blended_tensor_or = post_runner.GetOutput(0);
  ABSL_CHECK_OK(blended_tensor_or.status());
  auto blended_tensor = std::move(*blended_tensor_or);

  // Save image
  auto buffer_or = blended_tensor.GetBuffer();
  ABSL_CHECK_OK(buffer_or.status());
  auto lock_span = buffer_or->Lock();

  size_t num_elements = lock_span.size() / sizeof(float);
  std::vector<unsigned char> blended_image(num_elements);
  const float* final_blended_floats =
      reinterpret_cast<const float*>(lock_span.data());
  for (size_t i = 0; i < num_elements; ++i) {
    blended_image[i] = static_cast<unsigned char>(
        std::max(0.0f, std::min(1.0f, final_blended_floats[i])) * 255.0f);
  }

  std::string output_file = output_dir + "/segmented_output_webgpu.png";
  ImageUtils::SaveImage(output_file, 256, 256, 3, blended_image.data());
  ABSL_LOG(INFO) << "Saved blended image to " << output_file;

  ImageUtils::FreeImageData(img_data);
}

}  // namespace
}  // namespace litert::tensor

int main(int argc, char** argv) {
  std::string core_model_path =
      "third_party/odml/litert/tensor/examples/segmentation/"
      "selfie_multiclass_256x256.tflite";
  std::string image_path =
      "third_party/odml/litert/tensor/examples/segmentation/image.jpeg";
  std::string output_dir = "/tmp";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--output_dir=") == 0) {
      output_dir = arg.substr(13);
    } else if (arg.find("--core_model_path=") == 0) {
      core_model_path = arg.substr(18);
    } else if (arg.find("--image_path=") == 0) {
      image_path = arg.substr(13);
    }
  }

  litert::tensor::RunEndToEndPipeline(core_model_path, image_path, output_dir);
  return 0;
}
