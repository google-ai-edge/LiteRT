/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
==============================================================================*/
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "third_party/odml/litert/tensor/arithmetic.h"
#include "third_party/odml/litert/tensor/backends/tflite/arithmetic_tflite.h"
#include "third_party/odml/litert/tensor/buffer.h"
#include "third_party/odml/litert/tensor/datatypes.h"
#include "third_party/odml/litert/tensor/examples/segmentation/image_utils.h"
#include "third_party/odml/litert/tensor/runners/litert/lambda_model_runner.h"
#include "third_party/odml/litert/tensor/runners/litert/litert_dynamic_runner.h"
#include "third_party/odml/litert/tensor/tensor.h"

namespace litert::tensor {
namespace {

// --- PRE-PROCESSING TYPES ---

void RunEndToEndPipeline(const std::string& core_model,
                         const std::string& image_file,
                         const std::string& output_dir,
                         const std::string& accelerator) {
  ABSL_LOG(INFO) << "Loading image: " << image_file;
  int width, height, channels;
  unsigned char* img_data =
      ImageUtils::LoadImage(image_file, width, height, channels, 3);
  if (!img_data) {
    ABSL_LOG(ERROR) << "Failed to load image.";
    return;
  }
  auto input_floats =
      ImageUtils::ResizeImageCpu(img_data, width, height, 3, 512, 512);

  auto env_or = litert::Environment::Create({});
  auto env = std::move(*env_or);

  auto options_or = litert::Options::Create();
  auto options = std::move(*options_or);
  if (accelerator == "cpu") {
    options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
  } else {
    options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
  }

  ABSL_LOG(INFO) << "Building and connecting models using runners...";
  auto pre_runner = CreateLambdaRunner(
      env, options,
      {{"raw_image", Tensor<TfLiteMixinTag>({.name = "raw_image",
                                             .type = Type::kFP32,
                                             .shape = {1, 512, 512, 3}})}},
      [](const auto& inputs) {
        Tensor resized = ResizeBilinear(inputs.at("raw_image"), {256, 256});
        Tensor scaled = Mul(resized, 2.0f);
        Tensor normalized = Add(scaled, -1.0f);
        return absl::flat_hash_map<std::string, Tensor<TfLiteMixinTag>>{
            {"normalized_image", normalized}};
      });

  auto post_runner = CreateLambdaRunner(
      env, options,
      {{"model_output", Tensor<TfLiteMixinTag>({.name = "model_output",
                                                .type = Type::kFP32,
                                                .shape = {1, 256, 256, 6}})},
       {"original_image", Tensor<TfLiteMixinTag>({.name = "original_image",
                                                  .type = Type::kFP32,
                                                  .shape = {1, 256, 256, 3}})},
       {"colors",
        Tensor<TfLiteMixinTag>(
            {.name = "colors", .type = Type::kFP32, .shape = {6, 3}})},
       {"factor", Tensor<TfLiteMixinTag>(
                      {.name = "factor", .type = Type::kFP32, .shape = {1}})}},
      [](const auto& inputs) {
        Tensor winning_classes =
            ArgMax(inputs.at("model_output"), -1, Type::kI32);
        Tensor winning_classes_1d = Reshape(winning_classes, {256 * 256});
        Tensor mask_color_1d =
            Gather(inputs.at("colors"), winning_classes_1d, 0);
        Tensor mask_color = Reshape(mask_color_1d, {1, 256, 256, 3});

        Tensor scaled_orig =
            Mul(inputs.at("original_image"), inputs.at("factor"));
        Tensor scaled_mask = Mul(mask_color, inputs.at("factor"));
        Tensor blended = Add(scaled_orig, scaled_mask);

        return absl::flat_hash_map<std::string, Tensor<TfLiteMixinTag>>{
            {"blended", blended}};
      });

  auto core_runner_or = LitertDynamicRunner::Create(env, core_model, options);
  ABSL_CHECK_OK(core_runner_or.status());
  auto core_runner = std::move(*core_runner_or);

  auto raw_image_tensor = Create("raw_image", Type::kFP32, {1, 512, 512, 3},
                                 std::move(input_floats));
  ABSL_CHECK_OK(pre_runner.SetInput("raw_image", raw_image_tensor));

  auto core_input_or = core_runner.GetInput(0);
  ABSL_CHECK_OK(core_input_or.status());

  // Hook up zero-copy!
  ABSL_CHECK_OK(pre_runner.SetOutput("normalized_image", *core_input_or));

  // 1. Pre-processing
  ABSL_CHECK_OK(pre_runner.Run());

  // 2. Core Model
  ABSL_CHECK_OK(core_runner.Run());

  // 3. Post-processing Setup
  auto core_output_or = core_runner.GetOutput(0);
  ABSL_CHECK_OK(core_output_or.status());
  TensorHandle core_output = std::move(*core_output_or);

  auto resized_original =
      ImageUtils::ResizeImageCpu(img_data, width, height, 3, 256, 256);
  std::vector<float> flat_colors = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                    0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                    1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
  std::vector<float> factor_val = {0.5f};
  auto resized_original_tensor =
      Create("original_image", Type::kFP32, {1, 256, 256, 3},
             std::move(resized_original));
  auto colors_tensor =
      Create("colors", Type::kFP32, {6, 3}, std::move(flat_colors));
  auto factor_tensor =
      Create("factor", Type::kFP32, {1}, std::move(factor_val));

  ABSL_CHECK_OK(post_runner.SetInput("model_output", core_output));
  ABSL_CHECK_OK(
      post_runner.SetInput("original_image", resized_original_tensor));
  ABSL_CHECK_OK(post_runner.SetInput("colors", colors_tensor));
  ABSL_CHECK_OK(post_runner.SetInput("factor", factor_tensor));

  // 4. Post-processing Run
  ABSL_CHECK_OK(post_runner.Run());

  auto post_output_or = post_runner.GetOutput("blended");
  ABSL_CHECK_OK(post_output_or.status());
  TensorHandle blended_tensor = std::move(*post_output_or);

  auto buffer_or = blended_tensor.GetBuffer();
  ABSL_CHECK_OK(buffer_or.status());
  auto locked_span = buffer_or->Lock();

  const float* final_blended_floats =
      reinterpret_cast<const float*>(locked_span.data());
  size_t num_elements = locked_span.size() / sizeof(float);

  std::vector<unsigned char> blended_image(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    blended_image[i] = static_cast<unsigned char>(
        std::max(0.0f, std::min(1.0f, final_blended_floats[i])) * 255.0f);
  }

  std::string output_file = output_dir + "/segmented_output.png";
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
  std::string accelerator = "gpu";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--output_dir=") == 0) {
      output_dir = arg.substr(13);
    } else if (arg.find("--core_model_path=") == 0) {
      core_model_path = arg.substr(18);
    } else if (arg.find("--image_path=") == 0) {
      image_path = arg.substr(13);
    } else if (arg.find("--accelerator=") == 0) {
      accelerator = arg.substr(14);
    }
  }

  litert::tensor::RunEndToEndPipeline(core_model_path, image_path, output_dir,
                                      accelerator);
  return 0;
}

