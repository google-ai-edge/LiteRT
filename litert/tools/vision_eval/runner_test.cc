// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/tools/vision_eval/runner.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/litert_expected.h"
#include "litert/tools/vision_eval/data.h"
#include "litert/tools/vision_eval/run_lock.h"
#include "tflite/converter/schema/schema_generated.h"

namespace litert::vision {
namespace {

bool Exists(const std::filesystem::path& path) {
  std::error_code ec;
  return std::filesystem::exists(path, ec) && !ec;
}

int CallMain(std::vector<std::string> args) {
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) argv.push_back(arg.data());
  return Main(static_cast<int>(argv.size()), argv.data());
}

class RunnerTest : public testing::Test {
 protected:
  void SetUp() override {
    const auto* test = testing::UnitTest::GetInstance()->current_test_info();
    root_ = Utf8ToPath(testing::TempDir()) / test->name();
    std::error_code ec;
    std::filesystem::create_directories(root_, ec);
    ASSERT_FALSE(ec);
    tflite::ModelT model;
    model.version = 3;
    model.buffers.push_back(std::make_unique<tflite::BufferT>());
    model.operator_codes.push_back(std::make_unique<tflite::OperatorCodeT>());
    model.operator_codes[0]->builtin_code = tflite::BuiltinOperator_ADD;
    auto graph = std::make_unique<tflite::SubGraphT>();
    graph->inputs = {0, 1};
    graph->outputs = {2, 3};
    for (const char* name : {"left", "right", "depth", "segmentation"}) {
      auto tensor = std::make_unique<tflite::TensorT>();
      tensor->name = name;
      tensor->type = tflite::TensorType_FLOAT32;
      tensor->shape = {1, 1, 1, 3};
      graph->tensors.push_back(std::move(tensor));
    }
    for (int output : {2, 3}) {
      auto op = std::make_unique<tflite::OperatorT>();
      op->opcode_index = 0;
      op->inputs = {0, 1};
      op->outputs = {output};
      op->builtin_options.Set(tflite::AddOptionsT());
      graph->operators.push_back(std::move(op));
    }
    model.subgraphs.push_back(std::move(graph));
    auto signature = std::make_unique<tflite::SignatureDefT>();
    signature->signature_key = "serving_default";
    for (int i = 0; i < 4; ++i) {
      auto map = std::make_unique<tflite::TensorMapT>();
      map->name = model.subgraphs[0]->tensors[i]->name;
      map->tensor_index = i;
      (i < 2 ? signature->inputs : signature->outputs).push_back(std::move(map));
    }
    model.signature_defs.push_back(std::move(signature));
    flatbuffers::FlatBufferBuilder builder;
    auto packed = tflite::Model::Pack(builder, &model);
    tflite::FinishModelBuffer(builder, packed);
    ASSERT_TRUE(WriteBytes(root_ / "model.tflite", builder.GetBufferPointer(),
                           builder.GetSize()));
    for (const char* binding : {"left", "right"}) {
      auto* input = manifest_.add_inputs();
      input->set_name(std::string("data_") + binding);
      input->set_binding_name(binding);
      for (int dim : {1, 1, 1, 3}) input->add_shape(dim);
    }
    for (bool depth : {true, false}) {
      auto* output = manifest_.add_outputs();
      output->mutable_tensor()->set_name(depth ? "depth" : "segmentation");
      for (int dim : {1, 1, 1, 3}) output->mutable_tensor()->add_shape(dim);
      output->mutable_scoring()->set_task(depth ? DEPTH : SEGMENTATION);
      output->mutable_scoring()->set_classes(1);
    }
    float inputs[] = {1, 2, 3};
    float truth[] = {2, 4, 6};
    uint8_t labels[] = {0, 0, 0};
    ASSERT_TRUE(WriteBytes(root_ / "input.raw", inputs, sizeof(inputs)));
    ASSERT_TRUE(WriteBytes(root_ / "depth.raw", truth, sizeof(truth)));
    ASSERT_TRUE(WriteBytes(root_ / "labels.raw", labels, sizeof(labels)));
    for (int i = 0; i < 3; ++i) {
      auto* sample = manifest_.add_samples();
      sample->set_id("sample_" + std::to_string(i));
      for (const auto& spec : manifest_.inputs()) {
        auto* input = sample->add_inputs();
        input->set_name(spec.name());
        input->mutable_file()->set_path("input.raw");
      }
      for (bool depth : {true, false}) {
        auto* target = sample->add_targets();
        target->set_name(depth ? "depth" : "segmentation");
        target->mutable_file()->set_path(depth ? "depth.raw" : "labels.raw");
        target->mutable_tensor()->set_type(depth ? FLOAT32 : UINT8);
        target->mutable_tensor()->add_shape(1);
        target->mutable_tensor()->add_shape(3);
      }
    }
    ASSERT_TRUE(WriteProto(root_ / "manifest.txtpb", manifest_, TEXT_PROTO));
    config_.set_model(PathToUtf8(root_ / "model.tflite"));
    config_.set_manifest(PathToUtf8(root_ / "manifest.txtpb"));
    config_.set_output_dir(PathToUtf8(root_ / "results"));
    config_.mutable_backend()->set_accelerator("cpu");
    config_.mutable_backend()->set_num_threads(1);
    config_.set_dump_predictions(true);
    config_.set_dump_inputs(true);
  }
  void QuantizedModel(const std::filesystem::path& path, bool unsigned_input,
                      int zero_point, float scale = .25f) {
    std::ifstream input(root_ / "model.tflite", std::ios::binary);
    ASSERT_TRUE(input);
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
    std::unique_ptr<tflite::ModelT> model(tflite::GetModel(bytes.data())->UnPack());
    for (auto& tensor : model->subgraphs[0]->tensors) {
      tensor->type = unsigned_input ? tflite::TensorType_UINT8 : tflite::TensorType_INT8;
      tensor->quantization = std::make_unique<tflite::QuantizationParametersT>();
      tensor->quantization->scale = {scale};
      tensor->quantization->zero_point = {zero_point};
    }
    model->operator_codes[0]->version = 2;
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, model.get()));
    ASSERT_TRUE(WriteBytes(path, builder.GetBufferPointer(), builder.GetSize()));
  }
  void WriteClassificationModel(const std::filesystem::path& path, bool int8_io,
                                const std::vector<float>& bias) {
    ASSERT_EQ(bias.size(), 12u);
    tflite::ModelT model;
    model.version = 3;
    model.buffers.push_back(std::make_unique<tflite::BufferT>());
    auto bias_buf = std::make_unique<tflite::BufferT>();
    if (int8_io) {
      bias_buf->data.resize(bias.size());
      for (size_t i = 0; i < bias.size(); ++i) {
        const int code = static_cast<int>(bias[i] * 4.0f);
        bias_buf->data[i] = static_cast<uint8_t>(static_cast<int8_t>(code));
      }
    } else {
      bias_buf->data.resize(bias.size() * sizeof(float));
      std::memcpy(bias_buf->data.data(), bias.data(), bias_buf->data.size());
    }
    model.buffers.push_back(std::move(bias_buf));
    model.operator_codes.push_back(std::make_unique<tflite::OperatorCodeT>());
    model.operator_codes[0]->builtin_code = tflite::BuiltinOperator_ADD;
    model.operator_codes[0]->version = int8_io ? 2 : 1;
    auto graph = std::make_unique<tflite::SubGraphT>();
    graph->inputs = {0};
    graph->outputs = {2};
    const int buffers[] = {0, 1, 0};
    const char* names[] = {"image", "bias", "logits"};
    for (int i = 0; i < 3; ++i) {
      auto tensor = std::make_unique<tflite::TensorT>();
      tensor->name = names[i];
      tensor->type =
          int8_io ? tflite::TensorType_INT8 : tflite::TensorType_FLOAT32;
      tensor->shape = {1, 2, 2, 3};
      tensor->buffer = buffers[i];
      if (int8_io) {
        tensor->quantization =
            std::make_unique<tflite::QuantizationParametersT>();
        tensor->quantization->scale = {0.25f};
        tensor->quantization->zero_point = {0};
      }
      graph->tensors.push_back(std::move(tensor));
    }
    auto op = std::make_unique<tflite::OperatorT>();
    op->opcode_index = 0;
    op->inputs = {0, 1};
    op->outputs = {2};
    op->builtin_options.Set(tflite::AddOptionsT());
    graph->operators.push_back(std::move(op));
    model.subgraphs.push_back(std::move(graph));
    auto signature = std::make_unique<tflite::SignatureDefT>();
    signature->signature_key = "serving_default";
    auto in_map = std::make_unique<tflite::TensorMapT>();
    in_map->name = "image";
    in_map->tensor_index = 0;
    signature->inputs.push_back(std::move(in_map));
    auto out_map = std::make_unique<tflite::TensorMapT>();
    out_map->name = "logits";
    out_map->tensor_index = 2;
    signature->outputs.push_back(std::move(out_map));
    model.signature_defs.push_back(std::move(signature));
    flatbuffers::FlatBufferBuilder builder;
    tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
    ASSERT_TRUE(
        WriteBytes(path, builder.GetBufferPointer(), builder.GetSize()));
  }
  Expected<void> Run(RunSummary& summary) {
    // The test's model is a stable stand-in for the executable identity.
    return Evaluate(config_, summary, Utf8ToPath(config_.model()));
  }
  std::filesystem::path root_;
  EvaluationConfig config_;
  DatasetManifest manifest_;
};

TEST_F(RunnerTest, NamedMultipleInputsAndOutputsBaselineAndResume) {
  config_.set_baseline_model(config_.model());
  config_.set_baseline_accelerator("cpu");
  config_.set_require_output_parity(true);
  config_.set_warmup_runs(1);
  RunSummary first;
  auto status = Run(first);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_EQ(first.status(), PASSED);
  EXPECT_EQ(first.completed(), 3);
  EXPECT_EQ(first.metrics_size(), 2);
  EXPECT_EQ(first.baseline_metrics_size(), 2);
  EXPECT_EQ(first.comparisons_size(), 2);
  EXPECT_TRUE(first.comparisons(0).allclose());
  EXPECT_TRUE(first.identity().candidate().has_model());
  EXPECT_EQ(first.identity().candidate().inputs_size(), 2);
  EXPECT_TRUE(Exists(root_ / "results/samples/sample_0/depth.raw"));
  RunSummary second;
  status = Run(second);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_EQ(second.resumed(), 3);
  EXPECT_EQ(second.metrics(0).SerializeAsString(), first.metrics(0).SerializeAsString());
  EXPECT_EQ(second.identity().config().SerializeAsString(),
            first.identity().config().SerializeAsString());
}

TEST_F(RunnerTest, FailedSampleLeavesCompletedSamplesResumable) {
  manifest_.mutable_samples(1)->mutable_inputs(0)->mutable_file()->set_path("missing.raw");
  ASSERT_TRUE(WriteProto(root_ / "manifest.txtpb", manifest_, TEXT_PROTO));
  RunSummary first;
  EXPECT_FALSE(Run(first));
  EXPECT_EQ(first.status(), FAILED);
  EXPECT_EQ(first.completed(), 1);
  EXPECT_TRUE(Exists(root_ / "results/samples/sample_0/result.binpb"));
  EXPECT_FALSE(Exists(root_ / "results/samples/sample_1/result.binpb"));
  float inputs[] = {1, 2, 3};
  ASSERT_TRUE(WriteBytes(root_ / "missing.raw", inputs, sizeof(inputs)));
  RunSummary second;
  auto status = Run(second);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_EQ(second.completed(), 3);
  EXPECT_EQ(second.resumed(), 1);
}

TEST_F(RunnerTest, ChangedProvenanceDoesNotOverwriteExistingSummary) {
  auto read_text = [](const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(input)), {});
  };
  RunSummary first;
  auto status = Run(first);
  ASSERT_TRUE(status) << status.Error().Message();
  const auto summary_path = root_ / "results/summary.txtpb";
  const auto original = read_text(summary_path);
  ASSERT_FALSE(original.empty());
  config_.set_rtol(.2);
  RunSummary second;
  EXPECT_FALSE(Run(second));
  EXPECT_EQ(original, read_text(summary_path));
}

TEST_F(RunnerTest, MissingCompletedInputOrTruncatedArtifactIsRejected) {
  RunSummary first;
  auto status = Run(first);
  ASSERT_TRUE(status) << status.Error().Message();
  float truncated = 1;
  ASSERT_TRUE(WriteBytes(root_ / "results/samples/sample_0/depth.raw",
                         &truncated, sizeof(truncated)));
  RunSummary second;
  EXPECT_FALSE(Run(second));
  EXPECT_EQ(second.status(), FAILED);
  EXPECT_EQ(second.resumed(), 0);
  std::error_code ec;
  ASSERT_TRUE(std::filesystem::remove(root_ / "input.raw", ec));
  ASSERT_FALSE(ec);
  RunSummary third;
  EXPECT_FALSE(Run(third));
  EXPECT_EQ(third.status(), FAILED);
  EXPECT_EQ(third.resumed(), 0);
}
TEST_F(RunnerTest, QuantizedBaselinesWithDifferentZeroPointsReceiveSameRealInputs) {
  for (bool unsigned_input : {false, true}) {
    const int zero = unsigned_input ? 128 : 7;
    const auto candidate = root_ / (unsigned_input ? "u8.tflite" : "i8.tflite");
    const auto baseline =
        root_ / (unsigned_input ? "base_u8.tflite" : "base_i8.tflite");
    ASSERT_NO_FATAL_FAILURE(QuantizedModel(candidate, unsigned_input, zero));
    ASSERT_NO_FATAL_FAILURE(QuantizedModel(baseline, unsigned_input, zero - 10));
    config_.set_model(PathToUtf8(candidate));
    config_.set_baseline_model(PathToUtf8(baseline));
    config_.set_baseline_accelerator("cpu");
    config_.set_require_output_parity(true);
    config_.set_output_dir(
        PathToUtf8(root_ / (unsigned_input ? "results_u8" : "results_i8")));
    const auto type = unsigned_input ? UINT8 : INT8;
    for (auto& input : *manifest_.mutable_inputs()) {
      input.set_type(type);
      input.set_scale(.25f);
      input.set_zero_point(zero);
    }
    for (auto& output : *manifest_.mutable_outputs()) {
      output.mutable_tensor()->set_type(type);
      output.mutable_tensor()->set_scale(.25f);
      output.mutable_tensor()->set_zero_point(zero);
    }
    const std::vector<uint8_t> input = {static_cast<uint8_t>(zero + 4),
                                       static_cast<uint8_t>(zero + 8),
                                       static_cast<uint8_t>(zero + 12)};
    ASSERT_TRUE(WriteBytes(root_ / "input.raw", input.data(), input.size()));
    ASSERT_TRUE(WriteProto(root_ / "manifest.txtpb", manifest_, TEXT_PROTO));
    RunSummary summary;
    auto status = Run(summary);
    ASSERT_TRUE(status) << status.Error().Message();
    ASSERT_EQ(summary.comparisons_size(), 2);
    EXPECT_TRUE(summary.comparisons(0).allclose());
    EXPECT_TRUE(summary.comparisons(1).allclose());
    EXPECT_EQ(summary.metrics(0).depth().abs_rel(), 0);
  }
}

TEST_F(RunnerTest, NonfinitePreparedInputsFailBeforeInvocation) {
  float input[] = {std::numeric_limits<float>::quiet_NaN(), 2, 3};
  ASSERT_TRUE(WriteBytes(root_ / "input.raw", input, sizeof(input)));
  RunSummary summary;
  EXPECT_FALSE(Run(summary));
  EXPECT_EQ(summary.status(), FAILED);
  EXPECT_EQ(summary.completed(), 0);
  EXPECT_FALSE(Exists(root_ / "results/samples/sample_0/result.binpb"));
}

TEST_F(RunnerTest, SemanticallyCorruptCheckpointIsRejected) {
  RunSummary first;
  auto status = Run(first);
  ASSERT_TRUE(status) << status.Error().Message();
  const auto path = root_ / "results/samples/sample_0/result.binpb";
  SampleResult row;
  ASSERT_TRUE(ReadProto(path, row));
  row.mutable_metrics(0)->mutable_depth()->set_images(2);
  ASSERT_TRUE(WriteProto(path, row));
  RunSummary second;
  EXPECT_FALSE(Run(second));
  EXPECT_EQ(second.status(), FAILED);
}

TEST_F(RunnerTest, OfflineScoreOnlyAndReferenceComparison) {
  RunSummary online;
  auto status = Run(online);
  ASSERT_TRUE(status) << status.Error().Message();

  EvaluationConfig offline;
  offline.set_score_only(true);
  offline.set_manifest(PathToUtf8(root_ / "results/resolved_manifest.txtpb"));
  offline.set_predictions_dir(PathToUtf8(root_ / "results/samples"));
  offline.set_reference_dir(PathToUtf8(root_ / "results/samples"));
  offline.set_output_dir(PathToUtf8(root_ / "offline_results"));
  offline.set_require_output_parity(true);

  RunSummary scored;
  status = Evaluate(offline, scored, Utf8ToPath(config_.model()));
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_EQ(scored.status(), PASSED);
  EXPECT_EQ(scored.completed(), 3);
  EXPECT_EQ(scored.metrics_size(), 2);
  EXPECT_EQ(scored.baseline_metrics_size(), 2);
  ASSERT_EQ(scored.comparisons_size(), 2);
  EXPECT_TRUE(scored.comparisons(0).allclose());
  EXPECT_EQ(scored.metrics(0).SerializeAsString(),
            online.metrics(0).SerializeAsString());

  RunSummary resumed;
  status = Evaluate(offline, resumed, Utf8ToPath(config_.model()));
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_EQ(resumed.resumed(), 3);
}

TEST_F(RunnerTest, TsvClassificationAutoBindingInt8CodesAndComparisonGates) {
  const std::string ppm_header = "P6\n2 2\n255\n";
  std::string ppm = ppm_header + std::string(12, '\0');
  const auto ppm_path = root_ / Utf8ToPath("样本_sample.ppm");
  ASSERT_TRUE(WriteBytes(ppm_path, ppm.data(), ppm.size()));
  const std::string tsv = "0\t2\t样本_sample.ppm\n1\t5\t样本_sample.ppm\n";
  const auto tsv_path = root_ / "samples.tsv";
  ASSERT_TRUE(WriteBytes(tsv_path, tsv.data(), tsv.size()));

  const auto cand_path = root_ / "cls_i8.tflite";
  const auto base_agree_path = root_ / "cls_fp32.tflite";
  const auto base_disagree_path = root_ / "cls_disagree.tflite";
  ASSERT_NO_FATAL_FAILURE(WriteClassificationModel(
      cand_path, /*int8_io=*/true, {0, 0, 5, 0, 0, 4, 0, 0, 0, 0, 0, 0}));
  ASSERT_NO_FATAL_FAILURE(WriteClassificationModel(
      base_agree_path, /*int8_io=*/false,
      {0, 0, 5, 0, 0, 4, 0, 0, 0, 0, 0, 0}));
  ASSERT_NO_FATAL_FAILURE(WriteClassificationModel(
      base_disagree_path, /*int8_io=*/false,
      {0, 0, 1, 0, 0, 5, 0, 0, 0, 0, 0, 0}));

  EvaluationConfig cls_config;
  cls_config.set_model(PathToUtf8(cand_path));
  cls_config.set_baseline_model(PathToUtf8(base_agree_path));
  cls_config.set_baseline_accelerator("cpu");
  cls_config.set_manifest(PathToUtf8(tsv_path));
  cls_config.set_output_dir(PathToUtf8(root_ / "cls_results"));
  cls_config.mutable_backend()->set_accelerator("cpu");
  cls_config.set_compare_int8_codes(true);
  cls_config.set_require_output_parity(true);
  cls_config.set_require_prediction_agreement(true);
  auto* prep = cls_config.mutable_preprocessing();
  prep->set_resize(2);
  prep->set_crop_h(2);
  prep->set_crop_w(2);
  for (int i = 0; i < 3; ++i) {
    prep->add_mean(0.0f);
    prep->add_stddev(1.0f);
  }

  RunSummary summary;
  auto status = Evaluate(cls_config, summary, cand_path);
  ASSERT_TRUE(status) << status.Error().Message();
  ASSERT_EQ(summary.metrics_size(), 1);
  EXPECT_DOUBLE_EQ(summary.metrics(0).classification().top1(), 0.5);
  EXPECT_DOUBLE_EQ(summary.metrics(0).classification().top5(), 1.0);
  ASSERT_EQ(summary.comparisons_size(), 1);
  EXPECT_TRUE(summary.comparisons(0).allclose());
  EXPECT_TRUE(summary.comparisons(0).has_max_code_error());
  EXPECT_TRUE(summary.comparisons(0).has_mean_code_error());
  EXPECT_TRUE(summary.comparisons(0).prediction_agreement());

  cls_config.set_baseline_model(PathToUtf8(base_disagree_path));
  cls_config.set_require_output_parity(false);
  cls_config.set_require_prediction_agreement(true);
  cls_config.set_output_dir(PathToUtf8(root_ / "cls_disagree_gate"));
  RunSummary disagree_summary;
  EXPECT_FALSE(Evaluate(cls_config, disagree_summary, cand_path));
  EXPECT_EQ(disagree_summary.status(), FAILED);

  cls_config.set_compare_int8_codes(false);
  cls_config.set_require_output_parity(true);
  cls_config.set_require_prediction_agreement(false);
  cls_config.set_output_dir(PathToUtf8(root_ / "cls_parity_gate"));
  RunSummary parity_summary;
  EXPECT_FALSE(Evaluate(cls_config, parity_summary, cand_path));
  EXPECT_EQ(parity_summary.status(), FAILED);

  cls_config.mutable_preprocessing()->set_crop_w(4);
  cls_config.set_output_dir(PathToUtf8(root_ / "cls_bad_crop"));
  RunSummary crop_summary;
  EXPECT_FALSE(Evaluate(cls_config, crop_summary, cand_path));
}

TEST_F(RunnerTest, TaskOverrideSupplyQuantizationAndFloatToInt8Baseline) {
  const auto fp32_path = root_ / "cls_f32.tflite";
  const auto int8_path = root_ / "cls_q8.tflite";
  ASSERT_NO_FATAL_FAILURE(WriteClassificationModel(
      fp32_path, /*int8_io=*/false, {0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  ASSERT_NO_FATAL_FAILURE(WriteClassificationModel(
      int8_path, /*int8_io=*/true, {0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

  const std::vector<float> f32_zeros(12, 0.0f);
  ASSERT_TRUE(WriteBytes(root_ / "cls_in_f32.raw", f32_zeros.data(),
                         f32_zeros.size() * sizeof(float)));
  DatasetManifest f32_manifest;
  auto* in_f32 = f32_manifest.add_inputs();
  in_f32->set_name("image");
  in_f32->set_type(FLOAT32);
  for (int d : {1, 2, 2, 3}) in_f32->add_shape(d);
  auto* out_f32 = f32_manifest.add_outputs();
  out_f32->mutable_tensor()->set_name("logits");
  out_f32->mutable_tensor()->set_type(FLOAT32);
  for (int d : {1, 2, 2, 3}) out_f32->mutable_tensor()->add_shape(d);
  auto* sample_f32 = f32_manifest.add_samples();
  sample_f32->set_id("s0");
  sample_f32->set_label(2);
  auto* sample_in_f32 = sample_f32->add_inputs();
  sample_in_f32->set_name("image");
  sample_in_f32->mutable_file()->set_path("cls_in_f32.raw");
  const auto f32_manifest_path = root_ / "cls_f32_manifest.txtpb";
  ASSERT_TRUE(WriteProto(f32_manifest_path, f32_manifest, TEXT_PROTO));

  EvaluationConfig cfg;
  cfg.set_model(PathToUtf8(fp32_path));
  cfg.set_baseline_model(PathToUtf8(int8_path));
  cfg.set_baseline_accelerator("cpu");
  cfg.set_manifest(PathToUtf8(f32_manifest_path));
  cfg.set_output_dir(PathToUtf8(root_ / "f32_to_i8_out"));
  cfg.set_task(CLASSIFICATION);
  cfg.mutable_backend()->set_accelerator("cpu");
  RunSummary f32_summary;
  auto status = Evaluate(cfg, f32_summary, fp32_path);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_DOUBLE_EQ(f32_summary.metrics(0).classification().top1(), 1.0);
  EXPECT_DOUBLE_EQ(f32_summary.baseline_metrics(0).classification().top1(),
                   1.0);

  const std::vector<uint8_t> i8_zeros(12, 0);
  ASSERT_TRUE(
      WriteBytes(root_ / "cls_in_i8.raw", i8_zeros.data(), i8_zeros.size()));
  DatasetManifest i8_manifest;
  auto* in_i8 = i8_manifest.add_inputs();
  in_i8->set_name("image");
  in_i8->set_type(INT8);
  for (int d : {1, 2, 2, 3}) in_i8->add_shape(d);
  auto* out_i8 = i8_manifest.add_outputs();
  out_i8->mutable_tensor()->set_name("logits");
  out_i8->mutable_tensor()->set_type(INT8);
  for (int d : {1, 2, 2, 3}) out_i8->mutable_tensor()->add_shape(d);
  out_i8->mutable_scoring()->set_task(CLASSIFICATION);
  auto* sample_i8 = i8_manifest.add_samples();
  sample_i8->set_id("s0");
  sample_i8->set_label(2);
  auto* sample_in_i8 = sample_i8->add_inputs();
  sample_in_i8->set_name("image");
  sample_in_i8->mutable_file()->set_path("cls_in_i8.raw");
  const auto i8_manifest_path = root_ / "cls_i8_manifest.txtpb";
  ASSERT_TRUE(WriteProto(i8_manifest_path, i8_manifest, TEXT_PROTO));

  EvaluationConfig i8_cfg;
  i8_cfg.set_model(PathToUtf8(int8_path));
  i8_cfg.set_manifest(PathToUtf8(i8_manifest_path));
  i8_cfg.set_output_dir(PathToUtf8(root_ / "supply_quant_out"));
  i8_cfg.mutable_backend()->set_accelerator("cpu");
  RunSummary i8_summary;
  status = Evaluate(i8_cfg, i8_summary, int8_path);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_DOUBLE_EQ(i8_summary.metrics(0).classification().top1(), 1.0);
}

TEST_F(RunnerTest, ExtremeQuantizationScaleSkipsRedundantInputRoundTrip) {
  const auto extreme_model = root_ / "extreme_i8.tflite";
  ASSERT_NO_FATAL_FAILURE(QuantizedModel(
      extreme_model, /*unsigned_input=*/false, /*zero_point=*/0,
      /*scale=*/1e37f));
  const int8_t pos[] = {100, 0, 0};
  const int8_t neg[] = {-100, 0, 0};
  const float empty_depth[] = {-1.0f, -1.0f, -1.0f};
  ASSERT_TRUE(WriteBytes(root_ / "pos.raw", pos, sizeof(pos)));
  ASSERT_TRUE(WriteBytes(root_ / "neg.raw", neg, sizeof(neg)));
  ASSERT_TRUE(
      WriteBytes(root_ / "depth.raw", empty_depth, sizeof(empty_depth)));
  for (auto& input : *manifest_.mutable_inputs()) {
    input.set_type(INT8);
    input.set_scale(1e37f);
    input.set_zero_point(0);
  }
  for (auto& output : *manifest_.mutable_outputs()) {
    output.mutable_tensor()->set_type(INT8);
    output.mutable_tensor()->set_scale(1e37f);
    output.mutable_tensor()->set_zero_point(0);
  }
  for (auto& sample : *manifest_.mutable_samples()) {
    sample.mutable_inputs(0)->mutable_file()->set_path("pos.raw");
    sample.mutable_inputs(1)->mutable_file()->set_path("neg.raw");
  }
  ASSERT_TRUE(WriteProto(root_ / "manifest.txtpb", manifest_, TEXT_PROTO));
  config_.set_model(PathToUtf8(extreme_model));
  RunSummary summary;
  auto status = Run(summary);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_EQ(summary.status(), PASSED);
}

TEST_F(RunnerTest, RejectsInvalidConfigsAndMismatchedTensorContracts) {
  RunSummary bad_cfg;
  EXPECT_FALSE(
      Evaluate(EvaluationConfig{}, bad_cfg, Utf8ToPath(config_.model())));
  EXPECT_EQ(bad_cfg.status(), FAILED);
  EXPECT_FALSE(bad_cfg.error().empty());

  auto bad_binding = manifest_;
  bad_binding.mutable_outputs(0)->mutable_tensor()->set_binding_name("missing");
  ASSERT_TRUE(
      WriteProto(root_ / "bad_binding.txtpb", bad_binding, TEXT_PROTO));
  config_.set_manifest(PathToUtf8(root_ / "bad_binding.txtpb"));
  config_.set_output_dir(PathToUtf8(root_ / "out_bad_binding"));
  RunSummary summary;
  EXPECT_FALSE(Run(summary));

  auto bad_shape = manifest_;
  bad_shape.mutable_outputs(0)->mutable_tensor()->set_shape(3, 99);
  ASSERT_TRUE(WriteProto(root_ / "bad_shape.txtpb", bad_shape, TEXT_PROTO));
  config_.set_manifest(PathToUtf8(root_ / "bad_shape.txtpb"));
  config_.set_output_dir(PathToUtf8(root_ / "out_bad_shape"));
  EXPECT_FALSE(Run(summary));

  const auto q_model = root_ / "q_mismatch.tflite";
  ASSERT_NO_FATAL_FAILURE(QuantizedModel(q_model, /*unsigned_input=*/false, 0));
  auto bad_quant = manifest_;
  for (auto& in : *bad_quant.mutable_inputs()) {
    in.set_type(INT8);
    in.set_scale(0.5f);
    in.set_zero_point(0);
  }
  for (auto& out : *bad_quant.mutable_outputs()) {
    out.mutable_tensor()->set_type(INT8);
    out.mutable_tensor()->set_scale(0.25f);
    out.mutable_tensor()->set_zero_point(0);
  }
  ASSERT_TRUE(WriteProto(root_ / "bad_quant.txtpb", bad_quant, TEXT_PROTO));
  config_.set_model(PathToUtf8(q_model));
  config_.set_manifest(PathToUtf8(root_ / "bad_quant.txtpb"));
  config_.set_output_dir(PathToUtf8(root_ / "out_bad_quant"));
  EXPECT_FALSE(Run(summary));

  auto one_input = manifest_;
  one_input.mutable_inputs()->DeleteSubrange(1, 1);
  for (auto& sample : *one_input.mutable_samples()) {
    sample.mutable_inputs()->DeleteSubrange(1, 1);
  }
  ASSERT_TRUE(WriteProto(root_ / "one_input.txtpb", one_input, TEXT_PROTO));
  config_.set_model(PathToUtf8(root_ / "model.tflite"));
  config_.set_manifest(PathToUtf8(root_ / "one_input.txtpb"));
  config_.set_output_dir(PathToUtf8(root_ / "out_one_input"));
  EXPECT_FALSE(Run(summary));
}

TEST_F(RunnerTest, MainHandlesHelpInvalidFlagsSuccessAndFailure) {
  EXPECT_EQ(CallMain({"litert_vision_eval", "--help"}), 0);
  EXPECT_EQ(CallMain({"litert_vision_eval", "invalid_positional"}), 1);

  const auto cli_out = root_ / "cli_out";
  EXPECT_EQ(CallMain({"litert_vision_eval",
                      "--model=" + PathToUtf8(root_ / "model.tflite"),
                      "--manifest=" + PathToUtf8(root_ / "manifest.txtpb"),
                      "--output_dir=" + PathToUtf8(cli_out)}),
            0);
  EXPECT_TRUE(Exists(cli_out / "summary.txtpb"));

  const auto cli_fail_out = root_ / "cli_fail_out";
  EXPECT_EQ(CallMain({"litert_vision_eval",
                      "--model=" + PathToUtf8(root_ / "model.tflite"),
                      "--manifest=" + PathToUtf8(root_ / "nonexistent.txtpb"),
                      "--output_dir=" + PathToUtf8(cli_fail_out)}),
            1);

  std::error_code ec;
  std::filesystem::create_directories(Utf8ToPath(config_.output_dir()), ec);
  ASSERT_FALSE(ec);
  RunLock active_lock;
  ASSERT_TRUE(
      active_lock.Acquire(Utf8ToPath(config_.output_dir()) / "run.lock"));
  RunSummary locked_summary;
  EXPECT_FALSE(Run(locked_summary));
}
}  // namespace
}  // namespace litert::vision
