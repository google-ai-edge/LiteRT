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

#include "litert/tools/vision_eval/runner_test_util.h"

#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <ios>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <streambuf>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "litert/cc/litert_expected.h"
#include "litert/tools/vision_eval/data.h"
#include "litert/tools/vision_eval/evaluation.pb.h"
#include "litert/tools/vision_eval/image.h"
#include "litert/tools/vision_eval/runner.h"
#include "google/protobuf/message.h"  // from @com_google_protobuf
#include "google/protobuf/text_format.h"  // from @com_google_protobuf
#include "tflite/converter/schema/schema_generated.h"

namespace litert::vision {
namespace {

void NoOpSignal(int) {}

}  // namespace

bool PathExistsForTest(const std::filesystem::path& path) {
  std::error_code ec;
  return std::filesystem::exists(path, ec) && !ec;
}

std::string ReadFileText(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(input)), {});
}

std::string FormatProto(const google::protobuf::Message& message) {
  std::string text;
  (void)google::protobuf::TextFormat::PrintToString(message, &text);
  return text;
}

int CallMain(absl::Span<const std::string> args) {
  std::vector<std::string> mutable_args(args.begin(), args.end());
  std::vector<char*> argv;
  argv.reserve(mutable_args.size());
  for (std::string& arg : mutable_args) argv.push_back(arg.data());
  return Main(static_cast<int>(argv.size()), argv.data());
}

SignalOnNewlineBuf::int_type SignalOnNewlineBuf::overflow(int_type ch) {
  if (ch != traits_type::eof()) {
    const char c = traits_type::to_char_type(ch);
    text_.push_back(c);
    if (c == '\n' && ++newlines_ == target_newline_) {
      std::raise(sig_);
    }
  }
  return traits_type::not_eof(ch);
}

ScopedCerrBuf::ScopedCerrBuf(std::streambuf* next)
    : prev_(std::cerr.rdbuf(next)) {}

ScopedCerrBuf::~ScopedCerrBuf() { std::cerr.rdbuf(prev_); }

InterruptedMainResult RunMainInterruptedBySignal(
    absl::Span<const std::string> args, int sig, int target_newline) {
  const auto saved_int = std::signal(SIGINT, NoOpSignal);
  const auto saved_term = std::signal(SIGTERM, NoOpSignal);
  SignalOnNewlineBuf buf(sig, target_newline);
  InterruptedMainResult out;
  {
    ScopedCerrBuf redirect(&buf);
    out.exit_code = CallMain(args);
  }
  out.sigint_restored = std::signal(SIGINT, saved_int) == NoOpSignal;
  out.sigterm_restored = std::signal(SIGTERM, saved_term) == NoOpSignal;
  out.stderr_text = buf.text();
  return out;
}

TensorSpec MakeTensorSpec(absl::string_view name, absl::Span<const int> shape,
                          absl::string_view binding_name) {
  TensorSpec spec;
  if (!name.empty()) spec.set_name(name);
  if (!binding_name.empty()) spec.set_binding_name(binding_name);
  for (int dim : shape) spec.add_shape(dim);
  return spec;
}

TensorSpec MakeTensorSpec(absl::string_view name, ElementType type,
                          absl::Span<const int> shape,
                          absl::string_view binding_name) {
  TensorSpec spec = MakeTensorSpec(name, shape, binding_name);
  spec.set_type(type);
  return spec;
}

void SetManifestQuantization(DatasetManifest& manifest, ElementType type,
                             float in_scale, int in_zero_point, float out_scale,
                             int out_zero_point) {
  for (auto& input : *manifest.mutable_inputs()) {
    input.set_type(type);
    input.set_scale(in_scale);
    input.set_zero_point(in_zero_point);
  }
  for (auto& output : *manifest.mutable_outputs()) {
    output.mutable_tensor()->set_type(type);
    output.mutable_tensor()->set_scale(out_scale);
    output.mutable_tensor()->set_zero_point(out_zero_point);
  }
}

void SetManifestQuantization(DatasetManifest& manifest, ElementType type,
                             float scale, int zero_point) {
  SetManifestQuantization(manifest, type, scale, zero_point, scale, zero_point);
}

DatasetManifest MakeSingleInputManifest(const TensorSpec& input_spec,
                                        absl::string_view input_file, int label,
                                        const TensorSpec* output_spec,
                                        std::optional<Task> output_task) {
  DatasetManifest manifest;
  *manifest.add_inputs() = input_spec;
  if (output_spec != nullptr) {
    auto* out = manifest.add_outputs();
    *out->mutable_tensor() = *output_spec;
    if (output_task.has_value()) {
      out->mutable_scoring()->set_task(*output_task);
    }
  }
  auto* sample = manifest.add_samples();
  sample->set_id("s0");
  sample->set_label(label);
  auto* in = sample->add_inputs();
  in->set_name(input_spec.name());
  in->mutable_file()->set_path(input_file);
  return manifest;
}

Expected<void> PackAndWriteModel(const std::filesystem::path& path,
                                 tflite::ModelT& model) {
  if (model.signature_defs.empty()) {
    const auto& graph = *model.subgraphs[0];
    auto signature = std::make_unique<tflite::SignatureDefT>();
    signature->signature_key = "serving_default";
    for (int32_t idx : graph.inputs) {
      auto map = std::make_unique<tflite::TensorMapT>();
      map->name = graph.tensors[idx]->name;
      map->tensor_index = static_cast<uint32_t>(idx);
      signature->inputs.push_back(std::move(map));
    }
    for (int32_t idx : graph.outputs) {
      auto map = std::make_unique<tflite::TensorMapT>();
      map->name = graph.tensors[idx]->name;
      map->tensor_index = static_cast<uint32_t>(idx);
      signature->outputs.push_back(std::move(map));
    }
    model.signature_defs.push_back(std::move(signature));
  }
  flatbuffers::FlatBufferBuilder builder;
  tflite::FinishModelBuffer(builder, tflite::Model::Pack(builder, &model));
  return WriteBytes(path, builder.GetBufferPointer(), builder.GetSize());
}

void WriteTwoInputTwoOutputAddModel(const std::filesystem::path& path) {
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
  ASSERT_TRUE(PackAndWriteModel(path, model));
}

void WriteQuantizedModelCopy(const std::filesystem::path& src_path,
                             const std::filesystem::path& dst_path,
                             ElementType tensor_type, int zero_point,
                             float scale) {
  ASSERT_TRUE(tensor_type == UINT8 || tensor_type == INT8);
  std::ifstream input(src_path, std::ios::binary);
  ASSERT_TRUE(input);
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
  std::unique_ptr<tflite::ModelT> model(
      tflite::GetModel(bytes.data())->UnPack());
  for (auto& tensor : model->subgraphs[0]->tensors) {
    tensor->type = tensor_type == UINT8 ? tflite::TensorType_UINT8
                                        : tflite::TensorType_INT8;
    tensor->quantization = std::make_unique<tflite::QuantizationParametersT>();
    tensor->quantization->scale = {scale};
    tensor->quantization->zero_point = {zero_point};
  }
  model->operator_codes[0]->version = 2;
  ASSERT_TRUE(PackAndWriteModel(dst_path, *model));
}

void WriteClassificationModel(const std::filesystem::path& path,
                              ElementType tensor_type,
                              absl::Span<const float> bias,
                              absl::Span<const int32_t> shape,
                              absl::string_view input_name) {
  ASSERT_TRUE(tensor_type == INT8 || tensor_type == FLOAT32);
  const bool int8_io = tensor_type == INT8;
  size_t elements = 1;
  for (int32_t dim : shape) elements *= static_cast<size_t>(dim);
  ASSERT_EQ(bias.size(), elements);

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
  const std::string names[] = {std::string(input_name), "bias", "logits"};
  for (int i = 0; i < 3; ++i) {
    auto tensor = std::make_unique<tflite::TensorT>();
    tensor->name = names[i];
    tensor->type =
        int8_io ? tflite::TensorType_INT8 : tflite::TensorType_FLOAT32;
    tensor->shape.assign(shape.begin(), shape.end());
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
  ASSERT_TRUE(PackAndWriteModel(path, model));
}

void RunnerTest::SetUp() {
  const auto* test = testing::UnitTest::GetInstance()->current_test_info();
  root_ = Utf8ToPath(testing::TempDir()) / test->name();
  std::error_code ec;
  std::filesystem::create_directories(root_, ec);
  ASSERT_FALSE(ec);
  ASSERT_NO_FATAL_FAILURE(
      WriteTwoInputTwoOutputAddModel(root_ / "model.tflite"));

  for (const char* binding : {"left", "right"}) {
    *manifest_.add_inputs() =
        MakeTensorSpec(std::string("data_") + binding, {1, 1, 1, 3}, binding);
  }
  for (bool depth : {true, false}) {
    auto* output = manifest_.add_outputs();
    *output->mutable_tensor() =
        MakeTensorSpec(depth ? "depth" : "segmentation", {1, 1, 1, 3});
    output->mutable_scoring()->set_task(depth ? DEPTH : SEGMENTATION);
    output->mutable_scoring()->set_classes(1);
  }

  const float inputs[] = {1, 2, 3};
  const float truth[] = {2, 4, 6};
  const uint8_t labels[] = {0, 0, 0};
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
      *target->mutable_tensor() =
          MakeTensorSpec("", depth ? FLOAT32 : UINT8, {1, 3});
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

void RunnerTest::WriteQuantizedModelCopy(const std::filesystem::path& path,
                                         ElementType tensor_type,
                                         int zero_point, float scale) {
  ::litert::vision::WriteQuantizedModelCopy(root_ / "model.tflite", path,
                                            tensor_type, zero_point, scale);
}

void RunnerTest::SetupClassificationDataset(ClassificationDatasetFixture& out) {
  out.img_dir = root_ / "imgs";
  std::error_code ec;
  std::filesystem::create_directories(out.img_dir, ec);
  ASSERT_FALSE(ec);
  const std::string ppm =
      std::string("P6\n2 2\n255\n") + std::string(12, '\0');
  const std::string ppm_b =
      std::string("P6\n2 2\n255\n\x00\x00\xff", 14) + std::string(9, '\0');
  ASSERT_TRUE(WriteBytes(out.img_dir / "sample.ppm", ppm.data(), ppm.size()));
  ASSERT_TRUE(
      WriteBytes(out.img_dir / "sample_b.ppm", ppm_b.data(), ppm_b.size()));
  const std::string tsv = "0\t2\tsample.ppm\n1\t5\tsample_b.ppm\n";
  out.tsv_path = root_ / "samples.tsv";
  ASSERT_TRUE(WriteBytes(out.tsv_path, tsv.data(), tsv.size()));

  out.cand_path = root_ / "cls_i8.tflite";
  out.base_agree_path = root_ / "cls_fp32.tflite";
  out.base_disagree_path = root_ / "cls_disagree.tflite";
  ASSERT_NO_FATAL_FAILURE(WriteClassificationModel(
      out.cand_path, INT8, {0, 0, 5, 0, 0, 4, 0, 0, 0, 0, 0, 0},
      {1, 2, 2, 3}, "pixels"));
  ASSERT_NO_FATAL_FAILURE(WriteClassificationModel(
      out.base_agree_path, FLOAT32,
      {0, 0, 5.15f, 0, 0, 4, 0, 0, 0, 0, 0, 0}, {1, 2, 2, 3}, "pixels"));
  ASSERT_NO_FATAL_FAILURE(WriteClassificationModel(
      out.base_disagree_path, FLOAT32,
      {0, 0, 4.5f, 0, 0, 5, 0, 0, 0, 0, 0, 0}, {1, 2, 2, 3}, "pixels"));

  out.config.Clear();
  out.config.set_model(PathToUtf8(out.cand_path));
  out.config.set_baseline_model(PathToUtf8(out.base_agree_path));
  out.config.set_baseline_accelerator("cpu");
  out.config.set_manifest(PathToUtf8(out.tsv_path));
  out.config.set_image_root(PathToUtf8(out.img_dir));
  out.config.set_output_dir(PathToUtf8(root_ / "cls_results"));
  out.config.mutable_backend()->set_accelerator("cpu");
  out.config.set_atol(0.25);  // NOLINT(runtime/deprecated_fn)
  out.config.set_compare_int8_codes(true);
  out.config.set_max_code_error(1);
  out.config.set_mean_code_error(1.0);
  out.config.set_require_output_parity(true);
  out.config.set_require_prediction_agreement(true);
  auto* prep = out.config.mutable_preprocessing();
  prep->set_resize(2);
  prep->set_crop_h(2);
  prep->set_crop_w(2);
  for (int i = 0; i < 3; ++i) {
    prep->add_mean(0.0f);
    prep->add_stddev(1.0f);
  }
}

Expected<void> RunnerTest::Run(RunSummary& summary) {
  return Evaluate(config_, Utf8ToPath(config_.model()), summary);
}

void RunnerTest::ExpectRunError(EvaluationConfig cfg,
                                absl::string_view out_subdir,
                                absl::string_view expected_substr,
                                RunSummary* summary_out) {
  SCOPED_TRACE(out_subdir);
  if (!out_subdir.empty()) {
    cfg.set_output_dir(PathToUtf8(root_ / Utf8ToPath(out_subdir)));
  }
  const auto binary_path = !cfg.model().empty() ? Utf8ToPath(cfg.model())
                                                : Utf8ToPath(config_.model());
  RunSummary summary;
  const auto res = Evaluate(cfg, binary_path, summary);
  ASSERT_FALSE(res) << "Expected error containing: " << expected_substr;
  EXPECT_TRUE(absl::StrContains(res.Error().Message(), expected_substr))
      << res.Error().Message();
  if (!summary.error().empty()) {
    EXPECT_EQ(summary.status(), FAILED);
    EXPECT_EQ(summary.error(), res.Error().Message());
  }
  if (summary_out != nullptr) *summary_out = std::move(summary);
}

void RunnerTest::ExpectManifestError(const EvaluationConfig& base_cfg,
                                     const DatasetManifest& manifest,
                                     absl::string_view case_name,
                                     absl::string_view expected_substr) {
  const auto path = root_ / Utf8ToPath(absl::StrCat(case_name, ".txtpb"));
  ASSERT_TRUE(WriteProto(path, manifest, TEXT_PROTO));
  EvaluationConfig cfg = base_cfg;
  cfg.set_manifest(PathToUtf8(path));
  RunSummary summary;
  ExpectRunError(cfg, absl::StrCat("out_", case_name), expected_substr,
                 &summary);
  EXPECT_EQ(summary.status(), FAILED);
  EXPECT_FALSE(summary.error().empty());
}

void RunnerTest::ExpectManifestError(const DatasetManifest& manifest,
                                     absl::string_view case_name,
                                     absl::string_view expected_substr) {
  ExpectManifestError(config_, manifest, case_name, expected_substr);
}

void RunnerTest::ExpectCorruptCheckpoint(
    const EvaluationConfig& cfg, const std::filesystem::path& row_path,
    absl::FunctionRef<void(SampleResult&)> mutate,
    absl::string_view expected_substr) {
  SCOPED_TRACE(expected_substr);
  SampleResult valid_row;
  ASSERT_TRUE(ReadProto(row_path, valid_row));
  SampleResult corrupt = valid_row;
  mutate(corrupt);
  ASSERT_TRUE(WriteProto(row_path, corrupt));
  RunSummary bad;
  const auto res = Evaluate(cfg, Utf8ToPath(cfg.model()), bad);
  ASSERT_TRUE(WriteProto(row_path, valid_row));
  ASSERT_FALSE(res) << "Expected error containing: " << expected_substr;
  EXPECT_EQ(bad.status(), FAILED);
  EXPECT_EQ(bad.resumed(), 0);
  EXPECT_TRUE(absl::StrContains(bad.error(), expected_substr)) << bad.error();
}

void RunnerTest::ExpectCorruptCheckpoint(
    const std::filesystem::path& row_path,
    absl::FunctionRef<void(SampleResult&)> mutate,
    absl::string_view expected_substr) {
  ExpectCorruptCheckpoint(config_, row_path, mutate, expected_substr);
}

}  // namespace litert::vision
