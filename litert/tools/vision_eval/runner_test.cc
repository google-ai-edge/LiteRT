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

#include <csignal>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/tools/vision_eval/data.h"
#include "litert/tools/vision_eval/evaluation.pb.h"
#include "litert/tools/vision_eval/image.h"
#include "litert/tools/vision_eval/run_lock.h"
#include "litert/tools/vision_eval/runner_test_util.h"

namespace litert::vision {
namespace {

TEST_F(RunnerTest, NamedMultipleInputsAndOutputsBaselineAndResume) {
  const auto empty_libs = root_ / "empty_libs";
  std::error_code ec;
  std::filesystem::create_directories(empty_libs, ec);
  ASSERT_FALSE(ec);
  config_.mutable_backend()->set_runtime_library_dir(PathToUtf8(empty_libs));
  config_.mutable_backend()->set_dispatch_library_dir(PathToUtf8(empty_libs));
  config_.mutable_backend()->set_compiler_plugin_library_dir(
      PathToUtf8(empty_libs));

  const uint8_t mask[] = {1, 1, 1};
  ASSERT_TRUE(WriteBytes(root_ / "mask.raw", mask, sizeof(mask)));
  manifest_.mutable_inputs(0)->set_layout(NHWC);
  for (auto& sample : *manifest_.mutable_samples()) {
    sample.mutable_targets(0)->mutable_mask()->set_path("mask.raw");
  }
  ASSERT_TRUE(WriteProto(root_ / "manifest.txtpb", manifest_, TEXT_PROTO));

  config_.set_baseline_model(config_.model());
  config_.set_baseline_accelerator("cpu");
  config_.set_compare_int8_codes(true);
  config_.set_require_output_parity(true);
  config_.set_require_prediction_agreement(true);
  config_.set_warmup_runs(1);

  RunSummary first;
  auto status = Run(first);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_EQ(first.status(), PASSED);
  EXPECT_EQ(first.requested(), 3);
  EXPECT_EQ(first.completed(), 3);
  EXPECT_EQ(first.resumed(), 0);
  EXPECT_GT(first.inference_ms(), 0.0);
  EXPECT_GT(first.baseline_inference_ms(), 0.0);
  EXPECT_GT(first.elapsed_seconds(), 0.0);
  EXPECT_EQ(first.metrics_size(), 2);
  EXPECT_EQ(first.baseline_metrics_size(), 2);
  ASSERT_EQ(first.comparisons_size(), 2);
  EXPECT_EQ(first.comparisons(0).name(), "depth");
  EXPECT_TRUE(first.comparisons(0).allclose());
  EXPECT_FALSE(first.comparisons(0).has_max_code_error());
  EXPECT_FALSE(first.comparisons(0).has_prediction_agreement());
  EXPECT_TRUE(first.identity().candidate().has_model());
  EXPECT_FALSE(first.identity().device().operating_system().empty());
  EXPECT_FALSE(first.identity().manifest().path().empty());
  EXPECT_FALSE(first.identity().binary().path().empty());
  ASSERT_EQ(first.identity().candidate().inputs_size(), 2);
  EXPECT_EQ(first.identity().candidate().inputs(0).layout(), NHWC);
  EXPECT_EQ(first.identity().candidate().inputs(1).layout(), NCHW);
  ASSERT_EQ(first.identity().baseline().inputs_size(), 2);
  EXPECT_EQ(first.identity().baseline().inputs(0).layout(), NHWC);
  EXPECT_EQ(first.identity().baseline().inputs(1).layout(), NCHW);
  EXPECT_EQ(first.identity().sample_ids_size(), 3);
  EXPECT_TRUE(PathExistsForTest(root_ / "results/samples/sample_0/depth.raw"));
  EXPECT_TRUE(
      PathExistsForTest(root_ / "results/samples/sample_0/inputs/input_0.raw"));

  SampleResult first_row;
  ASSERT_TRUE(
      ReadProto(root_ / "results/samples/sample_0/result.binpb", first_row));
  ASSERT_EQ(first_row.model_inputs_size(), 2);
  EXPECT_EQ(first_row.model_inputs(0).tensor().layout(), NHWC);
  EXPECT_EQ(first_row.model_inputs(1).tensor().layout(), NCHW);
  double expected_inf_ms = 0.0;
  double expected_base_inf_ms = 0.0;
  for (const char* id : {"sample_0", "sample_1", "sample_2"}) {
    SampleResult row;
    ASSERT_TRUE(
        ReadProto(root_ / "results/samples" / id / "result.binpb", row));
    expected_inf_ms += row.inference_ms();
    expected_base_inf_ms += row.baseline_inference_ms();
  }
  EXPECT_DOUBLE_EQ(first.inference_ms(), expected_inf_ms);
  EXPECT_DOUBLE_EQ(first.baseline_inference_ms(), expected_base_inf_ms);

  DatasetManifest resolved;
  ASSERT_TRUE(ReadProto(root_ / "results/resolved_manifest.txtpb", resolved));
  EXPECT_TRUE(
      Utf8ToPath(resolved.samples(0).targets(0).mask().path()).is_absolute());
  EXPECT_TRUE(absl::StrContains(resolved.samples(0).targets(0).mask().path(),
                                "mask.raw"));

  RunSummary second;
  status = Run(second);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_EQ(second.resumed(), 3);
  EXPECT_EQ(FormatProto(second.metrics(0)), FormatProto(first.metrics(0)));
  EXPECT_EQ(FormatProto(second.identity().config()),
            FormatProto(first.identity().config()));
}

TEST_F(RunnerTest, BaselineInheritsCandidateBackendOptionsByDefault) {
  config_.mutable_backend()->set_num_threads(2);
  config_.mutable_backend()->set_xnnpack_only(true);
  config_.set_baseline_model(config_.model());
  config_.set_require_output_parity(true);

  RunSummary summary;
  Expected<void> status = Run(summary);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_EQ(summary.status(), PASSED);
  EXPECT_EQ(summary.identity().config().baseline_accelerator(), "inherit");
  EXPECT_EQ(summary.identity().baseline().options().num_threads(), 2);
  EXPECT_TRUE(summary.identity().baseline().options().xnnpack_only());
  ASSERT_EQ(summary.comparisons_size(), 2);
  EXPECT_TRUE(summary.comparisons(0).allclose());
  EXPECT_GT(summary.baseline_inference_ms(), 0.0);
}

TEST_F(RunnerTest, NormalsEvaluationResumeAndCorruptCheckpoints) {
  DatasetManifest normals_manifest;
  for (const auto& in : manifest_.inputs()) *normals_manifest.add_inputs() = in;
  auto* normals_out = normals_manifest.add_outputs();
  *normals_out->mutable_tensor() =
      MakeTensorSpec("normals", {1, 1, 1, 3}, "segmentation");
  normals_out->mutable_tensor()->set_layout(NHWC);
  normals_out->mutable_scoring()->set_task(NORMALS);
  auto* sample = normals_manifest.add_samples();
  sample->set_id("n0");
  for (const auto& spec : normals_manifest.inputs()) {
    auto* in = sample->add_inputs();
    in->set_name(spec.name());
    in->mutable_file()->set_path("input.raw");
  }
  auto* target = sample->add_targets();
  target->set_name("normals");
  target->mutable_file()->set_path("depth.raw");
  *target->mutable_tensor() = MakeTensorSpec("", FLOAT32, {3, 1, 1});
  const auto normals_manifest_path = root_ / "normals_manifest.txtpb";
  ASSERT_TRUE(WriteProto(normals_manifest_path, normals_manifest, TEXT_PROTO));

  config_.set_baseline_model(config_.model());
  config_.set_baseline_accelerator("cpu");
  config_.set_manifest(PathToUtf8(normals_manifest_path));
  config_.set_output_dir(PathToUtf8(root_ / "normals_results"));

  RunSummary first;
  auto status = Run(first);
  ASSERT_TRUE(status) << status.Error().Message();
  ASSERT_EQ(first.identity().candidate().outputs_size(), 2);
  EXPECT_EQ(first.identity().candidate().outputs(1).layout(), NHWC);
  ASSERT_EQ(first.identity().baseline().outputs_size(), 2);
  EXPECT_EQ(first.identity().baseline().outputs(1).layout(), NHWC);
  EXPECT_NEAR(first.metrics(0).normals().mean_degrees(), 0.0, 1e-5);

  RunSummary resumed;
  status = Run(resumed);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_EQ(resumed.resumed(), 1);

  const auto row_path = root_ / "normals_results/samples/n0/result.binpb";
  ExpectCorruptCheckpoint(
      row_path,
      [](SampleResult& r) {
        r.mutable_metrics(0)->mutable_depth()->set_images(1);
      },
      "Checkpoint task mismatch");
  ExpectCorruptCheckpoint(
      row_path,
      [](SampleResult& r) {
        r.mutable_metrics(0)->mutable_normals()->set_images(2);
      },
      "Invalid per-sample observation count");

  SampleResult row;
  ASSERT_TRUE(ReadProto(row_path, row));
  ASSERT_TRUE(row.metrics(0).has_normal_errors());
  std::error_code ec;
  ASSERT_TRUE(std::filesystem::remove(
      Utf8ToPath(row.metrics(0).normal_errors().path()), ec));
  ASSERT_FALSE(ec);
  RunSummary missing_spool;
  EXPECT_FALSE(Run(missing_spool));
  EXPECT_EQ(missing_spool.status(), FAILED);
}

TEST_F(RunnerTest, FailedSampleLeavesCompletedSamplesResumable) {
  manifest_.mutable_samples(1)->mutable_inputs(0)->mutable_file()->set_path(
      "missing.raw");
  ASSERT_TRUE(WriteProto(root_ / "manifest.txtpb", manifest_, TEXT_PROTO));
  RunSummary first;
  EXPECT_FALSE(Run(first));
  EXPECT_EQ(first.status(), FAILED);
  EXPECT_EQ(first.completed(), 1);
  RunSummary saved_failed;
  ASSERT_TRUE(ReadProto(root_ / "results/summary.txtpb", saved_failed));
  EXPECT_EQ(saved_failed.status(), FAILED);
  EXPECT_TRUE(
      PathExistsForTest(root_ / "results/samples/sample_0/result.binpb"));
  EXPECT_FALSE(
      PathExistsForTest(root_ / "results/samples/sample_1/result.binpb"));
  const float inputs[] = {1, 2, 3};
  ASSERT_TRUE(WriteBytes(root_ / "missing.raw", inputs, sizeof(inputs)));
  RunSummary second;
  auto status = Run(second);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_EQ(second.completed(), 3);
  EXPECT_EQ(second.resumed(), 1);
}

TEST_F(RunnerTest, ChangedProvenanceDoesNotOverwriteExistingSummary) {
  RunSummary first;
  auto status = Run(first);
  ASSERT_TRUE(status) << status.Error().Message();
  const auto summary_path = root_ / "results/summary.txtpb";
  const auto original = ReadFileText(summary_path);
  ASSERT_FALSE(original.empty());
  config_.set_rtol(0.2);
  ExpectRunError(config_, /*out_subdir=*/"", "Run provenance changed");
  EXPECT_EQ(original, ReadFileText(summary_path));
}

TEST_F(RunnerTest, MissingCompletedInputOrTruncatedArtifactIsRejected) {
  RunSummary first;
  auto status = Run(first);
  ASSERT_TRUE(status) << status.Error().Message();

  const float truncated = 1.0f;
  ASSERT_TRUE(WriteBytes(root_ / "results/samples/sample_0/depth.raw",
                         &truncated, sizeof(truncated)));
  RunSummary second;
  ExpectRunError(config_, /*out_subdir=*/"",
                 "Checkpoint tensor file size mismatch", &second);
  EXPECT_EQ(second.resumed(), 0);

  std::error_code ec;
  ASSERT_TRUE(std::filesystem::remove(root_ / "input.raw", ec));
  ASSERT_FALSE(ec);
  RunSummary third;
  EXPECT_FALSE(Run(third));
  EXPECT_EQ(third.status(), FAILED);
  EXPECT_EQ(third.resumed(), 0);
}

TEST_F(RunnerTest,
       QuantizedBaselinesWithDifferentZeroPointsReceiveSameRealInputs) {
  const float signed_truth[] = {-6.0f, 4.0f, 6.0f};
  ASSERT_TRUE(
      WriteBytes(root_ / "depth.raw", signed_truth, sizeof(signed_truth)));
  for (ElementType qtype : {INT8, UINT8}) {
    const bool is_uint8 = qtype == UINT8;
    const int zero = is_uint8 ? 128 : 7;
    const auto candidate = root_ / (is_uint8 ? "u8.tflite" : "i8.tflite");
    const auto baseline =
        root_ / (is_uint8 ? "base_u8.tflite" : "base_i8.tflite");
    ASSERT_NO_FATAL_FAILURE(WriteQuantizedModelCopy(candidate, qtype, zero));
    ASSERT_NO_FATAL_FAILURE(
        WriteQuantizedModelCopy(baseline, qtype, zero - 10));
    config_.set_model(PathToUtf8(candidate));
    config_.set_baseline_model(PathToUtf8(baseline));
    config_.set_baseline_accelerator("cpu");
    config_.set_require_output_parity(true);
    config_.set_output_dir(
        PathToUtf8(root_ / (is_uint8 ? "results_u8" : "results_i8")));
    SetManifestQuantization(manifest_, qtype, 0.25f, zero);
    const std::vector<uint8_t> input = {static_cast<uint8_t>(zero - 12),
                                        static_cast<uint8_t>(zero + 8),
                                        static_cast<uint8_t>(zero + 12)};
    ASSERT_TRUE(WriteBytes(root_ / "input.raw", input.data(), input.size()));
    ASSERT_TRUE(WriteProto(root_ / "manifest.txtpb", manifest_, TEXT_PROTO));
    RunSummary summary;
    auto status = Run(summary);
    ASSERT_TRUE(status) << status.Error().Message();
    ASSERT_EQ(summary.comparisons_size(), 2);
    EXPECT_TRUE(summary.comparisons(0).allclose());
    EXPECT_FALSE(summary.comparisons(0).has_max_code_error());
    EXPECT_TRUE(summary.comparisons(1).allclose());
    EXPECT_EQ(summary.metrics(0).depth().abs_rel(), 0);
  }

  const auto scale_base = root_ / "base_scale_u8.tflite";
  ASSERT_NO_FATAL_FAILURE(
      WriteQuantizedModelCopy(scale_base, UINT8, 128, 0.5f));
  config_.set_baseline_model(PathToUtf8(scale_base));
  config_.set_output_dir(PathToUtf8(root_ / "results_u8_scale"));
  RunSummary scale_summary;
  auto scale_status = Run(scale_summary);
  ASSERT_TRUE(scale_status) << scale_status.Error().Message();
  EXPECT_TRUE(scale_summary.comparisons(0).allclose());

  SampleResult i8_row;
  ASSERT_TRUE(
      ReadProto(root_ / "results_i8/samples/sample_0/result.binpb", i8_row));
  ASSERT_GE(i8_row.predictions_size(), 1);
  EXPECT_EQ(i8_row.predictions(0).tensor().type(), FLOAT32);
  EXPECT_FALSE(i8_row.predictions(0).tensor().has_scale());
  EXPECT_FALSE(i8_row.predictions(0).tensor().has_zero_point());

  EvaluationConfig offline_q;
  offline_q.set_score_only(true);
  offline_q.set_manifest(PathToUtf8(root_ / "manifest.txtpb"));
  offline_q.set_predictions_dir(PathToUtf8(root_ / "results_i8/samples"));
  offline_q.set_output_dir(PathToUtf8(root_ / "offline_quant_manifest"));
  RunSummary offline_q_summary;
  auto offline_q_status =
      Evaluate(offline_q, Utf8ToPath(config_.model()), offline_q_summary);
  ASSERT_TRUE(offline_q_status) << offline_q_status.Error().Message();
  EXPECT_EQ(offline_q_summary.metrics(0).depth().abs_rel(), 0);
  SampleResult off_q_row;
  ASSERT_TRUE(ReadProto(
      root_ / "offline_quant_manifest/samples/sample_0/result.binpb",
      off_q_row));
  EXPECT_FALSE(off_q_row.predictions(0).tensor().has_zero_point());
  EXPECT_FALSE(off_q_row.predictions(0).file().path().empty());
}

TEST_F(RunnerTest, NonfinitePreparedInputsFailBeforeInvocation) {
  const float input[] = {std::numeric_limits<float>::quiet_NaN(), 2, 3};
  ASSERT_TRUE(WriteBytes(root_ / "input.raw", input, sizeof(input)));
  RunSummary summary;
  ExpectRunError(config_, /*out_subdir=*/"", "Nonfinite prepared input",
                 &summary);
  EXPECT_EQ(summary.completed(), 0);
  EXPECT_FALSE(
      PathExistsForTest(root_ / "results/samples/sample_0/result.binpb"));
}

TEST_F(RunnerTest, SemanticallyCorruptCheckpointIsRejected) {
  config_.set_baseline_model(config_.model());
  config_.set_baseline_accelerator("cpu");
  RunSummary first;
  auto status = Run(first);
  ASSERT_TRUE(status) << status.Error().Message();
  const auto path = root_ / "results/samples/sample_0/result.binpb";

  ExpectCorruptCheckpoint(
      path, [](SampleResult& r) { r.set_complete(false); },
      "Checkpoint identity or completion mismatch");
  ExpectCorruptCheckpoint(
      path, [](SampleResult& r) { r.set_version(2); },
      "Checkpoint identity or completion mismatch");
  ExpectCorruptCheckpoint(
      path, [](SampleResult& r) { r.set_id("wrong_id"); },
      "Checkpoint identity or completion mismatch");
  ExpectCorruptCheckpoint(
      path, [](SampleResult& r) { r.clear_sources(); },
      "Checkpoint identity or completion mismatch");
  ExpectCorruptCheckpoint(
      path, [](SampleResult& r) { r.clear_metrics(); },
      "Checkpoint identity or completion mismatch");
  ExpectCorruptCheckpoint(
      path,
      [](SampleResult& r) { r.mutable_sources(0)->set_path("other.raw"); },
      "Completed sample data changed");
  ExpectCorruptCheckpoint(
      path, [](SampleResult& r) { r.clear_baseline_metrics(); },
      "Checkpoint metric count mismatch");
  ExpectCorruptCheckpoint(
      path, [](SampleResult& r) { r.mutable_metrics(0)->set_name("unknown"); },
      "Checkpoint metric output mismatch");
  ExpectCorruptCheckpoint(
      path,
      [](SampleResult& r) {
        r.mutable_metrics(1)->set_name(r.metrics(0).name());
        *r.mutable_metrics(1)->mutable_depth() = r.metrics(0).depth();
      },
      "Checkpoint metric output mismatch");
  ExpectCorruptCheckpoint(
      path,
      [](SampleResult& r) {
        r.mutable_metrics(0)->mutable_classification()->set_images(1);
      },
      "Checkpoint task mismatch");
  ExpectCorruptCheckpoint(
      path,
      [](SampleResult& r) {
        r.mutable_metrics(1)->mutable_classification()->set_images(1);
      },
      "Checkpoint task mismatch");
  ExpectCorruptCheckpoint(
      path,
      [](SampleResult& r) {
        r.mutable_metrics(0)->mutable_depth()->set_images(2);
      },
      "Invalid per-sample observation count");
  ExpectCorruptCheckpoint(
      path,
      [](SampleResult& r) {
        r.mutable_metrics(1)->mutable_segmentation()->set_images(2);
      },
      "Invalid per-sample observation count");
  ExpectCorruptCheckpoint(
      path, [](SampleResult& r) { r.clear_predictions(); },
      "Checkpoint prediction count mismatch");
  ExpectCorruptCheckpoint(
      path,
      [](SampleResult& r) {
        r.mutable_predictions(0)->mutable_tensor()->set_type(INT8);
      },
      "Checkpoint tensor contract mismatch");
  ExpectCorruptCheckpoint(
      path,
      [](SampleResult& r) {
        r.mutable_predictions(0)->mutable_tensor()->set_name("wrong");
      },
      "Checkpoint tensor contract mismatch");
  ExpectCorruptCheckpoint(
      path, [](SampleResult& r) { r.clear_comparisons(); },
      "Checkpoint comparison count mismatch");
  ExpectCorruptCheckpoint(
      path, [](SampleResult& r) { r.mutable_comparisons(0)->set_elements(0); },
      "Invalid checkpoint comparison");
  ExpectCorruptCheckpoint(
      path,
      [](SampleResult& r) { r.mutable_comparisons(0)->set_max_abs(-1.0); },
      "Invalid checkpoint comparison");
  ExpectCorruptCheckpoint(
      path,
      [](SampleResult& r) { r.mutable_comparisons(0)->set_mean_abs(-1.0); },
      "Invalid checkpoint comparison");
  ExpectCorruptCheckpoint(
      path,
      [](SampleResult& r) {
        r.mutable_comparisons(0)->set_max_abs(
            std::numeric_limits<double>::quiet_NaN());
      },
      "Invalid checkpoint comparison");
  ExpectCorruptCheckpoint(
      path,
      [](SampleResult& r) {
        r.mutable_comparisons(0)->set_mean_abs(
            std::numeric_limits<double>::infinity());
      },
      "Invalid checkpoint comparison");
  ExpectCorruptCheckpoint(
      path,
      [](SampleResult& r) {
        r.mutable_comparisons(1)->set_name(r.comparisons(0).name());
      },
      "Invalid checkpoint comparison");
  ExpectCorruptCheckpoint(
      path, [](SampleResult& r) { r.set_inference_ms(-1.0); },
      "Invalid checkpoint timing");
  ExpectCorruptCheckpoint(
      path,
      [](SampleResult& r) {
        r.set_inference_ms(std::numeric_limits<double>::quiet_NaN());
      },
      "Invalid checkpoint timing");
  ExpectCorruptCheckpoint(
      path, [](SampleResult& r) { r.set_baseline_inference_ms(-1.0); },
      "Invalid checkpoint timing");
  ExpectCorruptCheckpoint(
      path,
      [](SampleResult& r) {
        r.set_baseline_inference_ms(std::numeric_limits<double>::infinity());
      },
      "Invalid checkpoint timing");
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
  status = Evaluate(offline, Utf8ToPath(config_.model()), scored);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_EQ(scored.status(), PASSED);
  EXPECT_EQ(scored.completed(), 3);
  EXPECT_EQ(scored.metrics_size(), 2);
  EXPECT_EQ(scored.baseline_metrics_size(), 2);
  ASSERT_EQ(scored.comparisons_size(), 2);
  EXPECT_TRUE(scored.comparisons(0).allclose());
  EXPECT_EQ(FormatProto(scored.metrics(0)), FormatProto(online.metrics(0)));

  RunSummary resumed;
  status = Evaluate(offline, Utf8ToPath(config_.model()), resumed);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_EQ(resumed.resumed(), 3);

  const auto mixed_preds = root_ / "mixed_preds";
  std::error_code ec;
  std::filesystem::copy(root_ / "results/samples", mixed_preds,
                        std::filesystem::copy_options::recursive, ec);
  ASSERT_FALSE(ec);
  const float drifted[] = {10.0f, 4.0f, 6.0f};
  ASSERT_TRUE(WriteBytes(mixed_preds / "sample_0/depth.raw", drifted,
                         sizeof(drifted)));
  offline.set_predictions_dir(PathToUtf8(mixed_preds));
  offline.set_atol(1e-5);  // NOLINT(runtime/deprecated_fn)
  RunSummary mixed_summary;
  ExpectRunError(offline, "offline_mixed_parity",
                 "Output parity tolerance exceeded", &mixed_summary);
  ASSERT_EQ(mixed_summary.comparisons_size(), 2);
  EXPECT_FALSE(mixed_summary.comparisons(0).allclose());
  EXPECT_DOUBLE_EQ(mixed_summary.comparisons(0).max_abs(), 8.0);
  EXPECT_TRUE(mixed_summary.comparisons(1).allclose());
}

TEST_F(RunnerTest, TsvClassificationAutoBindingInt8CodesAndComparisonGates) {
  ClassificationDatasetFixture ds;
  ASSERT_NO_FATAL_FAILURE(SetupClassificationDataset(ds));

  RunSummary summary;
  auto status = Evaluate(ds.config, ds.cand_path, summary);
  ASSERT_TRUE(status) << status.Error().Message();
  ASSERT_EQ(summary.metrics_size(), 1);
  EXPECT_DOUBLE_EQ(summary.metrics(0).classification().top1(), 0.5);
  EXPECT_DOUBLE_EQ(summary.metrics(0).classification().top5(), 1.0);
  ASSERT_EQ(summary.comparisons_size(), 1);
  EXPECT_TRUE(summary.comparisons(0).allclose());
  EXPECT_EQ(summary.comparisons(0).elements(), 24u);
  EXPECT_NEAR(summary.comparisons(0).max_abs(), 0.15, 1e-5);
  EXPECT_NEAR(summary.comparisons(0).mean_abs(), 0.15 / 12.0, 1e-5);
  EXPECT_EQ(summary.comparisons(0).max_code_error(), 1u);
  EXPECT_NEAR(summary.comparisons(0).mean_code_error(), 1.0 / 12.0, 1e-5);
  EXPECT_TRUE(summary.comparisons(0).prediction_agreement());

  DatasetManifest auto_bound;
  ASSERT_TRUE(
      ReadProto(root_ / "cls_results/resolved_manifest.txtpb", auto_bound));
  ASSERT_EQ(auto_bound.inputs_size(), 1);
  EXPECT_EQ(auto_bound.inputs(0).name(), "image");
  EXPECT_EQ(auto_bound.inputs(0).binding_name(), "pixels");
  EXPECT_EQ(auto_bound.inputs(0).layout(), NHWC);
  ASSERT_EQ(auto_bound.outputs_size(), 1);
  EXPECT_EQ(auto_bound.outputs(0).tensor().name(), "output");
  EXPECT_EQ(auto_bound.outputs(0).tensor().binding_name(), "logits");
  EXPECT_TRUE(auto_bound.outputs(0).scoring().has_task());
  EXPECT_EQ(auto_bound.outputs(0).scoring().task(), CLASSIFICATION);

  RunSummary cls_resumed;
  status = Evaluate(ds.config, ds.cand_path, cls_resumed);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_EQ(cls_resumed.resumed(), 2);

  const auto cls_row_path = root_ / "cls_results/samples/0/result.binpb";
  ExpectCorruptCheckpoint(
      ds.config, cls_row_path,
      [](SampleResult& r) {
        r.mutable_metrics(0)->mutable_depth()->set_images(1);
      },
      "Checkpoint task mismatch");
  ExpectCorruptCheckpoint(
      ds.config, cls_row_path,
      [](SampleResult& r) {
        r.mutable_metrics(0)->mutable_classification()->set_images(2);
      },
      "Invalid per-sample observation count");

  EvaluationConfig code_gate_cfg = ds.config;
  code_gate_cfg.set_atol(10.0);  // NOLINT(runtime/deprecated_fn)
  code_gate_cfg.set_max_code_error(0);
  ExpectRunError(code_gate_cfg, "cls_code_gate_fail",
                 "Output parity tolerance exceeded");

  EvaluationConfig disagree_cfg = ds.config;
  disagree_cfg.set_baseline_model(PathToUtf8(ds.base_disagree_path));
  disagree_cfg.set_require_output_parity(false);
  disagree_cfg.set_require_prediction_agreement(true);
  RunSummary disagree_summary;
  ExpectRunError(disagree_cfg, "cls_disagree_gate",
                 "Prediction agreement gate failed", &disagree_summary);
  ASSERT_EQ(disagree_summary.comparisons_size(), 1);
  EXPECT_FALSE(disagree_summary.comparisons(0).prediction_agreement());

  EvaluationConfig parity_cfg = ds.config;
  parity_cfg.set_atol(1e-5);  // NOLINT(runtime/deprecated_fn)
  parity_cfg.set_compare_int8_codes(false);
  parity_cfg.set_require_output_parity(true);
  parity_cfg.set_require_prediction_agreement(false);
  ExpectRunError(parity_cfg, "cls_parity_gate",
                 "Output parity tolerance exceeded");
}

TEST_F(RunnerTest, EncodedImageLayoutAutoBindingAndShapeValidation) {
  ClassificationDatasetFixture ds;
  ASSERT_NO_FATAL_FAILURE(SetupClassificationDataset(ds));

  const auto nchw_path = root_ / "cls_nchw.tflite";
  std::vector<float> nchw_bias(24, 0.0f);
  nchw_bias[2] = 5.0f;
  nchw_bias[5] = 4.0f;
  ASSERT_NO_FATAL_FAILURE(
      WriteClassificationModel(nchw_path, FLOAT32, nchw_bias, {1, 3, 2, 4}));

  EvaluationConfig single_cfg = ds.config;
  single_cfg.set_model(PathToUtf8(nchw_path));
  single_cfg.mutable_preprocessing()->set_resize(4);
  single_cfg.mutable_preprocessing()->set_crop_h(2);
  single_cfg.mutable_preprocessing()->set_crop_w(4);
  single_cfg.set_task(CLASSIFICATION);
  single_cfg.clear_baseline_model();
  single_cfg.clear_baseline_accelerator();
  single_cfg.set_compare_int8_codes(false);
  single_cfg.set_require_output_parity(false);
  single_cfg.set_require_prediction_agreement(false);
  single_cfg.set_output_dir(PathToUtf8(root_ / "cls_nchw_results"));

  RunSummary nchw_summary;
  auto status = Evaluate(single_cfg, nchw_path, nchw_summary);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_FALSE(nchw_summary.identity().has_baseline());
  DatasetManifest nchw_bound;
  ASSERT_TRUE(ReadProto(root_ / "cls_nchw_results/resolved_manifest.txtpb",
                        nchw_bound));
  EXPECT_EQ(nchw_bound.inputs(0).layout(), NCHW);

  const struct {
    const char* name;
    std::vector<int32_t> shape;
    size_t elements;
    const char* expected_error;
  } bad_image_models[] = {
      {"cls_batch2",
       {2, 2, 2, 3},
       24,
       "Image input must be unambiguous batch-one NCHW/NHWC"},
      {"cls_ambiguous", {1, 3, 3, 3}, 27, "Image input must be unambiguous"},
      {"cls_rank1",
       {3},
       3,
       "Image input must be unambiguous batch-one NCHW/NHWC"},
  };
  for (const auto& tc : bad_image_models) {
    const auto model_path = root_ / absl::StrCat(tc.name, ".tflite");
    ASSERT_NO_FATAL_FAILURE(WriteClassificationModel(
        model_path, FLOAT32, std::vector<float>(tc.elements, 0.0f), tc.shape));
    EvaluationConfig bad_cfg = single_cfg;
    bad_cfg.set_model(PathToUtf8(model_path));
    ExpectRunError(bad_cfg, absl::StrCat(tc.name, "_out"), tc.expected_error);
  }

  DatasetManifest two_in_image;
  auto* two_out = two_in_image.add_outputs();
  *two_out->mutable_tensor() = MakeTensorSpec("depth", {1, 1, 1, 3});
  two_out->mutable_scoring()->set_task(CLASSIFICATION);
  auto* img_sample = two_in_image.add_samples();
  img_sample->set_id("img0");
  img_sample->set_label(0);
  img_sample->set_image("sample.ppm");
  EvaluationConfig two_in_cfg = single_cfg;
  two_in_cfg.set_model(PathToUtf8(root_ / "model.tflite"));
  ExpectManifestError(two_in_cfg, two_in_image, "two_in_img",
                      "Encoded images require one model input");
  EXPECT_FALSE(PathExistsForTest(root_ / "out_two_in_img/identity.binpb"));

  DatasetManifest two_in_explicit = two_in_image;
  for (const TensorSpec& in : manifest_.inputs()) {
    *two_in_explicit.add_inputs() = in;
  }
  ExpectManifestError(two_in_cfg, two_in_explicit, "two_in_explicit",
                      "Encoded images require one model input");
  EXPECT_TRUE(PathExistsForTest(root_ / "out_two_in_explicit/identity.binpb"));

  EvaluationConfig bad_crop_h = ds.config;
  bad_crop_h.mutable_preprocessing()->set_resize(4);
  bad_crop_h.mutable_preprocessing()->set_crop_h(4);
  bad_crop_h.mutable_preprocessing()->set_crop_w(2);
  ExpectRunError(bad_crop_h, "cls_bad_crop_h",
                 "Preprocessing crop does not match the model input shape");

  EvaluationConfig bad_crop_w = bad_crop_h;
  bad_crop_w.mutable_preprocessing()->set_crop_h(2);
  bad_crop_w.mutable_preprocessing()->set_crop_w(4);
  ExpectRunError(bad_crop_w, "cls_bad_crop_w",
                 "Preprocessing crop does not match the model input shape");
}

TEST_F(RunnerTest, TaskOverrideSupplyQuantizationAndFloatToInt8Baseline) {
  const auto fp32_path = root_ / "cls_f32.tflite";
  const auto int8_path = root_ / "cls_q8.tflite";
  ASSERT_NO_FATAL_FAILURE(WriteClassificationModel(
      fp32_path, FLOAT32, std::vector<float>(12, 0.0f)));
  ASSERT_NO_FATAL_FAILURE(
      WriteClassificationModel(int8_path, INT8, std::vector<float>(12, 0.0f)));

  const std::vector<float> f32_input = {0, 0, 5.0f, 0, 0, 0,
                                        0, 0, 0,    0, 0, 0};
  ASSERT_TRUE(WriteBytes(root_ / "cls_in_f32.raw", f32_input.data(),
                         f32_input.size() * sizeof(float)));
  const auto f32_out_spec = MakeTensorSpec("logits", FLOAT32, {1, 2, 2, 3});
  const auto f32_manifest = MakeSingleInputManifest(
      MakeTensorSpec("image", FLOAT32, {1, 2, 2, 3}), "cls_in_f32.raw",
      /*label=*/2, &f32_out_spec);
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
  auto status = Evaluate(cfg, fp32_path, f32_summary);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_DOUBLE_EQ(f32_summary.metrics(0).classification().top1(), 1.0);
  EXPECT_DOUBLE_EQ(f32_summary.baseline_metrics(0).classification().top1(),
                   1.0);

  // Bytes 4..7 form 0xFFFFFFFF (quiet NaN if misinterpreted as FP32), while
  // signed int8 code -1 yields logit -0.25 < +5.0 at class 2.
  const uint8_t i8_signal[] = {0, 0, 20, 0, 255, 255, 255, 255, 0, 0, 0, 0};
  ASSERT_TRUE(
      WriteBytes(root_ / "cls_in_i8.raw", i8_signal, sizeof(i8_signal)));
  const auto i8_out_spec = MakeTensorSpec("logits", INT8, {1, 2, 2, 3});
  const auto i8_manifest = MakeSingleInputManifest(
      MakeTensorSpec("image", INT8, {1, 2, 2, 3}), "cls_in_i8.raw",
      /*label=*/2, &i8_out_spec, CLASSIFICATION);
  const auto i8_manifest_path = root_ / "cls_i8_manifest.txtpb";
  ASSERT_TRUE(WriteProto(i8_manifest_path, i8_manifest, TEXT_PROTO));

  EvaluationConfig i8_cfg;
  i8_cfg.set_model(PathToUtf8(int8_path));
  i8_cfg.set_manifest(PathToUtf8(i8_manifest_path));
  i8_cfg.set_output_dir(PathToUtf8(root_ / "supply_quant_out"));
  i8_cfg.mutable_backend()->set_accelerator("cpu");
  RunSummary i8_summary;
  status = Evaluate(i8_cfg, int8_path, i8_summary);
  ASSERT_TRUE(status) << status.Error().Message();
  EXPECT_DOUBLE_EQ(i8_summary.metrics(0).classification().top1(), 1.0);

  DatasetManifest supplied;
  ASSERT_TRUE(
      ReadProto(root_ / "supply_quant_out/resolved_manifest.txtpb", supplied));
  EXPECT_FLOAT_EQ(supplied.inputs(0).scale(), 0.25f);
  EXPECT_FLOAT_EQ(supplied.outputs(0).tensor().scale(), 0.25f);

  DatasetManifest depth_override;
  for (const auto& in : manifest_.inputs()) *depth_override.add_inputs() = in;
  *depth_override.add_outputs()->mutable_tensor() =
      MakeTensorSpec("depth", {1, 1, 1, 3});
  auto* depth_sample = depth_override.add_samples();
  depth_sample->set_id("d0");
  for (const auto& spec : depth_override.inputs()) {
    auto* in = depth_sample->add_inputs();
    in->set_name(spec.name());
    in->mutable_file()->set_path("input.raw");
  }
  auto* depth_target = depth_sample->add_targets();
  depth_target->set_name("depth");
  depth_target->mutable_file()->set_path("depth.raw");
  *depth_target->mutable_tensor() = MakeTensorSpec("", FLOAT32, {1, 3});
  const auto depth_override_path = root_ / "depth_override.txtpb";
  ASSERT_TRUE(WriteProto(depth_override_path, depth_override, TEXT_PROTO));

  EvaluationConfig override_cfg = config_;
  override_cfg.set_manifest(PathToUtf8(depth_override_path));
  override_cfg.set_task(DEPTH);
  override_cfg.set_output_dir(PathToUtf8(root_ / "depth_override_out"));
  RunSummary override_summary;
  status =
      Evaluate(override_cfg, Utf8ToPath(config_.model()), override_summary);
  ASSERT_TRUE(status) << status.Error().Message();
  ASSERT_EQ(override_summary.metrics_size(), 1);
  EXPECT_TRUE(override_summary.metrics(0).has_depth());

  for (bool cls_second : {false, true}) {
    DatasetManifest partial = manifest_;
    partial.mutable_samples()->DeleteSubrange(1, 2);
    partial.mutable_samples(0)->mutable_targets()->DeleteSubrange(1, 1);
    if (cls_second) {
      partial.mutable_outputs(1)->mutable_scoring()->set_task(CLASSIFICATION);
      partial.mutable_samples(0)->clear_label();
    } else {
      partial.mutable_samples(0)->set_label(0);
    }
    const auto partial_path =
        root_ / (cls_second ? "partial_cls.txtpb" : "partial_seg.txtpb");
    ASSERT_TRUE(WriteProto(partial_path, partial, TEXT_PROTO));
    EvaluationConfig partial_cfg = config_;
    partial_cfg.set_manifest(PathToUtf8(partial_path));
    partial_cfg.set_output_dir(PathToUtf8(
        root_ / (cls_second ? "partial_cls_out" : "partial_seg_out")));
    RunSummary p_first, p_resumed;
    ASSERT_TRUE(Evaluate(partial_cfg, Utf8ToPath(config_.model()), p_first));
    EXPECT_EQ(p_first.metrics_size(), 1);
    ASSERT_TRUE(Evaluate(partial_cfg, Utf8ToPath(config_.model()), p_resumed));
    EXPECT_EQ(p_resumed.resumed(), 1);
    EXPECT_EQ(p_resumed.metrics_size(), 1);
  }
}

TEST_F(RunnerTest, ExtremeQuantizationScaleSkipsRedundantInputRoundTrip) {
  const auto extreme_model = root_ / "extreme_i8.tflite";
  ASSERT_NO_FATAL_FAILURE(WriteQuantizedModelCopy(
      extreme_model, INT8, /*zero_point=*/0, /*scale=*/1e37f));
  const int8_t pos[] = {100, 0, 0};
  const int8_t neg[] = {-100, 0, 0};
  const float empty_depth[] = {-1.0f, -1.0f, -1.0f};
  ASSERT_TRUE(WriteBytes(root_ / "pos.raw", pos, sizeof(pos)));
  ASSERT_TRUE(WriteBytes(root_ / "neg.raw", neg, sizeof(neg)));
  ASSERT_TRUE(
      WriteBytes(root_ / "depth.raw", empty_depth, sizeof(empty_depth)));
  SetManifestQuantization(manifest_, INT8, 1e37f, 0);
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
      Evaluate(EvaluationConfig{}, Utf8ToPath(config_.model()), bad_cfg));
  EXPECT_EQ(bad_cfg.status(), FAILED);
  EXPECT_FALSE(bad_cfg.error().empty());

  EvaluationConfig bad_outdir = config_;
  bad_outdir.set_output_dir(PathToUtf8(root_ / "input.raw"));
  ExpectRunError(bad_outdir, /*out_subdir=*/"", "Cannot create output_dir");

  const std::string tsv = "0\t1\tinput.raw\n";
  const auto tsv_path = root_ / "offline_bad.tsv";
  ASSERT_TRUE(WriteBytes(tsv_path, tsv.data(), tsv.size()));
  EvaluationConfig offline_no_out;
  offline_no_out.set_score_only(true);
  offline_no_out.set_manifest(PathToUtf8(tsv_path));
  offline_no_out.set_predictions_dir(PathToUtf8(root_));
  ExpectRunError(offline_no_out, "out_offline_no_contract",
                 "Dense/offline evaluation requires output contracts");

  EvaluationConfig two_out_no_contract = config_;
  two_out_no_contract.set_manifest(PathToUtf8(tsv_path));
  ExpectRunError(two_out_no_contract, "out_two_out_no_contract",
                 "Dense/offline evaluation requires output contracts");

  const auto one_out_model = root_ / "one_out_cls.tflite";
  ASSERT_NO_FATAL_FAILURE(WriteClassificationModel(
      one_out_model, FLOAT32, std::vector<float>(12, 0.0f)));
  const std::vector<float> raw_cls_in(12, 0.0f);
  ASSERT_TRUE(WriteBytes(root_ / "one_out_in.raw", raw_cls_in.data(),
                         raw_cls_in.size() * sizeof(float)));
  EvaluationConfig dense_no_contract = config_;
  dense_no_contract.set_model(PathToUtf8(one_out_model));
  dense_no_contract.set_task(DEPTH);
  ExpectManifestError(
      dense_no_contract,
      MakeSingleInputManifest(MakeTensorSpec("image", FLOAT32, {1, 2, 2, 3}),
                              "one_out_in.raw", /*label=*/0),
      "no_out_manifest", "Dense/offline evaluation requires output contracts");

  auto bad_binding = manifest_;
  bad_binding.mutable_outputs(0)->mutable_tensor()->set_binding_name("missing");
  ExpectManifestError(bad_binding, "bad_binding",
                      "Model has no signature tensor named missing");

  auto dup_out = manifest_;
  dup_out.mutable_outputs(1)->mutable_tensor()->set_binding_name("depth");
  ExpectManifestError(dup_out, "dup_out", "Duplicate output binding");

  auto dup_in = manifest_;
  dup_in.mutable_inputs(1)->set_binding_name("left");
  ExpectManifestError(dup_in, "dup_in", "Duplicate input binding");

  auto bad_rank = manifest_;
  *bad_rank.mutable_outputs(0)->mutable_tensor() =
      MakeTensorSpec("depth", {1, 1, 1});
  ExpectManifestError(bad_rank, "bad_rank", "Tensor contract mismatch: depth");

  auto bad_shape = manifest_;
  bad_shape.mutable_outputs(0)->mutable_tensor()->set_shape(3, 99);
  ExpectManifestError(bad_shape, "bad_shape",
                      "Tensor contract mismatch: depth");

  const auto q_model = root_ / "q_mismatch.tflite";
  ASSERT_NO_FATAL_FAILURE(WriteQuantizedModelCopy(q_model, INT8, 0));
  EvaluationConfig q_cfg = config_;
  q_cfg.set_model(PathToUtf8(q_model));
  ExpectRunError(q_cfg, "out_bad_type", "Tensor contract mismatch: depth");

  const uint8_t i8_bytes[] = {1, 2, 3};
  ASSERT_TRUE(WriteBytes(root_ / "i8_3b.raw", i8_bytes, sizeof(i8_bytes)));
  auto bad_scale = manifest_;
  for (auto& sample : *bad_scale.mutable_samples()) {
    for (auto& in : *sample.mutable_inputs()) {
      in.mutable_file()->set_path("i8_3b.raw");
    }
  }
  SetManifestQuantization(bad_scale, INT8, /*in_scale=*/0.5f,
                          /*in_zero_point=*/0, /*out_scale=*/0.25f,
                          /*out_zero_point=*/0);
  ExpectManifestError(q_cfg, bad_scale, "bad_scale",
                      "Tensor quantization mismatch: data_left");

  auto bad_zp = bad_scale;
  SetManifestQuantization(bad_zp, INT8, /*in_scale=*/0.25f,
                          /*in_zero_point=*/3, /*out_scale=*/0.25f,
                          /*out_zero_point=*/0);
  ExpectManifestError(q_cfg, bad_zp, "bad_zp",
                      "Tensor quantization mismatch: data_left");

  auto one_input = manifest_;
  one_input.mutable_inputs()->DeleteSubrange(1, 1);
  for (auto& sample : *one_input.mutable_samples()) {
    sample.mutable_inputs()->DeleteSubrange(1, 1);
  }
  ExpectManifestError(one_input, "one_input",
                      "Manifest must describe every model input");

  DatasetManifest three_inputs = manifest_;
  *three_inputs.add_inputs() = MakeTensorSpec("extra", {1, 1, 1, 3});
  for (Sample& sample : *three_inputs.mutable_samples()) {
    InputFile* extra = sample.add_inputs();
    extra->set_name("extra");
    extra->mutable_file()->set_path("input.raw");
  }
  ExpectManifestError(three_inputs, "three_inputs",
                      "Manifest must describe every model input");

  DatasetManifest untargeted;
  for (const auto& in : manifest_.inputs()) *untargeted.add_inputs() = in;
  auto* depth_out = untargeted.add_outputs();
  *depth_out->mutable_tensor() = MakeTensorSpec("depth", {1, 1, 1, 3});
  depth_out->mutable_scoring()->set_task(DEPTH);
  auto* only_cls_label = untargeted.add_samples();
  only_cls_label->set_id("s0");
  only_cls_label->set_label(0);
  for (const auto& spec : untargeted.inputs()) {
    auto* in = only_cls_label->add_inputs();
    in->set_name(spec.name());
    in->mutable_file()->set_path("input.raw");
  }
  ExpectManifestError(untargeted, "untargeted",
                      "Sample has no applicable output targets");
}

TEST_F(RunnerTest, MainHandlesHelpInvalidFlagsSuccessAndFailure) {
  testing::internal::CaptureStderr();
  EXPECT_EQ(CallMain({"litert_vision_eval", "--help"}), 0);
  EXPECT_TRUE(absl::StrContains(testing::internal::GetCapturedStderr(),
                                "litert_vision_eval"));
  EXPECT_EQ(CallMain({"litert_vision_eval", "-h"}), 0);
  testing::internal::CaptureStderr();
  EXPECT_EQ(CallMain({"litert_vision_eval", "invalid_positional"}), 1);
  const std::string err_text = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(absl::StrContains(err_text, "Expected --name=value"));
  EXPECT_TRUE(absl::StrContains(err_text, "litert_vision_eval"));

  const auto cli_out = root_ / "cli_out";
  EXPECT_EQ(
      CallMain({"litert_vision_eval",
                absl::StrCat("--model=", PathToUtf8(root_ / "model.tflite")),
                absl::StrCat("--manifest=",
                             PathToUtf8(root_ / "manifest.txtpb")),
                absl::StrCat("--output_dir=", PathToUtf8(cli_out))}),
      0);
  EXPECT_TRUE(PathExistsForTest(cli_out / "summary.txtpb"));

  const auto cli_bin_out = root_ / "cli_bin_out";
  EXPECT_EQ(
      CallMain({"litert_vision_eval",
                absl::StrCat("--model=", PathToUtf8(root_ / "model.tflite")),
                absl::StrCat("--manifest=",
                             PathToUtf8(root_ / "manifest.txtpb")),
                "--summary_format=binary",
                absl::StrCat("--output_dir=", PathToUtf8(cli_bin_out))}),
      0);
  EXPECT_TRUE(PathExistsForTest(cli_bin_out / "summary.binpb"));

  const auto cli_fail_out = root_ / "cli_fail_out";
  EXPECT_EQ(
      CallMain({"litert_vision_eval",
                absl::StrCat("--model=", PathToUtf8(root_ / "model.tflite")),
                absl::StrCat("--manifest=",
                             PathToUtf8(root_ / "nonexistent.txtpb")),
                absl::StrCat("--output_dir=", PathToUtf8(cli_fail_out))}),
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

TEST_F(RunnerTest, MainHandlesSigintSigtermAndResumesCleanly) {
  const auto irq_out = root_ / "irq_out";
  const std::vector<std::string> args = {
      "litert_vision_eval",
      absl::StrCat("--model=", PathToUtf8(root_ / "model.tflite")),
      absl::StrCat("--manifest=", PathToUtf8(root_ / "manifest.txtpb")),
      absl::StrCat("--output_dir=", PathToUtf8(irq_out)),
  };

  const InterruptedMainResult first_irq =
      RunMainInterruptedBySignal(args, SIGINT, 1);
  EXPECT_EQ(first_irq.exit_code, 1);
  EXPECT_TRUE(first_irq.sigint_restored);
  EXPECT_TRUE(first_irq.sigterm_restored);
  EXPECT_TRUE(
      absl::StrContains(first_irq.stderr_text, "Evaluation interrupted"))
      << first_irq.stderr_text;
  RunSummary first_summary;
  ASSERT_TRUE(ReadProto(irq_out / "summary.txtpb", first_summary));
  EXPECT_EQ(first_summary.status(), FAILED);
  EXPECT_EQ(first_summary.completed(), 1);
  EXPECT_EQ(first_summary.resumed(), 0);

  const InterruptedMainResult second_irq =
      RunMainInterruptedBySignal(args, SIGTERM, 2);
  EXPECT_EQ(second_irq.exit_code, 1);
  EXPECT_TRUE(second_irq.sigint_restored);
  EXPECT_TRUE(second_irq.sigterm_restored);
  EXPECT_TRUE(
      absl::StrContains(second_irq.stderr_text, "Evaluation interrupted"))
      << second_irq.stderr_text;
  RunSummary second_summary;
  ASSERT_TRUE(ReadProto(irq_out / "summary.txtpb", second_summary));
  EXPECT_EQ(second_summary.status(), FAILED);
  EXPECT_EQ(second_summary.completed(), 2);
  EXPECT_EQ(second_summary.resumed(), 1);

  testing::internal::CaptureStderr();
  EXPECT_EQ(CallMain(args), 0);
  EXPECT_TRUE(
      absl::StrContains(testing::internal::GetCapturedStderr(), "3/3"));
  RunSummary final_summary;
  ASSERT_TRUE(ReadProto(irq_out / "summary.txtpb", final_summary));
  EXPECT_EQ(final_summary.status(), PASSED);
  EXPECT_EQ(final_summary.completed(), 3);
  EXPECT_EQ(final_summary.resumed(), 2);
  EXPECT_FALSE(final_summary.identity().binary().path().empty());
}

}  // namespace
}  // namespace litert::vision
