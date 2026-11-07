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

#ifndef LITERT_TOOLS_VISION_EVAL_RUNNER_TEST_UTIL_H_
#define LITERT_TOOLS_VISION_EVAL_RUNNER_TEST_UTIL_H_

#include <array>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <optional>
#include <streambuf>
#include <string>

#include <gtest/gtest.h>
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"
#include "litert/tools/vision_eval/evaluation.pb.h"
#include "google/protobuf/message.h"  // from @com_google_protobuf
#include "tflite/converter/schema/schema_generated.h"

namespace litert::vision {

// Returns true if `path` exists on disk without filesystem errors.
bool PathExistsForTest(const std::filesystem::path& path);

// Reads and returns the raw contents of `path` as a string.
std::string ReadFileText(const std::filesystem::path& path);

// Formats `message` as canonical textproto for human-readable test equality
// assertions.
std::string FormatProto(const google::protobuf::Message& message);

// Invokes `litert::vision::Main` with mutable `argv` pointers copied from
// `args`.
int CallMain(absl::Span<const std::string> args);

// Streambuf that captures output text and raises `sig` once `target_newline`
// newline characters have been written.
class SignalOnNewlineBuf : public std::streambuf {
 public:
  // Initializes the buffer to raise `sig` on the `target_newline`-th `'\n'`.
  SignalOnNewlineBuf(int sig, int target_newline)
      : sig_(sig), target_newline_(target_newline) {}

  // Returns all characters written to this buffer so far.
  const std::string& text() const { return text_; }

 protected:
  // Appends `ch` to `text_` and raises `sig_` at `target_newline_`.
  int_type overflow(int_type ch) override;

 private:
  int sig_;
  int target_newline_;
  int newlines_ = 0;
  std::string text_;
};

// RAII guard that redirects `std::cerr` to another streambuf within a scope.
class ScopedCerrBuf {
 public:
  // Redirects `std::cerr` to `next`, saving the previous streambuf.
  explicit ScopedCerrBuf(std::streambuf* next);
  ScopedCerrBuf(const ScopedCerrBuf&) = delete;
  ScopedCerrBuf& operator=(const ScopedCerrBuf&) = delete;

  // Restores `std::cerr` to its previous streambuf.
  ~ScopedCerrBuf();

 private:
  std::streambuf* prev_;
};

// Outcome captured from running `Main` until interrupted by a signal.
struct InterruptedMainResult {
  int exit_code = 0;
  bool sigint_restored = false;
  bool sigterm_restored = false;
  std::string stderr_text;
};

// Runs `Main` with `args`, raising `sig` on the `target_newline`-th progress
// line written to `std::cerr`, restores `SIGINT` and `SIGTERM` handlers, and
// returns the captured exit status, handler restoration state, and stderr text.
InterruptedMainResult RunMainInterruptedBySignal(
    absl::Span<const std::string> args, int sig, int target_newline);

// Builds a `TensorSpec` with `name`, `shape`, optional `binding_name`, and no
// explicit element type set (`has_type() == false`).
TensorSpec MakeTensorSpec(absl::string_view name, absl::Span<const int> shape,
                          absl::string_view binding_name = "");

// Builds a `TensorSpec` with `name`, explicit `type`, `shape`, and optional
// `binding_name`.
TensorSpec MakeTensorSpec(absl::string_view name, ElementType type,
                          absl::Span<const int> shape,
                          absl::string_view binding_name = "");

// Sets quantization `(type, scale, zero_point)` on all inputs and outputs of
// `manifest` using distinct input and output quantization parameters.
void SetManifestQuantization(DatasetManifest& manifest, ElementType type,
                             float in_scale, int in_zero_point, float out_scale,
                             int out_zero_point);

// Sets identical quantization `(type, scale, zero_point)` on all inputs and
// outputs of `manifest`.
void SetManifestQuantization(DatasetManifest& manifest, ElementType type,
                             float scale, int zero_point);

// Builds a single-sample (`id = "s0"`) classification manifest with one raw
// tensor input `input_spec` pointing to `input_file`, `label`, and an optional
// `output_spec` and `output_task`.
DatasetManifest MakeSingleInputManifest(
    const TensorSpec& input_spec, absl::string_view input_file, int label,
    const TensorSpec* output_spec = nullptr,
    std::optional<Task> output_task = std::nullopt);

// Populates `serving_default` from `model.subgraphs[0]` when `model` has no
// signatures, packs `model` into a FlatBuffer, and writes it to `path`.
Expected<void> PackAndWriteModel(const std::filesystem::path& path,
                                 tflite::ModelT& model);

// Writes the default two-input (`left`, `right`), two-output (`depth`,
// `segmentation`) FP32 `ADD` test model to `path`.
void WriteTwoInputTwoOutputAddModel(const std::filesystem::path& path);

// Unpacks `src_path`, rewrites all tensors in `subgraphs[0]` to `tensor_type`
// (`UINT8` or `INT8`) with `(scale, zero_point)`, and writes the quantized
// model to `dst_path`.
void WriteQuantizedModelCopy(const std::filesystem::path& src_path,
                             const std::filesystem::path& dst_path,
                             ElementType tensor_type, int zero_point,
                             float scale = 0.25f);

// Default `[1, 2, 2, 3]` NHWC shape for `WriteClassificationModel`.
inline constexpr std::array<int32_t, 4> kDefaultClassificationShape = {
    1, 2, 2, 3};

// Writes a single-input, single-output `ADD` classification model with constant
// `bias`, tensor `shape`, `input_name`, and `tensor_type` (`INT8` with
// `scale = 0.25, zp = 0`, or `FLOAT32`) I/O tensors to `path`.
void WriteClassificationModel(
    const std::filesystem::path& path, ElementType tensor_type,
    absl::Span<const float> bias,
    absl::Span<const int32_t> shape = kDefaultClassificationShape,
    absl::string_view input_name = "image");

// Reusable 2-sample PPM + TSV classification dataset and candidate/baseline
// models for classification and image-binding tests.
struct ClassificationDatasetFixture {
  std::filesystem::path img_dir;
  std::filesystem::path tsv_path;
  std::filesystem::path cand_path;
  std::filesystem::path base_agree_path;
  std::filesystem::path base_disagree_path;
  EvaluationConfig config;
};

// Base GoogleTest fixture that provisions a per-test scratch directory, a
// default two-input/two-output model, a three-sample manifest, and assertion
// helpers for error and checkpoint-corruption scenarios.
class RunnerTest : public testing::Test {
 protected:
  // Creates `root_`, writes `model.tflite` and `manifest.txtpb`, and populates
  // default CPU `config_`.
  void SetUp() override;

  // Writes a quantized (`UINT8` or `INT8`) copy of `root_ / "model.tflite"` to
  // `path`.
  void WriteQuantizedModelCopy(const std::filesystem::path& path,
                               ElementType tensor_type, int zero_point,
                               float scale = 0.25f);

  // Materializes a 2-sample `.ppm` + `.tsv` classification dataset and
  // candidate/baseline `.tflite` models under `root_`, populating `out`.
  void SetupClassificationDataset(ClassificationDatasetFixture& out);

  // Runs `Evaluate(config_, ..., summary)` using `config_.model()` as the
  // stand-in binary identity.
  Expected<void> Run(RunSummary& summary);

  // Runs `Evaluate` on `cfg` (setting `output_dir` to `root_ / out_subdir` when
  // `out_subdir` is non-empty), asserts failure containing `expected_substr`,
  // and optionally copies the resulting `RunSummary` to `*summary_out`.
  void ExpectRunError(EvaluationConfig cfg, absl::string_view out_subdir,
                      absl::string_view expected_substr,
                      RunSummary* summary_out = nullptr);

  // Writes `manifest` to `<root_>/<case_name>.txtpb`, binds it onto a copy of
  // `base_cfg`, and asserts `Evaluate` fails with `expected_substr`.
  void ExpectManifestError(const EvaluationConfig& base_cfg,
                           const DatasetManifest& manifest,
                           absl::string_view case_name,
                           absl::string_view expected_substr);

  // Convenience overload of `ExpectManifestError` using `config_` as base.
  void ExpectManifestError(const DatasetManifest& manifest,
                           absl::string_view case_name,
                           absl::string_view expected_substr);

  // Snapshots `SampleResult` at `row_path`, applies `mutate`, runs `Evaluate`
  // with `cfg`, restores the valid checkpoint, and asserts failure containing
  // `expected_substr` with zero resumed samples.
  void ExpectCorruptCheckpoint(
      const EvaluationConfig& cfg, const std::filesystem::path& row_path,
      absl::FunctionRef<void(SampleResult&)> mutate,
      absl::string_view expected_substr);

  // Convenience overload of `ExpectCorruptCheckpoint` using `config_`.
  void ExpectCorruptCheckpoint(
      const std::filesystem::path& row_path,
      absl::FunctionRef<void(SampleResult&)> mutate,
      absl::string_view expected_substr);

  std::filesystem::path root_;
  EvaluationConfig config_;
  DatasetManifest manifest_;
};

}  // namespace litert::vision

#endif  // LITERT_TOOLS_VISION_EVAL_RUNNER_TEST_UTIL_H_
