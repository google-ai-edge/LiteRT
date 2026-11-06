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

#ifndef LITERT_TOOLS_VISION_EVAL_DATA_H_
#define LITERT_TOOLS_VISION_EVAL_DATA_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <string>
#include <vector>

#include "litert/cc/litert_expected.h"
#include "litert/tools/vision_eval/evaluation.pb.h"
#include "litert/tools/vision_eval/image.h"
#include "google/protobuf/message.h"  // from @com_google_protobuf

namespace litert::vision {

// Returns a `kLiteRtStatusErrorInvalidArgument` error with `message`.
Error Invalid(const std::string& message);

// Reads and initializes `message` from a `.txtpb` or `.binpb` file at `path`
// (up to 64 MiB).
Expected<void> ReadProto(const std::filesystem::path& path,
                         google::protobuf::Message& message);

// Serializes `message` in `format` and atomically writes it to `path`,
// creating parent directories if needed.
Expected<void> WriteProto(const std::filesystem::path& path,
                          const google::protobuf::Message& message,
                          ProtoFormat format = BINARY_PROTO);

// Atomically writes `size` bytes from `data` to `path` via a temporary file
// rename, creating parent directories if needed.
Expected<void> WriteBytes(const std::filesystem::path& path, const void* data,
                          size_t size);

// Validates `spec` (element type, optional quantization, static dimensions up
// to rank 8, and 512 MiB cap) and returns the total tensor byte count.
Expected<size_t> TensorBytes(const TensorSpec& spec);

// Reads raw tensor bytes from `path`, requiring the file size to match the
// exact byte count computed from `spec`.
Expected<std::vector<uint8_t>> ReadTensor(const std::filesystem::path& path,
                                          const TensorSpec& spec);

// Reads a raw `FLOAT32` tensor matching `spec` from `path` into a vector of
// finite floats, rejecting NaN or infinite values.
Expected<std::vector<float>> ReadFloats(const std::filesystem::path& path,
                                        const TensorSpec& spec);

// Resolves `path` to an absolute path, verifies that it refers to an existing
// regular file, and returns its `FileIdentity`.
Expected<FileIdentity> Identify(const std::filesystem::path& path);

// Verifies that `file.path()` is non-empty and refers to an existing regular
// file.
Expected<void> VerifyFile(const FileIdentity& file);

// Returns true if `value` is 1 to 128 ASCII alphanumeric, `_`, or `-`
// characters safe for single-component filenames and identifiers.
bool SafeName(const std::string& value);

// Resolves `path` relative to `root` (or returns `path` unchanged if `path` is
// already absolute or `root` is empty).
std::filesystem::path Resolve(const std::filesystem::path& root,
                              const std::filesystem::path& path);

// Loads a `.tsv`, `.txtpb`, or `.binpb` dataset manifest specified by `config`,
// validates it against `config.task()`, and slices samples to the configured
// `[start_sample, start_sample + max_samples)` range.
Expected<DatasetManifest> LoadManifest(const EvaluationConfig& config);

// Validates manifest version, tensor specifications, task-specific scoring
// bounds, and per-sample input and target bindings for `task`.
Expected<void> ValidateManifest(const DatasetManifest& manifest, Task task);

// Converts `preprocessing` proto settings into a validated
// `ImagePreprocessing` configuration, applying defaults for unset fields.
Expected<ImagePreprocessing> ImageOptions(const Preprocessing& preprocessing);

// Validates required paths, execution mode constraints, baseline comparison
// flags, numerical tolerances, and preprocessing options in `config`.
Expected<void> ValidateConfig(const EvaluationConfig& config);

// Parses `--config` and `--name=value` command-line flags into a validated
// `EvaluationConfig`, applying explicit flags over any loaded config file.
Expected<EvaluationConfig> ParseArgs(int argc, char** argv);

// Prints command-line usage and common flag options to standard error.
void PrintUsage();

}  // namespace litert::vision

#endif  // LITERT_TOOLS_VISION_EVAL_DATA_H_
