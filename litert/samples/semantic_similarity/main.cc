/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cctype>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

// #include "base/init_google.h"
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/options/litert_cpu_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "litert/cc/options/litert_qualcomm_options.h"
#include "sentencepiece_processor.h"  // from @sentencepiece

// Command-line flags for the model paths and input sentences
ABSL_FLAG(std::string, tokenizer, "", "Path to the tokenizer model file.");
ABSL_FLAG(std::string, embedder, "",
          "Path to the sentence embedder model file.");
ABSL_FLAG(std::string, sentence1, "",
          "The first sentence for similarity comparison.");
ABSL_FLAG(std::string, sentence2, "",
          "The second sentence for similarity comparison.");
ABSL_FLAG(std::string, accelerator, "cpu",
          "Which backend to use. Comma delimited string of accelerators (e.g. "
          "cpu,gpu,npu). Will delegate to NPU, GPU, then CPU if they are "
          "specified in this flag.");
ABSL_FLAG(int, sequence_length, 0,
          "Number of threads to use for CPU inference.");
ABSL_FLAG(std::string, dispatch_library_dir, "",
          "Path to the dispatch library directory.");

litert::HwAcceleratorSet GetAccelerator() {
  const std::string accelerator_str = absl::GetFlag(FLAGS_accelerator);
  litert::HwAcceleratorSet accelerators(litert::HwAccelerators::kNone);
  for (absl::string_view accelerator : absl::StrSplit(accelerator_str, ',')) {
    if (accelerator == "gpu") {
      accelerators |= litert::HwAccelerators::kGpu;
    } else if (accelerator == "npu") {
      accelerators |= litert::HwAccelerators::kNpu;
      accelerators |= litert::HwAccelerators::kCpu;
    } else if (accelerator == "cpu") {
      accelerators |= litert::HwAccelerators::kCpu;
    }
  }
  return accelerators;
}

namespace litert {
namespace {

/**
 * @brief Parses the sequence length from the embedder model filename.
 *
 * Assumes the filename contains a pattern like "<number>_input_seq".
 *
 * @param embedder_path The file path to the embedder model.
 * @return The parsed sequence length, or an error status.
 */
absl::StatusOr<int> GetSeqLenFromPath(const absl::string_view embedder_path) {
  const size_t last_slash = embedder_path.find_last_of('/');
  const absl::string_view filename = (last_slash == absl::string_view::npos)
                                         ? embedder_path
                                         : embedder_path.substr(last_slash + 1);

  const size_t seq_pos = filename.find("_input_seq");
  if (seq_pos == absl::string_view::npos) {
    return absl::InvalidArgumentError(
        "Embedder filename must contain '_input_seq' to infer sequence "
        "length.");
  }

  size_t num_end = seq_pos;
  size_t num_start = num_end;
  while (num_start > 0 && isdigit(filename[num_start - 1])) {
    --num_start;
  }

  if (num_start == num_end) {
    return absl::InvalidArgumentError(
        "Could not find sequence length number before '_input_seq' in "
        "filename.");
  }

  int seq_len;
  if (!absl::SimpleAtoi(filename.substr(num_start, num_end - num_start),
                        &seq_len)) {
    return absl::InvalidArgumentError(
        "Failed to parse sequence length number.");
  }
  return seq_len;
}

/**
 * @brief Calculates the cosine similarity between two vectors.
 */
float CosineSimilarity(const std::vector<float>& v1,
                       const std::vector<float>& v2) {
  ABSL_QCHECK(!v1.empty() && !v2.empty() && v1.size() == v2.size());
  double dot_product = 0.0;
  double norm1 = 0.0;
  double norm2 = 0.0;
  for (size_t i = 0; i < v1.size(); ++i) {
    dot_product += v1[i] * v2[i];
    norm1 += v1[i] * v1[i];
    norm2 += v2[i] * v2[i];
  }
  const double magnitude = sqrt(norm1) * sqrt(norm2);
  return (magnitude == 0.0) ? 0.0f
                            : static_cast<float>(dot_product / magnitude);
}

/**
 * @brief Preprocesses the token IDs by truncating, adding BOS/EOS, and padding.
 * @param tokenizer_processor The initialized SentencePiece processor.
 * @param token_ids The vector of token IDs to preprocess in-place.
 * @param seq_len The target sequence length for padding.
 */
void PreprocessTokens(
    sentencepiece::SentencePieceProcessor* tokenizer_processor,
    std::vector<int>* token_ids, const int seq_len) {
  // Truncate to max length.
  const int max_len = seq_len - 2;
  if (token_ids->size() > max_len) {
    token_ids->resize(max_len);
  }

  // Add BOS and EOS tokens.
  token_ids->insert(token_ids->begin(), tokenizer_processor->bos_id());
  token_ids->push_back(tokenizer_processor->eos_id());

  // Pad to sequence length.
  const int num_padding = seq_len - token_ids->size();
  if (num_padding > 0) {
    token_ids->insert(token_ids->end(), num_padding,
                      tokenizer_processor->pad_id());
  }
}

/**
 * @brief Tokenizes a sentence using a SentencePiece tokenizer model.
 * @param tokenizer_processor The initialized SentencePiece processor.
 * @param sentence The sentence to tokenize.
 * @return A vector of token IDs, or an error status.
 */
absl::StatusOr<std::vector<int>> Tokenize(
    sentencepiece::SentencePieceProcessor* tokenizer_processor,
    const std::string& sentence) {
  std::vector<int> token_ids;
  auto status = tokenizer_processor->Encode(sentence, &token_ids);
  if (!status.ok()) {
    return status;
  }
  return token_ids;
}

/**
 * @brief Generates a sentence embedding using a LiteRT embedder model.
 * @param embedder_model The compiled embedder model.
 * @param token_ids The token IDs from the tokenizer.
 * @return A vector representing the sentence embedding, or an error status.
 */
Expected<std::vector<float>> GetEmbedding(
    CompiledModel* embedder_model, std::vector<TensorBuffer>& input_buffers,
    std::vector<TensorBuffer>& output_buffers,
    const std::vector<int>& token_ids) {
  if (input_buffers.size() != 1) {
    return Unexpected(::litert::Status::kErrorInvalidArgument,
                      "Expected 1 input tensor for embedder");
  }

  LITERT_RETURN_IF_ERROR(input_buffers[0].Write<int>(token_ids));

  LITERT_RETURN_IF_ERROR(embedder_model->Run(input_buffers, output_buffers));

  if (output_buffers.size() != 1) {
    return Unexpected(::litert::Status::kErrorInvalidArgument,
                      "Expected 1 output tensor for embedder");
  }
  auto& output_tensor = output_buffers[0];
  LITERT_ASSIGN_OR_RETURN(size_t output_size_bytes, output_tensor.PackedSize());
  std::vector<float> embedding(output_size_bytes / sizeof(float));
  LITERT_RETURN_IF_ERROR(output_tensor.Read(absl::MakeSpan(embedding)));

  return embedding;
}

// Helper function to run the main logic and handle status returns.
absl::Status RealMain() {
  const std::string embedder_path = absl::GetFlag(FLAGS_embedder);
  const std::string dispatch_library_dir =
      absl::GetFlag(FLAGS_dispatch_library_dir);
  ABSL_QCHECK(!absl::GetFlag(FLAGS_tokenizer).empty())
      << "Please provide --tokenizer.";
  ABSL_QCHECK(!embedder_path.empty()) << "Please provide --embedder.";
  ABSL_QCHECK(!absl::GetFlag(FLAGS_sentence1).empty())
      << "Please provide --sentence1.";
  ABSL_QCHECK(!absl::GetFlag(FLAGS_sentence2).empty())
      << "Please provide --sentence2.";

  // 0. Get sequence length from flag or embedder model path.
  int seq_len;
  if (absl::GetFlag(FLAGS_sequence_length) > 0) {
    seq_len = absl::GetFlag(FLAGS_sequence_length);
  } else {
    absl::StatusOr<int> seq_len_or = GetSeqLenFromPath(embedder_path);
    if (!seq_len_or.ok()) {
      return seq_len_or.status();
    }
    seq_len = *seq_len_or;
  }

  // 1. Load Models
  sentencepiece::SentencePieceProcessor tokenizer_processor;
  auto status = tokenizer_processor.Load(absl::GetFlag(FLAGS_tokenizer));
  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to load tokenizer model: ", status.ToString()));
  }

  // Create LiteRT Environment and CompiledModel, ensuring lifetimes
  // are managed correctly within this scope.
  std::vector<litert::Environment::Option> environment_options = {};
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  auto accelerator = GetAccelerator();
  // Set CPU compilation options.
  if (accelerator & litert::HwAccelerators::kNpu) {
    if (!absl::GetFlag(FLAGS_dispatch_library_dir).empty()) {
      environment_options.push_back(litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          absl::string_view(dispatch_library_dir)});
    } else {
      return absl::InvalidArgumentError("Dispatch library directory is empty.");
    }
    // QNN options
    LITERT_ASSIGN_OR_RETURN(auto& qnn_opts, options.GetQualcommOptions());
    qnn_opts.SetLogLevel(::litert::qualcomm::QualcommOptions::LogLevel::kOff);
    qnn_opts.SetHtpPerformanceMode(
        ::litert::qualcomm::QualcommOptions::HtpPerformanceMode::kBurst);
    options.SetHardwareAccelerators(accelerator);
    // Add other NPU options here..
  } else if (accelerator & litert::HwAccelerators::kCpu) {
    LITERT_ASSIGN_OR_RETURN(auto& cpu_compilation_options,
                            options.GetCpuOptions());
    LITERT_RETURN_IF_ERROR(cpu_compilation_options.SetNumThreads(4));
    options.SetHardwareAccelerators(accelerator);
    // Set GPU compilation options.
  } else if (accelerator & litert::HwAccelerators::kGpu) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_compilation_options,
                            options.GetGpuOptions());
    gpu_compilation_options.SetPrecision(GpuOptions::Precision::kFp32);

    options.SetHardwareAccelerators(accelerator);
  } else {
    return absl::InvalidArgumentError("No supported accelerators specified.");
  }

  LITERT_ASSIGN_OR_RETURN(
      auto env, Environment::Create(absl::MakeConstSpan(environment_options)));

  LITERT_ASSIGN_OR_RETURN(auto embedder_model,
                          CompiledModel::Create(env, embedder_path, options));

  // 2. Process Sentences
  auto tokens1_or =
      Tokenize(&tokenizer_processor, absl::GetFlag(FLAGS_sentence1));
  if (!tokens1_or.ok()) {
    return absl::InternalError(absl::StrCat("Failed to tokenize sentence 1: ",
                                            tokens1_or.status().ToString()));
  }
  auto tokens1 = std::move(*tokens1_or);
  PreprocessTokens(&tokenizer_processor, &tokens1, seq_len);

  auto tokens2_or =
      Tokenize(&tokenizer_processor, absl::GetFlag(FLAGS_sentence2));
  if (!tokens2_or.ok()) {
    return absl::InternalError(absl::StrCat("Failed to tokenize sentence 2: ",
                                            tokens2_or.status().ToString()));
  }
  auto tokens2 = std::move(*tokens2_or);
  PreprocessTokens(&tokenizer_processor, &tokens2, seq_len);

  LITERT_ASSIGN_OR_RETURN(auto input_buffers,
                          embedder_model.CreateInputBuffers());
  LITERT_ASSIGN_OR_RETURN(auto output_buffers,
                          embedder_model.CreateOutputBuffers());

  ABSL_LOG(INFO) << "Getting embedding for sentence 1";
  LITERT_ASSIGN_OR_RETURN(
      auto embedding1,
      GetEmbedding(&embedder_model, input_buffers, output_buffers, tokens1));
  ABSL_LOG(INFO) << "Getting embedding for sentence 2";
  LITERT_ASSIGN_OR_RETURN(
      auto embedding2,
      GetEmbedding(&embedder_model, input_buffers, output_buffers, tokens2));

  // 3. Calculate and Print the Similarity Score
  ABSL_LOG(INFO) << "Calculating similarity score";
  const float similarity = CosineSimilarity(embedding1, embedding2);
  std::cout.precision(2);
  std::cout << "Cosine Similarity: " << std::fixed << similarity << std::endl;

  return absl::OkStatus();
}

}  // namespace
}  // namespace litert

int main(int argc, char** argv) {
  // InitGoogle(argv[0], &argc, &argv, true);
  absl::ParseCommandLine(argc, argv);
  absl::Status status = litert::RealMain();
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run semantic similarity: " << status;
    return 1;
  }
  return 0;
}
