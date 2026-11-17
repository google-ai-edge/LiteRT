// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// carve: rewrites a model so that each region of its graphs that matches a
// pattern becomes one custom op, a STABLEHLO_COMPOSITE whose decomposition
// subgraph holds the region's ops (the form odml.rms_norm and the other
// composites of converted models have). A backend with a kernel for the
// composite runs the region as one op; every other runtime runs the
// decomposition and computes what it computed before.
//
// USAGE
//
//   carve --model=IN --output=OUT [--patterns=FILE] [--numeric_check]
//         [--numeric_signatures=A,B] [--num_threads=N]
//   carve --model=IN                 report what would be carved, write nothing
//   carve --print_patterns           print the built-in pattern description
//
// IN is a .tflite file or a LiteRT-LM container (.litertlm); OUT has the same
// format. Every TFLite model of a container is searched, in every subgraph
// that is not itself the decomposition of a composite. A model without a
// matching region is copied byte for byte, so carving a carved file changes
// nothing. In a rewritten model the constants are not touched: those the
// model keeps behind its flatbuffer are copied as one block from the input to
// the output, and those in other sections of a container stay where they are.
// (One exception in form, not in content: a constant without elements whose
// empty data sits inside the flatbuffer is written as data of no bytes behind
// it, because the schema's objects cannot write empty data back.)
// All other sections of a container are copied byte for byte; the header
// keeps its size, with the sections' offsets and the checksum of each
// rewritten section updated.
//
// PATTERNS
//
// A pattern describes a region: its ops, predicates on its tensors, its
// inputs and outputs, and the attributes of the composite. The format is
// documented in carve_pattern.h; `carve --print_patterns` prints the built-in
// description, which is the template for a new one:
//
//   pattern odml.my_region            # the composite's name
//   in  x : f16 [1, M, F]             # an input: type and shape predicates;
//                                     #   M and F take the dimensions
//   in  w : const                     # an input that is a constant and stays
//                                     #   one in the graph (weights)
//   out y
//   q = QUANTIZE(x)        : i8<S> [1, M, F]   # <S>: one scale, zero point 0
//   y = FULLY_CONNECTED(q, w) {fused_activation_function = "NONE"} : i8<T>
//   check F == 2 * 1024               # a condition on symbols and constants
//   attr features = F                 # an attribute of the composite
//   attr scale_in = S
//
// The matcher starts at each op of the kind of the last statement and follows
// the pattern's edges. A constant that an op uses and the pattern does not
// declare as an input (a shape, a slice position) moves into the
// decomposition. The report lists, for each subgraph and pattern, the regions
// carved and the places that came close with the reason.
//
// THE FACTORIZED MLP (built in)
//
// odml.factorized_mlp is a feed-forward whose up and down projections are each
// the product of two factors. The region runs from the F16 tensor that enters
// the RESHAPE before the MLP's QUANTIZE to the result of its last DEQUANTIZE:
//   inputs   0: x F16 [1, M, features]
//            1, 2: up 1, first and second half, [up1_features, features / 2]
//            3: up 2 [groups, 2 * up1_features / groups, hidden / groups]
//            4: down 1 [groups, hidden / groups, 2 * up1_features / groups]
//            5, 6: down 2, first and second half, [features / 2, up1_features]
//            The weights are the graph's constants as they were, with their
//            quantization (centered INT2 has blockwise scale and zero point
//            tensors). Up 2 and down 1 may have a leading dimension of 1, as
//            in a decode graph.
//   output   0: y F16 [1, M, features]
//   attributes (a flexbuffer map): version 1, features, blocks 2, groups,
//            up1_features, hidden_features, activation "silu", and the scale
//            of each INT8 stage (zero point 0): scale_in (the quantized
//            input), scale_up1, scale_up2, scale_act (the quantized
//            activation), scale_down1, scale_out.
// The decomposition holds the 28 original ops and the shape constants.
//
// CHECKS
//
// After writing, the tool reads its output back and checks, for each
// rewritten model:
//   - structure: with the new composites replaced by their decompositions,
//     every subgraph is the graph it was (operators and their options, the
//     type, shape and quantization of every tensor including the scale and
//     zero point tensors of blockwise quantization, the bytes of every
//     constant); every tensor is written before it is read; signatures,
//     metadata and operator codes are unchanged; every new subgraph is the
//     decomposition of exactly one composite; no tensor is left unused. See
//     carve_verify.h.
//   - a second pass finds nothing to carve.
//   - with --numeric_check: the original and the carved model give
//     bit-identical outputs on the CPU for the same pseudo-random inputs, in
//     every signature (or those of --numeric_signatures). This runs the whole
//     model twice. A model with constants outside its own bytes (TFLite
//     external buffers) cannot be run this way, which the tool then says.
// and, for a container, that every other section is byte for byte the
// input's. If a check fails the output file is removed and the exit status is
// 1.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/flags/usage.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"
#include "litert/vendors/nvidia/tools/carve_container.h"
#include "litert/vendors/nvidia/tools/carve_model.h"
#include "litert/vendors/nvidia/tools/carve_pattern.h"
#include "litert/vendors/nvidia/tools/carve_verify.h"
#include "tflite/model_builder.h"
#include "tflite/stderr_reporter.h"

ABSL_FLAG(std::string, model, "",
          "The .tflite file or LiteRT-LM container to carve.");
ABSL_FLAG(std::string, output, "",
          "Where to write the carved file. Without it the tool only reports "
          "what it would carve.");
ABSL_FLAG(std::string, patterns, "",
          "A pattern description file (the format is in carve_pattern.h). "
          "Without it the built-in patterns are used.");
ABSL_FLAG(bool, print_patterns, false,
          "Print the description of the built-in patterns and exit.");
ABSL_FLAG(bool, numeric_check, false,
          "Also run the original and the carved model on the CPU and require "
          "bit-identical outputs.");
ABSL_FLAG(std::string, numeric_signatures, "",
          "The signatures --numeric_check runs, separated by commas; all of "
          "them when empty.");
ABSL_FLAG(int32_t, num_threads, 8, "The CPU threads of --numeric_check.");

namespace litert::tools {
namespace {

using Bytes = litert::BufferRef<uint8_t>;

// A TFLite model of the input: the file itself, or a section of a container.
struct ModelJob {
  std::string label;
  size_t section = 0;
  Bytes source;
  CarveReport report;
  std::optional<CarvedModel> carved;
};

absl::StatusOr<std::unique_ptr<tflite::Allocation>> MapFile(
    const std::string& path) {
  auto allocation = tflite::GetAllocationFromFile(
      path.c_str(), tflite::DefaultErrorReporter());
  if (allocation == nullptr || !allocation->valid()) {
    return absl::NotFoundError(absl::StrCat("cannot read ", path));
  }
  return std::move(allocation);
}

Bytes BytesOf(const tflite::Allocation& allocation) {
  return Bytes(static_cast<const uint8_t*>(allocation.base()),
               allocation.bytes());
}

absl::StatusOr<std::vector<CarvePattern>> LoadPatterns() {
  const std::string path = absl::GetFlag(FLAGS_patterns);
  if (path.empty()) return ParseCarvePatterns(BuiltinCarvePatternText());
  std::ifstream stream(path);
  if (!stream) {
    return absl::NotFoundError(absl::StrCat("cannot read ", path));
  }
  std::stringstream text;
  text << stream.rdbuf();
  auto patterns = ParseCarvePatterns(text.str());
  if (!patterns.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat(path, ": ", patterns.status().message()));
  }
  return patterns;
}

void PrintReport(const ModelJob& job) {
  const auto& report = job.report;
  std::cout << job.label << ", " << job.source.Size() << " bytes: "
            << report.num_subgraphs - report.num_decompositions_skipped
            << " of " << report.num_subgraphs << " subgraphs searched ("
            << report.num_decompositions_skipped << " are decompositions), "
            << report.num_carved << " regions to carve\n";
  for (const auto& entry : report.entries) {
    std::cout << "  subgraph " << entry.subgraph_index << " \""
              << entry.subgraph_name << "\": " << entry.pattern << ": "
              << entry.num_carved << " regions";
    size_t near_misses = 0;
    for (const auto& near_miss : entry.near_misses) {
      near_misses += near_miss.count;
    }
    std::cout << ", " << near_misses << " near misses\n";
    for (const auto& near_miss : entry.near_misses) {
      std::cout << "    near miss at " << near_miss.count
                << " places (first: " << near_miss.first_place
                << "): " << near_miss.reason << "\n";
    }
  }
}

// The checks on one rewritten model, read back from the output.
absl::Status CheckModel(const ModelJob& job, Bytes written,
                        const std::vector<CarvePattern>& patterns) {
  std::string summary;
  if (auto status = VerifyCarvedStructure(job.source, written, &summary);
      !status.ok()) {
    return status;
  }
  std::cout << "  structure: " << summary << "\n";

  CarveReport again;
  auto second = CarveModel(written, patterns, &again);
  if (!second.ok()) return second.status();
  if (again.num_carved != 0) {
    return absl::FailedPreconditionError(absl::StrCat(
        "a second pass still finds ", again.num_carved, " regions"));
  }
  std::cout << "  second pass: nothing left to carve\n";

  if (absl::GetFlag(FLAGS_numeric_check)) {
    NumericCheckOptions options;
    options.num_threads = absl::GetFlag(FLAGS_num_threads);
    options.signatures = absl::StrSplit(absl::GetFlag(FLAGS_numeric_signatures),
                                        ',', absl::SkipEmpty());
    const auto status = CompareOnCpu(job.source, written, options, &summary);
    if (absl::IsUnimplemented(status)) {
      // Not a difference: the check cannot run this model.
      std::cout << "  numeric: not run: " << status.message() << "\n";
    } else if (!status.ok()) {
      return status;
    } else {
      std::cout << "  numeric: " << summary << "\n";
    }
  }
  return absl::OkStatus();
}

absl::Status Run() {
  auto patterns = LoadPatterns();
  if (!patterns.ok()) return patterns.status();
  const std::string input_path = absl::GetFlag(FLAGS_model);
  const std::string output_path = absl::GetFlag(FLAGS_output);
  if (input_path.empty()) {
    return absl::InvalidArgumentError("--model is required");
  }
  // The input is read through a mapping while the output is written.
  if (input_path == output_path) {
    return absl::InvalidArgumentError("--output must not be the input file");
  }
  auto input = MapFile(input_path);
  if (!input.ok()) return input.status();
  const Bytes file = BytesOf(**input);

  const bool is_container = IsContainer(file);
  std::vector<ContainerSection> sections;
  std::vector<ModelJob> jobs;
  if (is_container) {
    auto parsed = ParseContainer(file);
    if (!parsed.ok()) return parsed.status();
    sections = std::move(*parsed);
    for (size_t i = 0; i < sections.size(); ++i) {
      if (sections[i].data_type != kContainerTfliteModel) continue;
      auto& job = jobs.emplace_back();
      job.label = absl::StrCat("section ", i, " ", sections[i].model_type);
      job.section = i;
      job.source = Bytes(file.Data() + sections[i].begin,
                         sections[i].end - sections[i].begin);
    }
    std::cout << input_path << ": LiteRT-LM container, " << sections.size()
              << " sections, " << jobs.size() << " TFLite models\n";
  } else {
    auto& job = jobs.emplace_back();
    job.label = "model";
    job.source = file;
    std::cout << input_path << ": TFLite model\n";
  }

  size_t num_carved = 0;
  for (auto& job : jobs) {
    auto carved = CarveModel(job.source, *patterns, &job.report);
    if (!carved.ok()) {
      return absl::Status(
          carved.status().code(),
          absl::StrCat(job.label, ": ", carved.status().message()));
    }
    job.carved = std::move(*carved);
    num_carved += job.report.num_carved;
    PrintReport(job);
  }
  if (output_path.empty()) return absl::OkStatus();

  // Nothing to carve: the output is the input.
  if (num_carved == 0) {
    if (auto status = WriteFile(file, output_path); !status.ok()) return status;
    std::cout << "wrote " << output_path << ": a copy of the input\n";
    return absl::OkStatus();
  }
  std::vector<ContainerSection> written_sections;
  if (is_container) {
    std::vector<std::optional<CarvedModel>> replacements(sections.size());
    for (auto& job : jobs) {
      if (job.carved.has_value()) replacements[job.section] = *job.carved;
    }
    auto written = WriteContainer(file, sections, replacements, output_path);
    if (!written.ok()) return written.status();
    written_sections = std::move(*written);
  } else if (auto status = WriteModel(*jobs[0].carved, output_path);
             !status.ok()) {
    return status;
  }

  // The checks run on the file as it is on disk.
  auto output = MapFile(output_path);
  if (!output.ok()) return output.status();
  const Bytes out = BytesOf(**output);
  std::cout << "wrote " << output_path << " (" << out.Size() << " bytes)\n";
  auto check = [&]() -> absl::Status {
    if (is_container) {
      auto reread = ParseContainer(out);
      if (!reread.ok()) return reread.status();
      if (reread->size() != sections.size()) {
        return absl::DataLossError("the output has a different section count");
      }
      size_t job_index = 0;
      for (size_t i = 0; i < sections.size(); ++i) {
        const auto& section = (*reread)[i];
        const Bytes bytes(out.Data() + section.begin,
                          section.end - section.begin);
        const bool rewritten = job_index < jobs.size() &&
                               jobs[job_index].section == i &&
                               jobs[job_index].carved.has_value();
        if (job_index < jobs.size() && jobs[job_index].section == i) {
          if (rewritten) {
            std::cout << "checking " << jobs[job_index].label << "\n";
            if (auto status = CheckModel(jobs[job_index], bytes, *patterns);
                !status.ok()) {
              return status;
            }
          }
          ++job_index;
        }
        if (rewritten) continue;
        const uint64_t size = sections[i].end - sections[i].begin;
        if (bytes.Size() != size ||
            std::memcmp(bytes.Data(), file.Data() + sections[i].begin, size) !=
                0) {
          return absl::DataLossError(
              absl::StrCat("section ", i, " is not a copy of the input's"));
        }
      }
      std::cout << "every other section is a byte-for-byte copy\n";
      return absl::OkStatus();
    }
    std::cout << "checking " << jobs[0].label << "\n";
    return CheckModel(jobs[0], out, *patterns);
  };
  if (auto status = check(); !status.ok()) {
    output->reset();
    std::remove(output_path.c_str());
    return absl::Status(status.code(),
                        absl::StrCat("check failed, ", output_path,
                                     " removed: ", status.message()));
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace litert::tools

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(
      "carve --model=IN [--output=OUT] [--patterns=FILE] [--numeric_check]\n"
      "Rewrites a .tflite file or LiteRT-LM container so that each region "
      "matching a pattern becomes one composite op. The guide is at the top "
      "of carve_main.cc.");
  absl::ParseCommandLine(argc, argv);
  if (absl::GetFlag(FLAGS_print_patterns)) {
    std::cout << litert::tools::BuiltinCarvePatternText();
    return 0;
  }
  const absl::Status status = litert::tools::Run();
  if (!status.ok()) {
    std::cerr << "carve: " << status.message() << "\n";
    return 1;
  }
  return 0;
}
