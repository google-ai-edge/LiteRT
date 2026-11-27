// Copyright 2026 Amlogic, Inc.
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

#include "litert/tools/flags/vendors/amlogic_flags.h"

#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_amlogic_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_amlogic_options.h"
#include "litert/tools/flags/options_parser_registry.h"

ABSL_FLAG(std::string, amlogic_model_path, "",
          "Directory passed to the AML compiler while it produces the ADLA "
          "binary. The path is not stored in the compiled model.");
ABSL_FLAG(std::string, amlogic_model_name, "",
          "Base name passed to the AML compiler for the ADLA artifact.");
ABSL_FLAG(std::string, amlogic_log_level, "",
          "AML compiler log level: off, error, warn, info, verbose, debug. "
          "Empty follows LiteRT's logger. Passed to aml_compiler_core.");

namespace litert::amlogic {
namespace {

Expected<LrtAmlogicOptionsLogLevel> ParseLogLevel(absl::string_view text) {
  if (text == "off") return kLrtAmlogicLogOff;
  if (text == "error") return kLrtAmlogicLogLevelError;
  if (text == "warn") return kLrtAmlogicLogLevelWarn;
  if (text == "info") return kLrtAmlogicLogLevelInfo;
  if (text == "verbose") return kLrtAmlogicLogLevelVerbose;
  if (text == "debug") return kLrtAmlogicLogLevelDebug;
  return Unexpected(kLiteRtStatusErrorInvalidArgument,
                    "invalid --amlogic_log_level");
}

}  // namespace

Expected<void> UpdateAmlogicOptionsFromFlags(AmlogicOptions& options) {
  const std::string model_path = absl::GetFlag(FLAGS_amlogic_model_path);
  if (!model_path.empty()) {
    LITERT_RETURN_IF_ERROR(options.SetModelPath(model_path));
  }
  const std::string model_name = absl::GetFlag(FLAGS_amlogic_model_name);
  if (!model_name.empty()) {
    LITERT_RETURN_IF_ERROR(options.SetModelName(model_name));
  }
  const std::string log_level = absl::GetFlag(FLAGS_amlogic_log_level);
  if (!log_level.empty()) {
    LITERT_ASSIGN_OR_RETURN(const auto parsed, ParseLogLevel(log_level));
    LITERT_RETURN_IF_ERROR(options.SetLogLevel(parsed));
  }
  return {};
}

LITERT_REGISTER_OPTIONS_PARSER([](Options& options) -> Expected<void> {
  LITERT_ASSIGN_OR_RETURN(auto& aml_opts, options.GetOptions<AmlogicOptions>());
  return UpdateAmlogicOptionsFromFlags(aml_opts);
});

}  // namespace litert::amlogic
