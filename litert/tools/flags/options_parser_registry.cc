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

#include "litert/tools/flags/options_parser_registry.h"

#include <utility>

#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_options.h"

namespace litert::tools {

OptionsParserRegistry& OptionsParserRegistry::GetInstance() {
  static OptionsParserRegistry* registry = new OptionsParserRegistry();
  return *registry;
}

void OptionsParserRegistry::RegisterParser(ParserFunc parser) {
  parsers_.push_back(std::move(parser));
}

Expected<void> OptionsParserRegistry::RunAllParsers(Options& options) {
  for (const auto& parser : parsers_) {
    LITERT_RETURN_IF_ERROR(parser(options));
  }
  return {};
}

}  // namespace litert::tools
