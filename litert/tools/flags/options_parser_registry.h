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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_OPTIONS_PARSER_REGISTRY_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_OPTIONS_PARSER_REGISTRY_H_

#include <functional>
#include <vector>

#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"

namespace litert::tools {

// Registry for options parsers.
//
// Vendor-specific flags libraries can register their options parser function
// with this registry. CLI tools can then iterate over all registered parsers
// to apply options from flags without knowing about specific vendors.
class OptionsParserRegistry {
 public:
  // Signature of the parser function.
  using ParserFunc = std::function<Expected<void>(Options&)>;

  // Returns the singleton instance of the registry.
  static OptionsParserRegistry& GetInstance();

  // Registers a parser function.
  void RegisterParser(ParserFunc parser);

  // Runs all registered parser functions on the provided options object.
  Expected<void> RunAllParsers(Options& options);

 private:
  OptionsParserRegistry() = default;

  std::vector<ParserFunc> parsers_;
};

// Helper class to register a parser function at static initialization time.
class OptionsParserRegistrar {
 public:
  explicit OptionsParserRegistrar(OptionsParserRegistry::ParserFunc parser) {
    OptionsParserRegistry::GetInstance().RegisterParser(std::move(parser));
  }
};

// Macro to register a parser function.
#define LITERT_REGISTER_OPTIONS_PARSER(parser_func) \
  static ::litert::tools::OptionsParserRegistrar    \
  g_options_parser_registrar_##__COUNTER__(parser_func)

}  // namespace litert::tools

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_OPTIONS_PARSER_REGISTRY_H_
