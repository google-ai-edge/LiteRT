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

#ifndef ODML_LITERT_LITERT_CC_OPTIONS_LITERT_AMLOGIC_OPTIONS_H_
#define ODML_LITERT_LITERT_CC_OPTIONS_LITERT_AMLOGIC_OPTIONS_H_

#include <memory>
#include <string>

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_amlogic_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/options/litert_concrete_options_base.h"

namespace litert::amlogic {

class AmlogicOptions : public ConcreteOptionsBase {
 public:
  AmlogicOptions() = delete;
  explicit AmlogicOptions(LrtAmlogicOptions options)
      : options_(options, LrtDestroyAmlogicOptions) {}

  AmlogicOptions(const AmlogicOptions&) = delete;
  AmlogicOptions& operator=(const AmlogicOptions&) = delete;
  AmlogicOptions(AmlogicOptions&&) = default;
  AmlogicOptions& operator=(AmlogicOptions&&) = default;
  ~AmlogicOptions() override = default;

  static const char* Discriminator() { return LrtAmlogicOptionsGetIdentifier(); }

  static Expected<AmlogicOptions> Create() {
    LrtAmlogicOptions options = nullptr;
    LITERT_RETURN_IF_ERROR(LrtCreateAmlogicOptions(&options));
    return AmlogicOptions(options);
  }

  LrtAmlogicOptions Get() const { return options_.get(); }

  LiteRtStatus GetOpaqueOptionsData(
      const char** identifier, void** payload,
      void (**payload_deleter)(void*)) const override {
    return LrtGetOpaqueAmlogicOptionsData(Get(), identifier, payload,
                                      payload_deleter);
  }

  Expected<void> SetLogLevel(LrtAmlogicOptionsLogLevel log_level) {
    LITERT_RETURN_IF_ERROR(LrtAmlogicOptionsSetLogLevel(Get(), log_level));
    return {};
  }
  Expected<LrtAmlogicOptionsLogLevel> GetLogLevel() const {
    LrtAmlogicOptionsLogLevel log_level;
    LITERT_RETURN_IF_ERROR(LrtAmlogicOptionsGetLogLevel(Get(), &log_level));
    return log_level;
  }

  Expected<void> SetModelPath(const std::string& model_path) {
    LITERT_RETURN_IF_ERROR(LrtAmlogicOptionsSetModelPath(Get(), model_path.c_str()));
    return {};
  }
  Expected<const char*> GetModelPath() const {
    const char* model_path = nullptr;
    LITERT_RETURN_IF_ERROR(LrtAmlogicOptionsGetModelPath(Get(), &model_path));
    return model_path;
  }

  Expected<void> SetModelName(const std::string& model_name) {
    LITERT_RETURN_IF_ERROR(LrtAmlogicOptionsSetModelName(Get(), model_name.c_str()));
    return {};
  }
  Expected<const char*> GetModelName() const {
    const char* model_name = nullptr;
    LITERT_RETURN_IF_ERROR(LrtAmlogicOptionsGetModelName(Get(), &model_name));
    return model_name;
  }

 private:
  std::unique_ptr<LrtAmlogicOptionsT, void (*)(LrtAmlogicOptions)> options_;
};

}  // namespace litert::amlogic

#endif  // ODML_LITERT_LITERT_CC_OPTIONS_LITERT_AMLOGIC_OPTIONS_H_
