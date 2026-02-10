// Copyright 2025 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_CC_OPTIONS_HELPER_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_CC_OPTIONS_HELPER_H_

#include <type_traits>
#include <utility>

#include "litert/c/litert_common.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/litert_options.h"

namespace litert {

// Idea "parse" all the options in a tuple with a variable length
// template param

template <class... Discriminated>
auto ParseOptions(LiteRtOptions options) {
  static constexpr auto kErr = kLiteRtStatusErrorInvalidArgument;

  auto opts = options ? Expected<litert::Options>(options, OwnHandle::kNo)
                      : Error(kErr, "Null litert options");
  auto opq =
      opts ? opts->GetOpaqueOptions() : Error(kErr, "Null opaque options");

  return std::make_tuple(
      std::move(opts), std::move(opq), [&]() -> Expected<Discriminated> {
        static_assert(std::is_base_of<OpaqueOptions, Discriminated>::value);
        if (opq) {
          return FindOpaqueOptions<Discriminated>(*opq);
        }
        return opq.Error();
      }()...);
}

}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_VENDORS_CC_OPTIONS_HELPER_H_
