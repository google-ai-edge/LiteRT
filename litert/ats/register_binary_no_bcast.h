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

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_ATS_REGISTER_BINARY_NO_BCAST_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_ATS_REGISTER_BINARY_NO_BCAST_H_

#include <cstddef>

#include "litert/ats/compile_fixture.h"
#include "litert/ats/configure.h"
#include "litert/ats/inference_fixture.h"

namespace litert::testing {

void RegisterBinaryNoBroadcast(const AtsConf& options, size_t& test_id,
                               size_t iters, AtsInferenceTest::Capture& cap);
void RegisterBinaryNoBroadcast(const AtsConf& options, size_t& test_id,
                               size_t iters, AtsCompileTest::Capture& cap);

}  // namespace litert::testing

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_ATS_REGISTER_BINARY_NO_BCAST_H_
