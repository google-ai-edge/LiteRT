// Copyright (C) 2026 Samsung Electronics Co. LTD.
// SPDX-License-Identifier: Apache-2.0
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

#pragma once

#include "litert/c/litert_common.h"
#include "litert/c/options/litert_samsung_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_opaque_options.h"

namespace litert::samsung {

class SamsungOptions : public OpaqueOptions {
public:
    using OpaqueOptions::OpaqueOptions;

    SamsungOptions() = delete;

    static const char* Discriminator();

    static Expected<SamsungOptions> Create(OpaqueOptions& options);

    static Expected<SamsungOptions> Create();

private:
    LiteRtSamsungOptions Data() const;
};

}

