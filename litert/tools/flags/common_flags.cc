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

#include "litert/tools/flags/common_flags.h"

#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl

ABSL_FLAG(std::string, model, "", "Path to flatbuffer file.");

ABSL_FLAG(std::string, soc_manufacturer, "ExampleSocManufacturer",
          "String identifier of SoC manufacturer (e.g., GoogleTensor, "
          "Qualcomm).");

ABSL_FLAG(std::string, soc_model, "ExampleSocModel", "Target SoC model.");

ABSL_FLAG(
    std::vector<std::string>, libs,
    std::vector<std::string>({"third_party/odml/litert/litert/vendors/examples",
                              "third_party/odml/litert/litert/vendors/qualcomm/"
                              "compiler",
                              "third_party/odml/litert/litert/vendors/mediatek/"
                              "compiler",
                              "third_party/odml/litert/litert/vendors/"
                              "google_tensor/compiler"}),
    "List of directories in which to search for suitable "
    "shared libraries.");

ABSL_FLAG(std::vector<std::string>, o, std::vector<std::string>({"-"}),
          "Path to files for output, \"-\" indicates standard out, "
          "\"--\" for standard err, \"none\" for null stream.");

ABSL_FLAG(std::string, err, "--",
          "Path to file for err output, \"-\" indicates standard out, "
          "\"--\" for standard err, \"none\" for null stream.");
