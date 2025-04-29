// Copyright 2024 Google LLC.
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

#include "litert/test/common.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <ios>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/const_init.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_model_predicates.h"
#include "litert/core/filesystem.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tsl/platform/platform.h"

namespace litert::testing {

Expected<UniqueTestDirectory> UniqueTestDirectory::Create() {
  constexpr size_t kMaxTries = 1000;
  ABSL_CONST_INIT static absl::Mutex mutex(absl::kConstInit);

  // We don't want multiple threads to create the same directory.
  absl::MutexLock l(&mutex);

  auto tmp_dir = std::filesystem::temp_directory_path();
  std::random_device dev;
  std::mt19937 prng(dev());
  std::uniform_int_distribution<uint64_t> rand(0);
  std::stringstream ss;

  for (auto i = 0; i < kMaxTries; ++i) {
    ss.clear();
    ss << std::hex << rand(prng);
    auto path = tmp_dir / ss.str();
    if (std::filesystem::create_directory(path)) {
      LITERT_LOG(LITERT_INFO, "Created unique temporary directory %s",
                 path.c_str());
      return UniqueTestDirectory(path);
    }
  }

  return Error(kLiteRtStatusErrorRuntimeFailure,
               "Could not create a unique temporary directory");
}

UniqueTestDirectory::~UniqueTestDirectory() {
  std::filesystem::remove_all(tmpdir_);
}

#ifdef __ANDROID__
// Test is run on a mobile device and files are stored under
// "/data/local/tmp/runfiles".
constexpr char kBaseDir[] = "/data/local/tmp/runfiles";
#else
constexpr char kBaseDir[] = "";
#endif  // __ANDROID__

constexpr absl::string_view kLiteRtDir = "litert";
constexpr absl::string_view kInternalPrefx = "third_party/odml/litert";

std::string GetTestFilePath(absl::string_view filename) {
  static constexpr absl::string_view kTestDataDir = "test/testdata/";
  if constexpr (!tsl::kIsOpenSource) {
    return internal::Join(
        {kBaseDir, kInternalPrefx, kLiteRtDir, kTestDataDir, filename});
  } else {
    return internal::Join({kBaseDir, kLiteRtDir, kTestDataDir, filename});
  }
}

std::string GetTfliteFilePath(absl::string_view filename) {
  if constexpr (!tsl::kIsOpenSource) {
    return internal::Join({kBaseDir, "third_party/tensorflow/lite/", filename});
  } else {
    return internal::Join({kBaseDir, "external/tflite/", filename});
  }
}

std::string GetLiteRtPath(absl::string_view rel_path) {
  if constexpr (!tsl::kIsOpenSource) {
    return internal::Join({kBaseDir, kInternalPrefx, kLiteRtDir, rel_path});
  } else {
    return internal::Join({kBaseDir, kLiteRtDir, rel_path});
  }
}

Model LoadTestFileModel(absl::string_view filename) {
  LITERT_ASSIGN_OR_ABORT(auto model,
                         Model::CreateFromFile(GetTestFilePath(filename)));
  return model;
}

Expected<TflRuntime::Ptr> TflRuntime::CreateFromFlatBuffer(
    internal::FlatbufferWrapper::Ptr flatbuffer) {
  ::tflite::Interpreter::Ptr interp;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(flatbuffer->FlatbufferModel(), resolver)(&interp);
  if (interp == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure);
  }
  return TflRuntime::Ptr(
      new TflRuntime(std::move(flatbuffer), std::move(interp)));
}

}  // namespace litert::testing
