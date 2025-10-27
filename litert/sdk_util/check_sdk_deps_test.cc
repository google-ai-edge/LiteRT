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

#include <assert.h>
#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#include <stdio.h>

#include <filesystem>  // NOLINT
#include <iostream>
#include <string>

#include <gtest/gtest.h>  // IWYU pragma: keep
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/internal/litert_logging.h"
#include "litert/test/common.h"
#include "QnnCommon.h"  // from @qairt  // IWYU pragma: keep
#include "System/QnnSystemCommon.h"  // from @qairt  // IWYU pragma: keep

#if !LITERT_IS_OSS
#include "neuron/api/NeuronAdapter.h"  // IWYU pragma: keep
#endif

namespace {

using ::litert::IsDbg;
using ::litert::testing::IsOss;
using ::testing::TestWithParam;
using ::testing::Values;

void DumpLinkDbg(void* lib_handle, absl::string_view lib_name) {
  if (lib_handle != nullptr) {
    std::string dl_info(512, '\0');
    dlinfo(lib_handle, RTLD_DI_ORIGIN, dl_info.data());
    std::cerr << "------ Loaded .so @ -----\n";
    std::cerr << absl::StreamFormat("%s/%s", dl_info, lib_name) << "\n";
    return;
  }

  // PWD AND RUNFILES TREE
  auto pwd = std::filesystem::current_path();
  std::cerr << "pwd: " << std::string(pwd) << "\n";
  for (const auto& file : std::filesystem::recursive_directory_iterator(pwd)) {
    if (file.is_regular_file()) {
      std::cerr << "file: " << std::string(file.path()) << "\n";
    }
  }

  // DL API OUTPUT
  char* err = dlerror();
  std::cerr << "------ dlerror() -----\n";
  std::cerr << std::string(err) << "\n";

  // RPATH OF THIS ELF

  const ElfW(Dyn)* dyn = _DYNAMIC;
  const ElfW(Dyn)* rpath = nullptr;
  const ElfW(Dyn)* runpath = nullptr;
  const char* strtab = nullptr;
  for (; dyn->d_tag != DT_NULL; ++dyn) {
    if (dyn->d_tag == DT_RPATH) {
      rpath = dyn;
    } else if (dyn->d_tag == DT_RUNPATH) {
      runpath = dyn;
    } else if (dyn->d_tag == DT_STRTAB) {
      strtab = (const char*)dyn->d_un.d_val;
    }
  }

  ABSL_CHECK(strtab != nullptr);

  if (rpath != nullptr) {
    printf("RPATH: %s\n", strtab + rpath->d_un.d_val);
  } else if (runpath != nullptr) {
    printf("RUNPATH: %s\n", strtab + runpath->d_un.d_val);
  }
}

template <bool InternalOnly>
class CheckSdkTestImpl : public TestWithParam<absl::string_view> {
 public:
  void SetUp() override {
    if constexpr (InternalOnly && IsOss()) {
      GTEST_SKIP() << "Skipping test for non-OSS builds.";
    }
  }
  void DoTest() {
    void* lib_handle = dlopen(GetParam().data(), RTLD_LAZY);
    EXPECT_NE(lib_handle, nullptr);
    if constexpr (IsDbg()) {
      DumpLinkDbg(lib_handle, GetParam());
    }
  }
};

using CheckSdkTest = CheckSdkTestImpl<false>;
TEST_P(CheckSdkTest, CheckSdk) { DoTest(); }

using InternalOnlyCheckSdkTest = CheckSdkTestImpl<true>;
TEST_P(InternalOnlyCheckSdkTest, CheckSdk) { DoTest(); }

////////////////////////////////////////////////////////////////////////////////

INSTANTIATE_TEST_SUITE_P(Qairt, CheckSdkTest,
                         Values("libQnnHtp.so", "libQnnSystem.so"));

INSTANTIATE_TEST_SUITE_P(NueronAdapter, InternalOnlyCheckSdkTest,
                         Values("libneuron_adapter.so"));

}  // namespace
