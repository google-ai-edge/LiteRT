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

#ifdef LITERT_HAS_MTK_SDK
// TODO: Enable once mtk sdk is available.
#include "neuron/api/NeuronAdapter.h"  // IWYU pragma: keep
#endif
#include <gtest/gtest.h>  // IWYU pragma: keep
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "QnnCommon.h"  // from @qairt  // IWYU pragma: keep
#include "System/QnnSystemCommon.h"  // from @qairt  // IWYU pragma: keep

namespace {

void DumpLinkDbg() {
  // PWD AND RUNFILES TREE
  auto pwd = std::filesystem::current_path();
  std::cerr << "pwd: " << std::string(pwd) << "\n";
  for (const auto &file : std::filesystem::recursive_directory_iterator(pwd)) {
    if (file.is_regular_file()) {
      std::cerr << "file: " << std::string(file.path()) << "\n";
    }
  }

  // RPATH OF THIS ELF

  const ElfW(Dyn) *dyn = _DYNAMIC;
  const ElfW(Dyn) *rpath = nullptr;
  const ElfW(Dyn) *runpath = nullptr;
  const char *strtab = nullptr;
  for (; dyn->d_tag != DT_NULL; ++dyn) {
    if (dyn->d_tag == DT_RPATH) {
      rpath = dyn;
    } else if (dyn->d_tag == DT_RUNPATH) {
      runpath = dyn;
    } else if (dyn->d_tag == DT_STRTAB) {
      strtab = (const char *)dyn->d_un.d_val;
    }
  }

  ABSL_CHECK(strtab != nullptr);

  if (rpath != nullptr) {
    printf("RPATH: %s\n", strtab + rpath->d_un.d_val);
  } else if (runpath != nullptr) {
    printf("RUNPATH: %s\n", strtab + runpath->d_un.d_val);
  }
}

TEST(CheckSoAvailabilityTest, CheckQnnSdk) {
  void *lib_qnn_handle = dlopen("libQnnHtp.so", RTLD_LAZY);
  EXPECT_NE(lib_qnn_handle, nullptr);
  if (lib_qnn_handle == nullptr) {
    char *err = dlerror();
    std::cerr << "------ dlerror() -----\n";
    std::cerr << std::string(err) << "\n";
    std::cerr << "------ libQnnHtp.so link error info -----\n";
    DumpLinkDbg();
  }
}

TEST(CheckSoAvailabilityTest, CheckQnnSystemSdk) {
  void *lib_qnn_handle = dlopen("libQnnSystem.so", RTLD_LAZY);
  ASSERT_NE(lib_qnn_handle, nullptr);
  if (lib_qnn_handle == nullptr) {
    char *err = dlerror();
    std::cerr << "------ dlerror() -----\n";
    std::cerr << std::string(err) << "\n";
    std::cerr << "------ libQnnSystem.so link error info -----\n";
    DumpLinkDbg();
  }
}


TEST(CheckSoAvailabilityTest, CheckLatestMediatekSdk) {
  #ifndef LITERT_HAS_MTK_SDK
  GTEST_SKIP() << "MTK SDK is not available.";
  #endif
  void *lib_qnn_handle = dlopen("libneuron_adapter.so", RTLD_LAZY);
  ASSERT_NE(lib_qnn_handle, nullptr);
  if (lib_qnn_handle == nullptr) {
    char *err = dlerror();
    std::cerr << "------ dlerror() -----\n";
    std::cerr << std::string(err) << "\n";
    std::cerr << "------ libneuron_adapter.so link error info -----\n";
    DumpLinkDbg();
  }
}

}  // namespace
