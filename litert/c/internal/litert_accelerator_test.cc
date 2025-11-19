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

#include "litert/c/internal/litert_accelerator.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/internal/litert_accelerator_registration.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/runtime/accelerator.h"
#include "litert/test/matchers.h"

namespace {

using testing::Eq;
using testing::Ge;
using testing::Ne;
using testing::NotNull;

class DummyAccelerator {
 public:
  // `hardware_support` is a bitfield of `LiteRtHwAccelerators` values.
  static LiteRtStatus RegisterAccelerator(int hardware_support,
                                          LiteRtEnvironment env) {
    auto dummy_accelerator = std::make_unique<DummyAccelerator>();
    dummy_accelerator->hardware_support_ = hardware_support;
    LiteRtAccelerator accelerator;
    LiteRtCreateAccelerator(&accelerator);
    LITERT_RETURN_IF_ERROR(
        LiteRtSetAcceleratorGetName(accelerator, DummyAccelerator::GetName));
    LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorGetVersion(
        accelerator, DummyAccelerator::GetVersion));
    LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorGetHardwareSupport(
        accelerator, DummyAccelerator::GetHardwareSupport));
    LITERT_RETURN_IF_ERROR(
        LiteRtRegisterAccelerator(env, accelerator, dummy_accelerator.release(),
                                  DummyAccelerator::Destroy));
    return kLiteRtStatusOk;
  }

  static void Destroy(void* dummy_accelerator) {
    DummyAccelerator* instance =
        reinterpret_cast<DummyAccelerator*>(dummy_accelerator);
    delete instance;
  }

  static LiteRtStatus GetName(LiteRtAccelerator accelerator,
                              const char** name) {
    if (!accelerator || !accelerator->data || !name) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    DummyAccelerator& self =
        *reinterpret_cast<DummyAccelerator*>(accelerator->data);
    if (self.name_.empty()) {
      self.name_.append("Dummy");
      if (self.hardware_support_ & kLiteRtHwAcceleratorCpu) {
        self.name_.append("Cpu");
      }
      if (self.hardware_support_ & kLiteRtHwAcceleratorGpu) {
        self.name_.append("Gpu");
      }
      self.name_.append("Accelerator");
    }
    *name = self.name_.c_str();
    return kLiteRtStatusOk;
  }

  static LiteRtStatus GetVersion(LiteRtAccelerator accelerator,
                                 LiteRtApiVersion* version) {
    if (!version) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    version->major = 1;
    version->minor = 2;
    version->patch = 3;
    return kLiteRtStatusOk;
  }

  static LiteRtStatus GetHardwareSupport(
      LiteRtAccelerator accelerator,
      LiteRtHwAcceleratorSet* supported_hardware) {
    if (!accelerator || !accelerator->data || !supported_hardware) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    const DummyAccelerator& self =
        *reinterpret_cast<DummyAccelerator*>(accelerator->data);
    *supported_hardware = self.hardware_support_;
    return kLiteRtStatusOk;
  }

  int hardware_support_;
  std::string name_;
};

class LiteRtAcceleratorTest : public testing::Test {
 public:
  LiteRtEnvironment env_;
  void SetUp() override {
    LiteRtCreateEnvironment(/*num_options=*/0, nullptr, &env_);
    DummyAccelerator::RegisterAccelerator(kLiteRtHwAcceleratorCpu, env_);
  }

  void TearDown() override { LiteRtDestroyEnvironment(env_); }

  litert::Expected<LiteRtAccelerator> FindAccelerator(
      const absl::string_view accelerator_name) {
    LiteRtParamIndex num_accelerators = 0;
    LITERT_RETURN_IF_ERROR(LiteRtGetNumAccelerators(env_, &num_accelerators));
    LiteRtAccelerator accelerator = nullptr;
    for (LiteRtParamIndex i = 0; i < num_accelerators; ++i) {
      LITERT_RETURN_IF_ERROR(LiteRtGetAccelerator(env_, i, &accelerator));
      const char* name;
      LITERT_RETURN_IF_ERROR(LiteRtGetAcceleratorName(accelerator, &name));
      if (accelerator_name == name) {
        return accelerator;
      }
    }
    return litert::ErrorStatusBuilder(kLiteRtStatusErrorNotFound)
           << "Accelerator " << accelerator_name << " is not registered.";
  }
};

TEST_F(LiteRtAcceleratorTest, IteratingOverAcceleratorsWorks) {
  // CPU accelerator is registered in the SetUp function.
  DummyAccelerator::RegisterAccelerator(kLiteRtHwAcceleratorGpu, env_);

  LiteRtParamIndex num_accelerators = 0;
  ASSERT_THAT(LiteRtGetNumAccelerators(env_, &num_accelerators),
              kLiteRtStatusOk);
  ASSERT_THAT(num_accelerators, Ge(2));

  EXPECT_THAT(LiteRtGetAccelerator(env_, 0, nullptr),
              kLiteRtStatusErrorInvalidArgument);
  LiteRtAccelerator accelerator0 = nullptr;
  EXPECT_THAT(LiteRtGetAccelerator(env_, 0, &accelerator0), kLiteRtStatusOk);
  EXPECT_THAT(accelerator0, NotNull());

  EXPECT_THAT(LiteRtGetAccelerator(env_, 1, nullptr),
              kLiteRtStatusErrorInvalidArgument);
  LiteRtAccelerator accelerator1 = nullptr;
  EXPECT_THAT(LiteRtGetAccelerator(env_, 1, &accelerator1), kLiteRtStatusOk);
  EXPECT_THAT(accelerator1, NotNull());

  EXPECT_THAT(accelerator0, Ne(accelerator1));

  LiteRtAccelerator accelerator2 = nullptr;
  EXPECT_THAT(LiteRtGetAccelerator(env_, num_accelerators, &accelerator2),
              kLiteRtStatusErrorNotFound);
}

TEST_F(LiteRtAcceleratorTest, GetAcceleratorNameWorks) {
  // FindAccelerator uses LiteRtGetAcceleratorName. If it returns a value then
  // the function works.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtAccelerator accelerator,
                              FindAccelerator("DummyCpuAccelerator"));

  const char* name = nullptr;
  EXPECT_THAT(LiteRtGetAcceleratorName(nullptr, &name),
              kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(LiteRtGetAcceleratorName(accelerator, nullptr),
              kLiteRtStatusErrorInvalidArgument);
  // Make the accelerator invalid.
  accelerator->GetName = nullptr;
  EXPECT_THAT(LiteRtGetAcceleratorName(accelerator, &name),
              kLiteRtStatusErrorInvalidArgument);
}

TEST_F(LiteRtAcceleratorTest, GetAcceleratorIdWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtAccelerator accelerator,
                              FindAccelerator("DummyCpuAccelerator"));
  LiteRtAcceleratorId accelerator_id;
  EXPECT_THAT(LiteRtGetAcceleratorId(accelerator, &accelerator_id),
              kLiteRtStatusOk);
  EXPECT_THAT(LiteRtGetAcceleratorId(nullptr, &accelerator_id),
              kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(LiteRtGetAcceleratorId(accelerator, nullptr),
              kLiteRtStatusErrorInvalidArgument);
  // Make the accelerator invalid.
  accelerator->env = nullptr;
  EXPECT_THAT(LiteRtGetAcceleratorId(accelerator, &accelerator_id),
              kLiteRtStatusErrorInvalidArgument);
}

TEST_F(LiteRtAcceleratorTest, GetAcceleratorVersionWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtAccelerator accelerator,
                              FindAccelerator("DummyCpuAccelerator"));
  LiteRtApiVersion version;
  ASSERT_THAT(LiteRtGetAcceleratorVersion(accelerator, &version),
              kLiteRtStatusOk);
  EXPECT_THAT(version.major, Eq(1));
  EXPECT_THAT(version.minor, Eq(2));
  EXPECT_THAT(version.patch, Eq(3));

  EXPECT_THAT(LiteRtGetAcceleratorVersion(nullptr, &version),
              kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(LiteRtGetAcceleratorVersion(accelerator, nullptr),
              kLiteRtStatusErrorInvalidArgument);
  // Make the accelerator invalid.
  accelerator->GetVersion = nullptr;
  EXPECT_THAT(LiteRtGetAcceleratorVersion(accelerator, &version),
              kLiteRtStatusErrorInvalidArgument);
}

TEST_F(LiteRtAcceleratorTest, GetAcceleratorHardwareSupportWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtAccelerator accelerator,
                              FindAccelerator("DummyCpuAccelerator"));
  int hardware_support;
  ASSERT_THAT(
      LiteRtGetAcceleratorHardwareSupport(accelerator, &hardware_support),
      kLiteRtStatusOk);
  EXPECT_THAT(hardware_support & kLiteRtHwAcceleratorCpu, true);
  EXPECT_THAT(hardware_support & kLiteRtHwAcceleratorGpu, false);
  EXPECT_THAT(hardware_support & kLiteRtHwAcceleratorNpu, false);

  EXPECT_THAT(LiteRtGetAcceleratorHardwareSupport(nullptr, &hardware_support),
              kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(LiteRtGetAcceleratorHardwareSupport(accelerator, nullptr),
              kLiteRtStatusErrorInvalidArgument);
  // Make the accelerator invalid.
  accelerator->GetHardwareSupport = nullptr;
  EXPECT_THAT(
      LiteRtGetAcceleratorHardwareSupport(accelerator, &hardware_support),
      kLiteRtStatusErrorInvalidArgument);
}

TEST_F(LiteRtAcceleratorTest,
       IsAcceleratorDelegateResponsibleForJitCompilationWorks) {
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtAccelerator accelerator,
                              FindAccelerator("DummyCpuAccelerator"));
  bool does_jit_compilation;
  ASSERT_THAT(LiteRtIsAcceleratorDelegateResponsibleForJitCompilation(
                  accelerator, &does_jit_compilation),
              kLiteRtStatusOk);
  EXPECT_THAT(does_jit_compilation, false);

  EXPECT_THAT(LiteRtIsAcceleratorDelegateResponsibleForJitCompilation(
                  nullptr, &does_jit_compilation),
              kLiteRtStatusErrorInvalidArgument);
  EXPECT_THAT(LiteRtIsAcceleratorDelegateResponsibleForJitCompilation(
                  accelerator, nullptr),
              kLiteRtStatusErrorInvalidArgument);

  // Add an implementation to the function.
  accelerator->IsTfLiteDelegateResponsibleForJitCompilation =
      [](LiteRtAccelerator, bool* does_jit) {
        *does_jit = true;
        return kLiteRtStatusOk;
      };
  EXPECT_THAT(LiteRtIsAcceleratorDelegateResponsibleForJitCompilation(
                  accelerator, &does_jit_compilation),
              kLiteRtStatusOk);
  EXPECT_THAT(does_jit_compilation, true);
}

}  // namespace
