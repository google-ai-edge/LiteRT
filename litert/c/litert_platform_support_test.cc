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

#include "litert/c/litert_platform_support.h"

#include <cstring>

#include <gtest/gtest.h>
#include "litert/c/internal/litert_platform_support_internal.h"
#include "litert/c/litert_common.h"

namespace {

using litert::internal::DlOps;
using litert::internal::ProbeFastRpcSupport;
using litert::internal::ProbeIonSupport;

struct FakeDlState {
  void* handle = nullptr;
  const char* missing_symbol = nullptr;
  int ion_open_fd = -1;
  int ion_close_calls = 0;
  int last_closed_fd = -1;
  int dlclose_calls = 0;
};

FakeDlState* g_state = nullptr;

class ScopedFakeDlState {
 public:
  explicit ScopedFakeDlState(FakeDlState* state) { g_state = state; }
  ~ScopedFakeDlState() { g_state = nullptr; }
};

int FakeIonOpen() { return g_state->ion_open_fd; }

int FakeIonClose(int fd) {
  ++g_state->ion_close_calls;
  g_state->last_closed_fd = fd;
  return 0;
}

void FakeDummySymbol() {}

void* FakeDlOpen(const char* /*filename*/, int /*flags*/) {
  return g_state->handle;
}

void* FakeDlSym(void* /*handle*/, const char* symbol) {
  if (g_state->missing_symbol &&
      std::strcmp(symbol, g_state->missing_symbol) == 0) {
    return nullptr;
  }
  if (std::strcmp(symbol, "ion_open") == 0) {
    return reinterpret_cast<void*>(&FakeIonOpen);
  }
  if (std::strcmp(symbol, "ion_close") == 0) {
    return reinterpret_cast<void*>(&FakeIonClose);
  }
  return reinterpret_cast<void*>(&FakeDummySymbol);
}

int FakeDlClose(void* /*handle*/) {
  ++g_state->dlclose_calls;
  return 0;
}

constexpr DlOps kFakeOps = {FakeDlOpen, FakeDlSym, FakeDlClose};

TEST(LiteRtPlatformSupportTest, ProbeIonSupportFailsWhenDlopenReturnsNull) {
  FakeDlState state;
  ScopedFakeDlState scope(&state);
  EXPECT_FALSE(ProbeIonSupport(kFakeOps));
  EXPECT_EQ(state.dlclose_calls, 0);
}

TEST(LiteRtPlatformSupportTest, ProbeIonSupportFailsWhenSymbolsMissing) {
  int dummy_handle = 1;
  for (const char* missing : {"ion_open", "ion_close", "ion_alloc_fd"}) {
    FakeDlState state{.handle = &dummy_handle, .missing_symbol = missing};
    ScopedFakeDlState scope(&state);
    EXPECT_FALSE(ProbeIonSupport(kFakeOps)) << "missing: " << missing;
    EXPECT_EQ(state.ion_close_calls, 0);
    EXPECT_EQ(state.dlclose_calls, 1);
  }
}

TEST(LiteRtPlatformSupportTest, ProbeIonSupportFailsWhenIonOpenFails) {
  int dummy_handle = 1;
  FakeDlState state{.handle = &dummy_handle, .ion_open_fd = -1};
  ScopedFakeDlState scope(&state);
  EXPECT_FALSE(ProbeIonSupport(kFakeOps));
  EXPECT_EQ(state.ion_close_calls, 0);
  EXPECT_EQ(state.dlclose_calls, 1);
}

TEST(LiteRtPlatformSupportTest, ProbeIonSupportSucceedsAndClosesFd) {
  int dummy_handle = 1;
  FakeDlState state{.handle = &dummy_handle, .ion_open_fd = 42};
  ScopedFakeDlState scope(&state);
  EXPECT_TRUE(ProbeIonSupport(kFakeOps));
  EXPECT_EQ(state.ion_close_calls, 1);
  EXPECT_EQ(state.last_closed_fd, 42);
  EXPECT_EQ(state.dlclose_calls, 1);
}

TEST(LiteRtPlatformSupportTest, ProbeFastRpcSupportFailsWhenDlopenReturnsNull) {
  FakeDlState state;
  ScopedFakeDlState scope(&state);
  EXPECT_FALSE(ProbeFastRpcSupport(kFakeOps));
  EXPECT_EQ(state.dlclose_calls, 0);
}

TEST(LiteRtPlatformSupportTest, ProbeFastRpcSupportFailsWhenSymbolsMissing) {
  int dummy_handle = 1;
  for (const char* missing : {"rpcmem_alloc", "rpcmem_free", "rpcmem_to_fd"}) {
    FakeDlState state{.handle = &dummy_handle, .missing_symbol = missing};
    ScopedFakeDlState scope(&state);
    EXPECT_FALSE(ProbeFastRpcSupport(kFakeOps)) << "missing: " << missing;
    EXPECT_EQ(state.dlclose_calls, 1);
  }
}

TEST(LiteRtPlatformSupportTest,
     ProbeFastRpcSupportSucceedsWhenAllSymbolsFound) {
  int dummy_handle = 1;
  FakeDlState state{.handle = &dummy_handle};
  ScopedFakeDlState scope(&state);
  EXPECT_TRUE(ProbeFastRpcSupport(kFakeOps));
  EXPECT_EQ(state.dlclose_calls, 1);
}

TEST(LiteRtPlatformSupportTest, PublicEntryPointsMatchCompileTimeAndCache) {
#if !LITERT_HAS_ION_SUPPORT
  EXPECT_FALSE(LiteRtHasIonSupport());
#endif
#if !LITERT_HAS_FASTRPC_SUPPORT
  EXPECT_FALSE(LiteRtHasFastRpcSupport());
#endif
  EXPECT_EQ(LiteRtHasIonSupport(), LiteRtHasIonSupport());
  EXPECT_EQ(LiteRtHasFastRpcSupport(), LiteRtHasFastRpcSupport());
}

}  // namespace
