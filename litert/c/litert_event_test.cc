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

#include "litert/c/litert_event.h"

#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_event_type.h"

namespace {

TEST(LiteRtEventTest, RejectsNullCreateOutput) {
  EXPECT_EQ(
      LiteRtCreateManagedEvent(nullptr, LiteRtEventTypeSyncFenceFd, nullptr),
      kLiteRtStatusErrorInvalidArgument);
}

TEST(LiteRtEventTest, RejectsNullEventAccessors) {
  LiteRtEventType type = LiteRtEventTypeSyncFenceFd;
  EXPECT_EQ(LiteRtGetEventEventType(nullptr, &type),
            kLiteRtStatusErrorInvalidArgument);

  int fd = -1;
  EXPECT_EQ(LiteRtGetEventSyncFenceFd(nullptr, &fd),
            kLiteRtStatusErrorInvalidArgument);

  bool is_signaled = false;
  EXPECT_EQ(LiteRtIsEventSignaled(nullptr, &is_signaled),
            kLiteRtStatusErrorInvalidArgument);

  EXPECT_EQ(LiteRtWaitEvent(nullptr, 0), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtSignalEvent(nullptr), kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtDupFdEvent(nullptr, &fd), kLiteRtStatusErrorInvalidArgument);
}

TEST(LiteRtEventTest, RejectsNullOutputParams) {
  EXPECT_EQ(LiteRtGetEventEventType(nullptr, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtGetEventSyncFenceFd(nullptr, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtIsEventSignaled(nullptr, nullptr),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtDupFdEvent(nullptr, nullptr),
            kLiteRtStatusErrorInvalidArgument);
}

}  // namespace
