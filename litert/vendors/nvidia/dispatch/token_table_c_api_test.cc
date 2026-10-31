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

// The token table registration's contract: malformed records are refused,
// the dispatch must be initialized, and a table nobody awaits is not taken.
// The step feed that takes one needs a compiled decode step (the benchmark's
// 26B run shows "step feed ready (... from the runtime ...)").

#include "litert/vendors/nvidia/dispatch/token_table_c_api.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/vendors/c/litert_dispatch_api.h"

namespace litert::nvidia {
namespace {

LiteRtRuntimeContext RuntimeContext() {
  LiteRtRuntimeContext context{};
  context.get_environment_options = [](LiteRtEnvironment,
                                       LiteRtEnvironmentOptions*) {
    return kLiteRtStatusErrorNotFound;
  };
  return context;
}

LiteRtDispatchNvidiaTokenTable ValidRecord(const std::vector<uint8_t>& table,
                                           const std::vector<float>& scales) {
  LiteRtDispatchNvidiaTokenTable record{};
  record.vocab = 4;
  record.dim = 8;
  record.bits = 4;
  record.table = table.data();
  record.table_bytes = table.size();
  record.table_fd = -1;
  record.scales = scales.data();
  record.scales_bytes = scales.size() * sizeof(float);
  record.scale_block = 0;
  record.multiplier = 2.0f;
  return record;
}

TEST(TokenTableCApiTest, RefusesMalformedRecordsBeforeAnything) {
  const std::vector<uint8_t> table(4 * 8 / 2, 0x11);
  const std::vector<float> scales(4, 0.5f);
  EXPECT_EQ(LiteRtDispatchNvidiaRegisterTokenTable(nullptr),
            kLiteRtStatusErrorInvalidArgument);
  LiteRtDispatchNvidiaTokenTable record = ValidRecord(table, scales);
  record.table = nullptr;  // neither bytes nor a file
  EXPECT_EQ(LiteRtDispatchNvidiaRegisterTokenTable(&record),
            kLiteRtStatusErrorInvalidArgument);
  record = ValidRecord(table, scales);
  record.table_bytes = 15;  // not vocab * dim / 2
  EXPECT_EQ(LiteRtDispatchNvidiaRegisterTokenTable(&record),
            kLiteRtStatusErrorInvalidArgument);
  record = ValidRecord(table, scales);
  record.scales_bytes = 8;  // not one FP32 per row
  EXPECT_EQ(LiteRtDispatchNvidiaRegisterTokenTable(&record),
            kLiteRtStatusErrorInvalidArgument);
  record = ValidRecord(table, scales);
  record.scale_block = 3;  // blocks must divide the row
  record.scales_bytes = 4 * 2 * sizeof(uint16_t);
  EXPECT_EQ(LiteRtDispatchNvidiaRegisterTokenTable(&record),
            kLiteRtStatusErrorInvalidArgument);
  record = ValidRecord(table, scales);
  record.scale_block = 4;                       // FP16 [vocab, dim / 4]
  record.scales_bytes = 4 * 2 * sizeof(float);  // but sized as FP32
  EXPECT_EQ(LiteRtDispatchNvidiaRegisterTokenTable(&record),
            kLiteRtStatusErrorInvalidArgument);
  record = ValidRecord(table, scales);
  record.multiplier = 0.0f;
  EXPECT_EQ(LiteRtDispatchNvidiaRegisterTokenTable(&record),
            kLiteRtStatusErrorInvalidArgument);
  record = ValidRecord(table, scales);
  record.bits = 3;  // 4 or 2
  EXPECT_EQ(LiteRtDispatchNvidiaRegisterTokenTable(&record),
            kLiteRtStatusErrorInvalidArgument);
  record = ValidRecord(table, scales);
  record.bits = 2;  // then half the bytes
  EXPECT_EQ(LiteRtDispatchNvidiaRegisterTokenTable(&record),
            kLiteRtStatusErrorInvalidArgument);
  // Per-layer tables: the same contract.
  const void* tables[2] = {table.data(), table.data()};
  const float* row_scales[2] = {scales.data(), scales.data()};
  LiteRtDispatchNvidiaPerLayerTokenTables per_layer{};
  per_layer.vocab = 4;
  per_layer.width = 8;
  per_layer.layers = 2;
  per_layer.bits = 4;
  per_layer.tables = tables;
  per_layer.table_bytes = table.size();
  per_layer.row_scales = row_scales;
  per_layer.multiplier = 1.0f;
  EXPECT_EQ(LiteRtDispatchNvidiaRegisterPerLayerTokenTables(nullptr),
            kLiteRtStatusErrorInvalidArgument);
  LiteRtDispatchNvidiaPerLayerTokenTables bad = per_layer;
  bad.table_bytes = 15;
  EXPECT_EQ(LiteRtDispatchNvidiaRegisterPerLayerTokenTables(&bad),
            kLiteRtStatusErrorInvalidArgument);
  bad = per_layer;
  bad.layers = 0;
  EXPECT_EQ(LiteRtDispatchNvidiaRegisterPerLayerTokenTables(&bad),
            kLiteRtStatusErrorInvalidArgument);
  EXPECT_EQ(LiteRtDispatchNvidiaRegisterPerLayerTokenTables(&per_layer),
            kLiteRtStatusErrorUnsupported);
  // Well formed, but the dispatch is not initialized.
  record = ValidRecord(table, scales);
  EXPECT_EQ(LiteRtDispatchNvidiaRegisterTokenTable(&record),
            kLiteRtStatusErrorUnsupported);
}

TEST(TokenTableCApiTest, TakesOnlyAwaitedTables) {
  LiteRtInterface raw_interface = nullptr;
  ASSERT_EQ(LiteRtDispatchQueryInterface(kLiteRtInterfaceBasic,
                                         {LITERT_DISPATCH_ABI_VERSION_MAJOR,
                                          LITERT_DISPATCH_ABI_VERSION_MINOR,
                                          LITERT_DISPATCH_ABI_VERSION_PATCH},
                                         &raw_interface),
            kLiteRtStatusOk);
  const auto* interface =
      static_cast<const LiteRtDispatchInterface_V1*>(raw_interface);
  ASSERT_NE(interface, nullptr);
  LiteRtRuntimeContext runtime = RuntimeContext();
  ASSERT_EQ(interface->initialize(&runtime, nullptr, nullptr), kLiteRtStatusOk);
  const std::vector<uint8_t> table(4 * 8 / 2, 0x11);
  const std::vector<float> scales(4, 0.5f);
  LiteRtDispatchNvidiaTokenTable record = ValidRecord(table, scales);
  // No decode step awaits a table [4, 8]: nothing is copied.
  EXPECT_EQ(LiteRtDispatchNvidiaRegisterTokenTable(&record),
            kLiteRtStatusErrorNotFound);
  const std::vector<uint16_t> block_scales(4 * 2, 0x3c00);
  record.scales = block_scales.data();
  record.scales_bytes = block_scales.size() * sizeof(uint16_t);
  record.scale_block = 4;
  EXPECT_EQ(LiteRtDispatchNvidiaRegisterTokenTable(&record),
            kLiteRtStatusErrorNotFound);
  const void* tables[2] = {table.data(), table.data()};
  const float* row_scales[2] = {scales.data(), scales.data()};
  LiteRtDispatchNvidiaPerLayerTokenTables per_layer{};
  per_layer.vocab = 4;
  per_layer.width = 8;
  per_layer.layers = 2;
  per_layer.bits = 4;
  per_layer.tables = tables;
  per_layer.table_bytes = table.size();
  per_layer.row_scales = row_scales;
  per_layer.multiplier = 1.0f;
  EXPECT_EQ(LiteRtDispatchNvidiaRegisterPerLayerTokenTables(&per_layer),
            kLiteRtStatusErrorNotFound);
  EXPECT_EQ(interface->initialize(nullptr, nullptr, nullptr), kLiteRtStatusOk);
}

}  // namespace
}  // namespace litert::nvidia
