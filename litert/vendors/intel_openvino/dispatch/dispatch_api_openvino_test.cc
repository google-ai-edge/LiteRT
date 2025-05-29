// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"       // from @com_google_absl
#include "absl/types/span.h"    // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_expected.h"
#include "litert/core/filesystem.h"
#include "litert/test/common.h"
#include "litert/test/testdata/simple_model_test_vectors.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"

TEST(OpenVino, DispatchApi) {
  LiteRtDispatchOption dispatch_option = {
      /*.name=*/kDispatchOptionSharedLibraryDir,
      // For now path for Linux is current dir, can also be /opt/...
      /*.value=*/*litert::ToLiteRtAny(std::any("./")),  // path for Android -->
                                                        // /data/local/tmp")),
  };

  ASSERT_EQ(
      LiteRtDispatchInitialize(/*options=*/&dispatch_option, /*num_options=*/1),
      kLiteRtStatusOk);

  const char* vendor_id;
  EXPECT_EQ(LiteRtDispatchGetVendorId(&vendor_id), kLiteRtStatusOk);
  LITERT_LOG(LITERT_INFO, "vendor_id:%s", vendor_id);

  const char* build_id;
  EXPECT_EQ(LiteRtDispatchGetBuildId(&build_id), kLiteRtStatusOk);
  LITERT_LOG(LITERT_INFO, "build_id:%s ", build_id);

  LiteRtApiVersion api_version;
  EXPECT_EQ(LiteRtDispatchGetApiVersion(&api_version), kLiteRtStatusOk);
}
