// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {

namespace {

TEST_F(QnnModelTest, Sanity) {
  SetUpQnnModel(::qnn::Options(), "SM8650");
  EXPECT_NE(qnn_manager_ptr_->BackendHandle(), nullptr);
  EXPECT_NE(context_handle_.get(), nullptr);
  EXPECT_NE(qnn_manager_ptr_->Api(), nullptr);
  EXPECT_TRUE(qnn_model_.ValidateOpConfig());
  // no nodes to finalize
  EXPECT_FALSE(qnn_model_.Finalize());
}

}  // namespace
}  // namespace litert::qnn
