// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "litert/vendors/qualcomm/qnn_backend_test/test_utils.h"

namespace litert::qnn {
namespace {

INSTANTIATE_TEST_SUITE_P(, QnnModelTest, GetDefaultQnnModelParams(),
                         QnnTestPrinter);

TEST_P(QnnModelTest, Sanity) {
  ASSERT_TRUE(qnn_manager_ptr_->BackendHandle());
  ASSERT_TRUE(context_handle_.get());
  ASSERT_TRUE(qnn_manager_ptr_->Api());
  ASSERT_TRUE(qnn_model_.ValidateOpConfig());
  // no nodes to finalize
  ASSERT_FALSE(qnn_model_.Finalize());
}

}  // namespace
}  // namespace litert::qnn
