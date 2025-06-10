#include "litert/vendors/qualcomm/qnn_backend_test/qnn_backend_creator.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "litert/vendors/qualcomm/core/common.h"

using testing::FloatNear;
using testing::Pointwise;
namespace litert::qnn {
namespace {

TEST(QnnModelTest, QnnBackendCreator) {
  auto options = ::qnn::Options();
  QnnBackendCreator backend_creator(options, "SM8650");
  EXPECT_NE(backend_creator.GetBackendHandle(), nullptr);
  EXPECT_NE(backend_creator.GetContextHandle(), nullptr);
  EXPECT_NE(backend_creator.GetQnnApi(), nullptr);
}

}  // namespace
}  // namespace litert::qnn