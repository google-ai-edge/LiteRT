// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift_delegate/tflite/model_builder_helper.h"

#include <cstdint>
#include <initializer_list>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/types/span.h"  // from @com_google_absl
#include "tflite/core/c/common.h"
#include "tflite/testing/matchers.h"

namespace litert::ml_drift {
namespace {

using ::testing::ElementsAre;
using ::testing::tflite::SimpleConstTensor;

struct IntArrayWrapper {
  IntArrayWrapper() = delete;
  IntArrayWrapper(std::initializer_list<int> dims) {
    array = TfLiteIntArrayCreate(dims.size());
    int i = 0;
    for (int dim : dims) array->data[i++] = dim;
  }
  ~IntArrayWrapper() { TfLiteIntArrayFree(array); }
  TfLiteIntArray* array;
};

TEST(ModelBuilderHelperTest, CopyDataDifferentSize) {
  int32_t data[4] = {1, 2, 3, 4};
  SimpleConstTensor tensor(kTfLiteInt32, {4}, absl::MakeSpan(data));

  int32_t dst[4];
  CopyData(tensor, dst);

  EXPECT_THAT(dst, ElementsAre(1, 2, 3, 4));
}

TEST(ModelBuilderHelperTest, Broadcastable1d) {
  IntArrayWrapper dims_4d = {256, 128, 64, 32};

  IntArrayWrapper dims_1d0 = /*         */{01};
  EXPECT_TRUE(IsBroadcastable(dims_1d0.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_1d0.array));

  IntArrayWrapper dims_1d1 = /*         */{32};
  EXPECT_TRUE(IsBroadcastable(dims_1d1.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_1d1.array));
}

TEST(ModelBuilderHelperTest, NotBroadcastable1d) {
  IntArrayWrapper dims_4d = {256, 128, 64, 32};

  IntArrayWrapper dims_1d = /*          */{31};
  //                                       ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_1d.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_1d.array));
}

TEST(ModelBuilderHelperTest, Broadcastable2d) {
  IntArrayWrapper dims_4d = {256, 128, 64, 32};

  IntArrayWrapper dims_2d0 = /*     */{01, 01};
  EXPECT_TRUE(IsBroadcastable(dims_2d0.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_2d0.array));

  IntArrayWrapper dims_2d1 = /*     */{01, 32};
  EXPECT_TRUE(IsBroadcastable(dims_2d1.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_2d1.array));

  IntArrayWrapper dims_2d2 = /*     */{64, 01};
  EXPECT_TRUE(IsBroadcastable(dims_2d2.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_2d2.array));

  IntArrayWrapper dims_2d3 = /*     */{64, 32};
  EXPECT_TRUE(IsBroadcastable(dims_2d3.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_2d3.array));
}

TEST(ModelBuilderHelperTest, NotBroadcastable2d) {
  IntArrayWrapper dims_4d = {256, 128, 64, 32};

  IntArrayWrapper dims_2d0 = /*     */{01, 31};
  //                                       ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_2d0.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_2d0.array));

  IntArrayWrapper dims_2d1 = /*     */{63, 01};
  //                                   ^^     not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_2d1.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_2d1.array));

  IntArrayWrapper dims_2d2 = /*     */{63, 31};
  //                                   ^^  ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_2d2.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_2d2.array));

  IntArrayWrapper dims_2d3 = /*     */{63, 32};
  //                                   ^^     not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_2d3.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_2d3.array));

  IntArrayWrapper dims_2d4 = /*     */{64, 31};
  //                                       ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_2d4.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_2d4.array));
}

TEST(ModelBuilderHelperTest, Broadcastable3d) {
  IntArrayWrapper dims_4d = {256, 128, 64, 32};

  IntArrayWrapper dims_3d0 = /**/{001, 01, 01};
  EXPECT_TRUE(IsBroadcastable(dims_3d0.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_3d0.array));

  IntArrayWrapper dims_3d1 = /**/{001, 01, 32};
  EXPECT_TRUE(IsBroadcastable(dims_3d1.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_3d1.array));

  IntArrayWrapper dims_3d2 = /**/{001, 64, 01};
  EXPECT_TRUE(IsBroadcastable(dims_3d2.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_3d2.array));

  IntArrayWrapper dims_3d3 = /**/{001, 64, 32};
  EXPECT_TRUE(IsBroadcastable(dims_3d3.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_3d3.array));

  IntArrayWrapper dims_3d4 = /**/{128, 01, 01};
  EXPECT_TRUE(IsBroadcastable(dims_3d4.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_3d4.array));

  IntArrayWrapper dims_3d5 = /**/{128, 01, 32};
  EXPECT_TRUE(IsBroadcastable(dims_3d5.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_3d5.array));

  IntArrayWrapper dims_3d6 = /**/{128, 64, 01};
  EXPECT_TRUE(IsBroadcastable(dims_3d6.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_3d6.array));

  IntArrayWrapper dims_3d7 = /**/{128, 64, 32};
  EXPECT_TRUE(IsBroadcastable(dims_3d7.array, dims_4d.array));
  EXPECT_TRUE(IsBroadcastable(dims_4d.array, dims_3d7.array));
}

TEST(ModelBuilderHelperTest, NotBroadcastable3d) {
  IntArrayWrapper dims_4d = {256, 128, 64, 32};

  IntArrayWrapper dims_3d0 = /**/{001, 01, 31};
  //                                       ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d0.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d0.array));

  IntArrayWrapper dims_3d1 = /**/{001, 63, 01};
  //                                   ^^     not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d1.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d1.array));

  IntArrayWrapper dims_3d2 = /**/{001, 63, 31};
  //                                   ^^  ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d2.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d2.array));

  IntArrayWrapper dims_3d3 = /**/{001, 63, 32};
  //                                   ^^     not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d3.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d3.array));

  IntArrayWrapper dims_3d4 = /**/{001, 64, 31};
  //                                       ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d4.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d4.array));

  IntArrayWrapper dims_3d5 = /**/{127, 01, 01};
  //                              ^^^         not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d5.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d5.array));

  IntArrayWrapper dims_3d6 = /**/{127, 01, 31};
  //                              ^^^      ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d6.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d6.array));

  IntArrayWrapper dims_3d7 = /**/{127, 01, 32};
  //                              ^^^         not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d7.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d7.array));

  IntArrayWrapper dims_3d8 = /**/{127, 63, 01};
  //                              ^^^  ^^     not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d8.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d8.array));

  IntArrayWrapper dims_3d9 = /**/{127, 63, 31};
  //                              ^^^  ^^  ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3d9.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3d9.array));

  IntArrayWrapper dims_3da = /**/{127, 63, 32};
  //                              ^^^  ^^     not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3da.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3da.array));

  IntArrayWrapper dims_3db = /**/{127, 64, 01};
  //                              ^^^         not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3db.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3db.array));

  IntArrayWrapper dims_3dc = /**/{127, 64, 31};
  //                              ^^^      ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3dc.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3dc.array));

  IntArrayWrapper dims_3dd = /**/{127, 64, 32};
  //                              ^^^         not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3dd.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3dd.array));

  IntArrayWrapper dims_3de = /**/{128, 01, 31};
  //                                       ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3de.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3de.array));

  IntArrayWrapper dims_3df = /**/{128, 63, 01};
  //                                   ^^     not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3df.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3df.array));

  IntArrayWrapper dims_3dg = /**/{128, 63, 31};
  //                                   ^^  ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3dg.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3dg.array));

  IntArrayWrapper dims_3dh = /**/{128, 63, 32};
  //                                   ^^     not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3dh.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3dh.array));

  IntArrayWrapper dims_3di = /**/{128, 64, 31};
  //                                       ^^ not broadcastable
  EXPECT_FALSE(IsBroadcastable(dims_3di.array, dims_4d.array));
  EXPECT_FALSE(IsBroadcastable(dims_4d.array, dims_3di.array));
}

}  // namespace
}  // namespace litert::ml_drift
