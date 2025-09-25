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

#include "litert/ats/capture.h"

#include <algorithm>
#include <sstream>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::testing {
namespace {

using ::testing::HasSubstr;

TEST(AtsCaptureTest, ModelDetailRow) {
  AtsCaptureEntry::ModelDetail m;
  m.name = "model.tflite";
  m.desc = "small";
  m.precompiled = true;

  std::ostringstream s;
  m.Row(s);
  EXPECT_EQ(s.str(), "model.tflite,small,true");
}

TEST(AtsCaptureTest, AcceleratorDetailRow) {
  AtsCaptureEntry::AcceleratorDetail m;
  m.a_type = AtsCaptureEntry::AcceleratorDetail::AcceleratorType::kNpu;
  m.soc_man = "Qualcomm";
  m.soc_model = "Adreno660";
  m.is_fully_accelerated = true;

  std::ostringstream s;
  m.Row(s);
  EXPECT_EQ(s.str(), "npu,Qualcomm,Adreno660,true");
}

TEST(AtsCaptureTest, LatencyRow) {
  AtsCaptureEntry::Latency l;
  l.Stop(l.Start());

  std::ostringstream s;
  l.Row(s);
  const auto str = s.str();
  absl::string_view str_view = str;
  EXPECT_THAT(str_view.substr(str_view.rfind(',')), HasSubstr("1"));
}

TEST(AtsCaptureTest, NumericsRow) {
  AtsCaptureEntry::Numerics n;
  n.reference_type = AtsCaptureEntry::Numerics::ReferenceType::kCpu;
  n.NewMse(0.123);
  n.NewMse(0.123);

  std::ostringstream s;
  n.Row(s);
  EXPECT_THAT(s.str(), HasSubstr("cpu"));
  EXPECT_THAT(s.str(), HasSubstr("e-01"));
}

TEST(AtsCaptureTest, RunDetailRow) {
  AtsCaptureEntry::RunDetail r;
  r.num_iterations = 10;
  r.status = AtsCaptureEntry::RunDetail::Status::kOk;

  std::ostringstream s;
  r.Row(s);
  EXPECT_EQ(s.str(), "10,ok");
}

TEST(AtsCaptureTest, All) {
  AtsCapture c;
  c.NewEntry();

  std::ostringstream s;
  c.Csv(s);
  const std::vector<std::string> split = absl::StrSplit(s.str(), '\n');
  ASSERT_EQ(split.size(), 3);
  EXPECT_TRUE(split.back().empty());
  EXPECT_EQ(std::count(split[0].begin(), split[0].end(), ','),
            std::count(split[1].begin(), split[1].end(), ','));
}

}  // namespace
}  // namespace litert::testing
