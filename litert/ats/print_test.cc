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

#include "litert/ats/print.h"

#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_split.h"  // from @com_google_absl

namespace litert::testing {
namespace {

using ::testing::HasSubstr;

struct TestPrintable1 : public Printable<int, int64_t> {
  using Printable::Fields;

  int i = 1;
  int64_t ii = 2;

  TestPrintable1() : Printable("TestPrintable1", "i", "ii") {}

 private:
  Fields GetFields() const override { return Fields{i, ii}; }
};

struct TestPrintable2 : public Printable<std::string> {
  using Printable::Fields;

  std::string s = "hello";

  TestPrintable2() : Printable("TestPrintable2", "s") {}  // NOLINT

 private:
  Fields GetFields() const override { return Fields{s}; }
};

struct TestPrintableRow : public PrintableRow<TestPrintable1, TestPrintable2> {
  using PrintableRow::Printables;

  TestPrintable1 p1;
  TestPrintable2 p2;

 private:
  Printables GetPrintables() const override {
    return Printables{std::cref(p1), std::cref(p2)};
  }

  std::string Name() const override { return "AnEntry"; }
};

struct TestPrintableCollection : public PrintableCollection<TestPrintableRow> {
 private:
  absl::string_view Name() const override { return "TestPrintableCollection"; }
};

TEST(AtsPrintableTest, Csv) {
  TestPrintable1 p1;

  {
    std::ostringstream s;
    p1.CsvAppendHeader(s);
    EXPECT_EQ(s.str(), "i,ii");
  }

  {
    std::ostringstream s;
    p1.CsvAppendRow(s);
    EXPECT_EQ(s.str(), "1,2");
  }
}

TEST(AtsPrintableTest, Print) {
  TestPrintable1 p1;

  std::ostringstream s;
  p1.Print(s);
  EXPECT_THAT(s.str(), HasSubstr("i: 1"));
  EXPECT_THAT(s.str(), HasSubstr("ii: 2"));
}

TEST(AtsPrintableRowTest, Csv) {
  TestPrintableRow r;

  {
    std::ostringstream s;
    r.CsvHeader(s);
    EXPECT_EQ(s.str(), "i,ii,s\n");
  }

  {
    std::ostringstream s;
    r.CsvRow(s);
    EXPECT_EQ(s.str(), "1,2,hello\n");
  }
}

TEST(AtsPrintableRowTest, Print) {
  TestPrintableRow r;

  std::ostringstream s;
  r.Print(s);

  const auto str = s.str();

  EXPECT_THAT(str, HasSubstr("i: 1"));
  EXPECT_THAT(str, HasSubstr("ii: 2"));
  EXPECT_THAT(str, HasSubstr("s: hello"));
  EXPECT_THAT(str, HasSubstr("TestPrintable1"));
  EXPECT_THAT(str, HasSubstr("TestPrintable2"));
  EXPECT_THAT(str, HasSubstr("AnEntry"));
}

TEST(AtsPrintableCollectionTest, Csv) {
  TestPrintableCollection c;
  c.NewEntry();
  auto& r2 = c.NewEntry();
  r2.p1.i = 2;
  r2.p1.ii = 4;
  r2.p2.s = "world";

  std::ostringstream s;
  c.Csv(s);
  const std::vector<std::string> split = absl::StrSplit(s.str(), '\n');
  ASSERT_EQ(split.size(), 4);
  EXPECT_TRUE(split.back().empty());

  EXPECT_EQ(split[0], "i,ii,s");
  EXPECT_EQ(split[1], "1,2,hello");
  EXPECT_EQ(split[2], "2,4,world");
}

TEST(AtsPrintableCollectionTest, Print) {
  TestPrintableCollection c;
  c.NewEntry();
  auto& r2 = c.NewEntry();
  r2.p1.i = 2;
  r2.p1.ii = 4;
  r2.p2.s = "world";

  std::ostringstream s;
  c.Print(s);

  const auto str = s.str();

  EXPECT_THAT(str, HasSubstr("TestPrintable1"));
  EXPECT_THAT(str, HasSubstr("TestPrintable2"));
  EXPECT_THAT(str, HasSubstr("AnEntry"));
  EXPECT_THAT(str, HasSubstr("TestPrintableCollection"));
  EXPECT_THAT(str, HasSubstr("i: 1"));
  EXPECT_THAT(str, HasSubstr("ii: 2"));
  EXPECT_THAT(str, HasSubstr("s: hello"));
  EXPECT_THAT(str, HasSubstr("i: 2"));
  EXPECT_THAT(str, HasSubstr("ii: 4"));
  EXPECT_THAT(str, HasSubstr("s: world"));
}

}  // namespace
}  // namespace litert::testing
