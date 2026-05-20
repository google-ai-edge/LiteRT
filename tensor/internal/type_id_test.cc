/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor/internal/type_id.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace litert::tensor::internal {

template <int N, class T>
struct DummyTemplate {};

struct DummyA {};
struct DummyB {};

// Names that may clash with keywords
struct Destruct {};
struct Reunion {};
template <>
struct DummyTemplate<3, struct enumerator> {};

namespace {

using ::testing::StrEq;

TEST(TypeIdTest, TypeNameExtractionWorks) {
  EXPECT_THAT((TypeId::Name<DummyTemplate<3, int>>()),
              StrEq("litert::tensor::internal::DummyTemplate<3,int>"));
  EXPECT_THAT(TypeId::Name<DummyA>(),
              StrEq("litert::tensor::internal::DummyA"));
  EXPECT_THAT(TypeId::Name<DummyB>(),
              StrEq("litert::tensor::internal::DummyB"));

  EXPECT_THAT(TypeId::Name<Destruct>(),
              StrEq("litert::tensor::internal::Destruct"));
  EXPECT_THAT(TypeId::Name<Reunion>(),
              StrEq("litert::tensor::internal::Reunion"));
  EXPECT_THAT((TypeId::Name<DummyTemplate<3, struct enumerator>>()),
              StrEq("litert::tensor::internal::DummyTemplate<3,enumerator>"));
}

TEST(TypeIdTest, Uniqueness) {
  // Verify distinct concrete types have distinct TypeIds.
  EXPECT_NE(TypeId::Get<DummyA>(), TypeId::Get<DummyB>());
  EXPECT_NE(TypeId::Get<int>(), TypeId::Get<float>());

  // Verify template specializations are precisely disambiguated. If the
  // compiler macro stops including template arguments, these checks will fail.
  EXPECT_NE((TypeId::Get<DummyTemplate<1, int>>()),
            (TypeId::Get<DummyTemplate<2, int>>()));
  EXPECT_NE((TypeId::Get<DummyTemplate<100, int>>()),
            (TypeId::Get<DummyTemplate<200, int>>()));
}

TEST(TypeIdTest, Equality) {
  EXPECT_EQ(TypeId::Get<DummyA>(), TypeId::Get<DummyA>());
  EXPECT_EQ((TypeId::Get<DummyTemplate<1, int>>()),
            (TypeId::Get<DummyTemplate<1, int>>()));
}

TEST(TypeIdTest, Decay) {
  // Verify const and reference types decay transparently.
  EXPECT_EQ(TypeId::Get<const DummyA>(), TypeId::Get<DummyA>());
  EXPECT_EQ(TypeId::Get<DummyA&>(), TypeId::Get<DummyA>());
}

}  // namespace
}  // namespace litert::tensor::internal
