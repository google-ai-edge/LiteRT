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

#include "litert/core/model/subgraph_refs.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/core/model/model.h"
#include "litert/core/util/flatbuffer_tools.h"
#include "litert/test/matchers.h"
#include "tflite/converter/schema/schema_generated.h"
// copybara:uncomment_begin(google-only)
// #include "absl/container/flat_hash_set.h"  // from @com_google_absl
// #include "absl/strings/match.h"  // from @com_google_absl
// #include "absl/strings/str_cat.h"  // from @com_google_absl
// #include "absl/strings/string_view.h"  // from @com_google_absl
// #include "absl/strings/strip.h"  // from @com_google_absl
// #include "flatbuffers/reflection_generated.h"  // from @flatbuffers
// #include "tflite/converter/schema/schema_reflection_data.h"
// copybara:uncomment_end

namespace litert::internal {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAreArray;

template <typename OptionsT>
void SetOptions(LiteRtOpT& op, tflite::BuiltinOptions type, OptionsT options) {
  TflOptions union_options;
  union_options.type = type;
  union_options.Set(std::move(options));
  SetTflOptions(op, std::move(union_options));
}

template <typename OptionsT>
void SetOptions2(LiteRtOpT& op, tflite::BuiltinOptions2 type,
                 OptionsT options) {
  TflOptions2 union_options;
  union_options.type = type;
  union_options.Set(std::move(options));
  SetTflOptions2(op, std::move(union_options));
}

// An op of each option type that holds subgraph indices, with the indices it
// is expected to report and the schema fields that hold them, as
// "<OptionsTable>.<field>".
struct RefCase {
  std::string name;
  LiteRtOpCode code;
  void (*set_options)(LiteRtOpT&);
  std::vector<int32_t> refs;
  std::vector<std::string> fields;
};

std::vector<RefCase> AllRefCases() {
  return {
      {"If",
       kLiteRtOpCodeTflIf,
       [](LiteRtOpT& op) {
         tflite::IfOptionsT o;
         o.then_subgraph_index = 1;
         o.else_subgraph_index = 2;
         SetOptions(op, tflite::BuiltinOptions_IfOptions, std::move(o));
       },
       {1, 2},
       {"IfOptions.then_subgraph_index",
        "IfOptions.else_subgraph_index"}},
      {"While",
       kLiteRtOpCodeTflWhile,
       [](LiteRtOpT& op) {
         tflite::WhileOptionsT o;
         o.cond_subgraph_index = 1;
         o.body_subgraph_index = 2;
         SetOptions(op, tflite::BuiltinOptions_WhileOptions, std::move(o));
       },
       {1, 2},
       {"WhileOptions.cond_subgraph_index",
        "WhileOptions.body_subgraph_index"}},
      {"CallOnce",
       kLiteRtOpCodeTflCallOnce,
       [](LiteRtOpT& op) {
         tflite::CallOnceOptionsT o;
         o.init_subgraph_index = 3;
         SetOptions(op, tflite::BuiltinOptions_CallOnceOptions, std::move(o));
       },
       {3},
       {"CallOnceOptions.init_subgraph_index"}},
      {"Composite",
       kLiteRtOpCodeShloComposite,
       [](LiteRtOpT& op) {
         tflite::StableHLOCompositeOptionsT o;
         o.decomposition_subgraph_index = 1;
         SetOptions2(op, tflite::BuiltinOptions2_StableHLOCompositeOptions,
                     std::move(o));
       },
       {1},
       {"StableHLOCompositeOptions.decomposition_subgraph_index"}},
      {"Reduce",
       kLiteRtOpCodeShloReduce,
       [](LiteRtOpT& op) {
         tflite::StablehloReduceOptionsT o;
         o.body_subgraph_index = 2;
         SetOptions2(op, tflite::BuiltinOptions2_StablehloReduceOptions,
                     std::move(o));
       },
       {2},
       {"StablehloReduceOptions.body_subgraph_index"}},
      {"ReduceWindow",
       kLiteRtOpCodeShloWindow,
       [](LiteRtOpT& op) {
         tflite::StablehloReduceWindowOptionsT o;
         o.body_subgraph_index = 3;
         SetOptions2(op, tflite::BuiltinOptions2_StablehloReduceWindowOptions,
                     std::move(o));
       },
       {3},
       {"StablehloReduceWindowOptions.body_subgraph_index"}},
      {"Scatter",
       kLiteRtOpCodeShloScatter,
       [](LiteRtOpT& op) {
         tflite::StablehloScatterOptionsT o;
         o.update_computation_subgraph_index = 1;
         SetOptions2(op, tflite::BuiltinOptions2_StablehloScatterOptions,
                     std::move(o));
       },
       {1},
       {"StablehloScatterOptions.update_computation_subgraph_index"}},
      {"Sort",
       kLiteRtOpCodeShloSort,
       [](LiteRtOpT& op) {
         tflite::StablehloSortOptionsT o;
         o.comparator_subgraph_index = 2;
         SetOptions2(op, tflite::BuiltinOptions2_StablehloSortOptions,
                     std::move(o));
       },
       {2},
       {"StablehloSortOptions.comparator_subgraph_index"}},
      {"CustomCall",
       kLiteRtOpCodeShloCustomCall,
       [](LiteRtOpT& op) {
         tflite::StablehloCustomCallOptionsT o;
         o.called_computations = {1, 3};
         SetOptions2(op, tflite::BuiltinOptions2_StablehloCustomCallOptions,
                     std::move(o));
       },
       {1, 3},
       {"StablehloCustomCallOptions.called_computations"}},
      {"StablehloWhile",
       kLiteRtOpCodeShloWhile,
       [](LiteRtOpT& op) {
         tflite::StablehloWhileOptionsT o;
         o.cond_subgraph_index = 2;
         o.body_subgraph_index = 3;
         SetOptions2(op, tflite::BuiltinOptions2_StablehloWhileOptions,
                     std::move(o));
       },
       {2, 3},
       {"StablehloWhileOptions.cond_subgraph_index",
        "StablehloWhileOptions.body_subgraph_index"}},
      {"Case",
       kLiteRtOpCodeShloCase,
       [](LiteRtOpT& op) {
         tflite::StablehloCaseOptionsT o;
         o.branch_subgraph_indices = {3, 1, 2};
         SetOptions2(op, tflite::BuiltinOptions2_StablehloCaseOptions,
                     std::move(o));
       },
       {3, 1, 2},
       {"StablehloCaseOptions.branch_subgraph_indices"}},
  };
}

class SubgraphRefsParamTest : public ::testing::TestWithParam<RefCase> {};

TEST_P(SubgraphRefsParamTest, ReportsAndRemapsEveryIndex) {
  const RefCase& ref_case = GetParam();
  LiteRtOpT op;
  op.SetOpCode(ref_case.code);
  ref_case.set_options(op);

  EXPECT_THAT(SubgraphRefs(op), ElementsAreArray(ref_case.refs));
  EXPECT_TRUE(HasSubgraphRefs(op));
  EXPECT_EQ(IsRegionOp(op), ref_case.code != kLiteRtOpCodeShloComposite);

  // Map index i to i + 10.
  const std::vector<int32_t> old_to_new = {10, 11, 12, 13};
  LITERT_ASSERT_OK(RemapSubgraphRefs(op, old_to_new));
  std::vector<int32_t> expected;
  for (const int32_t ref : ref_case.refs) {
    expected.push_back(ref + 10);
  }
  EXPECT_THAT(SubgraphRefs(op), ElementsAreArray(expected));
}

TEST_P(SubgraphRefsParamTest, RemapFailureLeavesOpUnchanged) {
  const RefCase& ref_case = GetParam();
  LiteRtOpT op;
  op.SetOpCode(ref_case.code);
  ref_case.set_options(op);

  // Every index maps to -1, i.e. "unmapped".
  const std::vector<int32_t> old_to_new = {-1, -1, -1, -1};
  EXPECT_FALSE(RemapSubgraphRefs(op, old_to_new));
  EXPECT_THAT(SubgraphRefs(op), ElementsAreArray(ref_case.refs));
}

INSTANTIATE_TEST_SUITE_P(AllOptionTypes, SubgraphRefsParamTest,
                         ::testing::ValuesIn(AllRefCases()),
                         [](const ::testing::TestParamInfo<RefCase>& info) {
                           return info.param.name;
                         });

TEST(SubgraphRefsTest, OpWithoutOptionsHasNoRefs) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflWhile);
  EXPECT_THAT(SubgraphRefs(op), IsEmpty());
  EXPECT_FALSE(HasSubgraphRefs(op));
  EXPECT_FALSE(IsRegionOp(op));
}

TEST(SubgraphRefsTest, CompositeWithoutDecompositionHasNoRefs) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeShloComposite);
  tflite::StableHLOCompositeOptionsT options;
  options.decomposition_subgraph_index = -1;
  SetOptions2(op, tflite::BuiltinOptions2_StableHLOCompositeOptions,
              std::move(options));

  EXPECT_THAT(SubgraphRefs(op), IsEmpty());
  EXPECT_EQ(CompositeDecompositionIndex(op), -1);
  // The nullable -1 survives remapping.
  LITERT_ASSERT_OK(RemapSubgraphRefs(op, {}));
  EXPECT_EQ(CompositeDecompositionIndex(op), -1);
}

TEST(SubgraphRefsTest, NonNullableMinusOneIsReported) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeTflCallOnce);
  tflite::CallOnceOptionsT options;
  options.init_subgraph_index = -1;
  SetOptions(op, tflite::BuiltinOptions_CallOnceOptions, std::move(options));

  EXPECT_THAT(SubgraphRefs(op), ElementsAre(-1));
  EXPECT_FALSE(RemapSubgraphRefs(op, {0}));
}

TEST(SubgraphRefsTest, RetargetRewritesMatchingSlotsOnly) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeShloCase);
  tflite::StablehloCaseOptionsT options;
  options.branch_subgraph_indices = {1, 2, 1};
  SetOptions2(op, tflite::BuiltinOptions2_StablehloCaseOptions,
              std::move(options));

  EXPECT_EQ(RetargetSubgraphRefs(op, /*from=*/1, /*to=*/5), 2);
  EXPECT_THAT(SubgraphRefs(op), ElementsAre(5, 2, 5));
}

TEST(SubgraphRefsTest, ClearCompositeDecomposition) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeShloComposite);
  tflite::StableHLOCompositeOptionsT options;
  options.name = "odml.rms_norm";
  options.decomposition_subgraph_index = 2;
  SetOptions2(op, tflite::BuiltinOptions2_StableHLOCompositeOptions,
              std::move(options));

  ClearCompositeDecomposition(op);

  EXPECT_EQ(CompositeDecompositionIndex(op), -1);
  EXPECT_EQ(GetTflOptions2(op).AsStableHLOCompositeOptions()->name,
            "odml.rms_norm");
}

TEST(SubgraphRefsTest, ClearCompositeDecompositionIgnoresOtherOps) {
  LiteRtOpT op;
  op.SetOpCode(kLiteRtOpCodeShloWindow);
  tflite::StablehloReduceWindowOptionsT options;
  options.body_subgraph_index = 2;
  SetOptions2(op, tflite::BuiltinOptions2_StablehloReduceWindowOptions,
              std::move(options));

  ClearCompositeDecomposition(op);

  EXPECT_THAT(SubgraphRefs(op), ElementsAre(2));
  EXPECT_EQ(CompositeDecompositionIndex(op), std::nullopt);
}

// copybara:uncomment_begin(google-only)
// // Fails when an op option table in the TFLite schema gains a field named
// // `*_subgraph_index`, `*_subgraph_indices` or `called_computations` that no
// // `RefCase` covers.
// TEST(SubgraphRefsTest, RegistryCoversAllSchemaIndexFields) {
//   ASSERT_EQ(tflite::schema_reflection_data_size(), 1);
//   const reflection::Schema* schema =
//       reflection::GetSchema(tflite::schema_reflection_data_create()->data);
//   ASSERT_NE(schema, nullptr);
// 
//   // Tables that are members of the op option unions.
//   absl::flat_hash_set<std::string> option_tables;
//   for (const reflection::Enum* union_enum : *schema->enums()) {
//     const absl::string_view union_name = union_enum->name()->string_view();
//     if (union_name != "tflite.BuiltinOptions" &&
//         union_name != "tflite.BuiltinOptions2") {
//       continue;
//     }
//     for (const reflection::EnumVal* value : *union_enum->values()) {
//       option_tables.insert(std::string(value->name()->string_view()));
//     }
//   }
//   ASSERT_FALSE(option_tables.empty());
// 
//   std::vector<std::string> schema_fields;
//   for (const reflection::Object* object : *schema->objects()) {
//     absl::string_view table = object->name()->string_view();
//     if (!absl::ConsumePrefix(&table, "tflite.") ||
//         !option_tables.contains(table)) {
//       continue;
//     }
//     for (const reflection::Field* field : *object->fields()) {
//       const absl::string_view field_name = field->name()->string_view();
//       if (absl::EndsWith(field_name, "_subgraph_index") ||
//           absl::EndsWith(field_name, "_subgraph_indices") ||
//           field_name == "called_computations") {
//         schema_fields.push_back(absl::StrCat(table, ".", field_name));
//       }
//     }
//   }
// 
//   std::vector<std::string> covered_fields;
//   for (const RefCase& ref_case : AllRefCases()) {
//     covered_fields.insert(covered_fields.end(), ref_case.fields.begin(),
//                           ref_case.fields.end());
//   }
//   EXPECT_THAT(covered_fields, UnorderedElementsAreArray(schema_fields));
// }
// copybara:uncomment_end

}  // namespace
}  // namespace litert::internal
