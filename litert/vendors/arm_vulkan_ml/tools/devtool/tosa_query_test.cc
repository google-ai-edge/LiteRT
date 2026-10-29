// Copyright 2026 The LiteRT Authors. All Rights Reserved.
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

#include "litert/vendors/arm_vulkan_ml/tools/devtool/tool.h"
#include "gtest/gtest.h"
namespace litert::arm_vulkan_ml::tool {
namespace {
Json Database() {
  return Json::parse(R"json({
    "schema_version":1,"instruction_set":"TOSA.001000.1","status":"reference",
    "profiles":["PRO-INT","PRO-FP"],"extensions":{"EXT-BF16":{}},"counts":{},
    "enums":{"mode_t":{"values":[
      {"name":"BASE","additional_requirements":{"any_of":[{"all_of":[]}]}},
      {"name":"EXTRA","additional_requirements":{"any_of":[{"all_of":["EXT-BF16"]}]}}
    ]}},
    "operators":[{"name":"ADD","arguments":[{"name":"mode","tosa_type":"mode_t"}],
      "variants":[
        {"id":"fp32","types":{"input1":{"element_type":"fp32_t"}},"parameters":{},"requirements":{"any_of":[{"all_of":["PRO-INT"]},{"all_of":["PRO-FP"]}]}},
        {"id":"bf16","types":{"input1":{"element_type":"bf16_t"}},"parameters":{},"requirements":{"any_of":[{"all_of":["PRO-FP","EXT-BF16"]}]}}
      ]}]
  })json");
}
TEST(TosaQuery, ExactTypesAndProfileOrExtensionAnd) {
  Options options; options.values["op"]="add";
  Json result; ASSERT_TRUE(QueryTosa(Database(),options,result));
  EXPECT_EQ(result["candidate_count"],2);
  options.repeated["capability"]={"PRO-INT"};
  ASSERT_TRUE(QueryTosa(Database(),options,result));EXPECT_EQ(result["candidate_count"],1);
  options.repeated["type"]={"input1=bf16_t"};
  ASSERT_TRUE(QueryTosa(Database(),options,result));EXPECT_EQ(result["candidate_count"],0);
  options.repeated["capability"]={"PRO-FP","EXT-BF16"};
  ASSERT_TRUE(QueryTosa(Database(),options,result));EXPECT_EQ(result["candidate_count"],1);
  options.repeated["capability"]={"EXT-BF16"};
  ASSERT_TRUE(QueryTosa(Database(),options,result));EXPECT_EQ(result["candidate_count"],0);
}
TEST(TosaQuery, EnumRequirementsAndNoImplicitHardwareClaim) {
  Options options; options.values["op"]="ADD";options.repeated["enum"]={"mode=EXTRA"};
  options.repeated["capability"]={"PRO-INT"};Json result;
  ASSERT_TRUE(QueryTosa(Database(),options,result));EXPECT_EQ(result["candidate_count"],0);
  options.repeated["capability"]={"PRO-INT","EXT-BF16"};
  ASSERT_TRUE(QueryTosa(Database(),options,result));EXPECT_EQ(result["candidate_count"],1);
  EXPECT_EQ(result["device_support"],"not_queried");EXPECT_EQ(result["shape_constraints"],"not_evaluated");
}
TEST(TosaQuery, RejectsUnknownFiltersAndMalformedDatabase) {
  Options options;options.values["op"]="ADD";Json result;
  for (const auto& filter : {"bad=fp32_t","input1=int4","input1="}) {
    options.repeated["type"]={filter};EXPECT_FALSE(QueryTosa(Database(),options,result));
  }
  options.repeated.clear();options.repeated["capability"]={"EXT-INT4"};
  EXPECT_FALSE(QueryTosa(Database(),options,result));
  options.repeated.clear();options.repeated["enum"]={"mode=UNKNOWN"};
  EXPECT_FALSE(QueryTosa(Database(),options,result));
  auto db=Database();db["operators"][0]["variants"][0]["types"]=nullptr;
  EXPECT_FALSE(QueryTosa(db,options,result));
  db=Database();db["schema_version"]=2;EXPECT_FALSE(QueryTosa(db,options,result));
  db=Database();db["profiles"]=Json::object();EXPECT_FALSE(QueryTosa(db,options,result));
  db=Database();db["operators"][0]["arguments"][0]["tosa_type"]=nullptr;
  EXPECT_FALSE(QueryTosa(db,options,result));
  db=Database();db["enums"]["mode_t"]["values"][0]["additional_requirements"]["any_of"]="invalid";
  EXPECT_FALSE(QueryTosa(db,options,result));
}
TEST(TosaQuery, ParsesRepeatableFiltersAndBareJsonFlag) {
  char binary[]="tool",command[]="tosa-query",type[]="--type",a[]="input1=fp32_t",b[]="input2=fp32_t",json[]="--json";
  char* argv[]={binary,command,type,a,type,b,json};
  auto options=ParseOptions(7,argv);ASSERT_TRUE(options);
  EXPECT_EQ(options->repeated["type"].size(),2);EXPECT_EQ(options->Get("json"),"1");
}
}  // namespace
}  // namespace litert::arm_vulkan_ml::tool
