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

#include <algorithm>
#include <cctype>
#include <iostream>
#include <set>

#include "litert/cc/litert_macros.h"

namespace litert::arm_vulkan_ml::tool {
namespace {
Error Invalid(const std::string& text) {
  return Error(kLiteRtStatusErrorInvalidArgument, text);
}
bool Strings(const Json& array) {
  return array.is_array() && std::all_of(array.begin(), array.end(),
                                       [](const Json& v) { return v.is_string(); });
}
bool Requirements(const Json& requirements) {
  if (!requirements.is_object() || !requirements.contains("any_of") ||
      !requirements["any_of"].is_array()) return false;
  for (const auto& clause : requirements["any_of"]) {
    if (!clause.is_object() || !clause.contains("all_of") ||
        !Strings(clause["all_of"])) return false;
  }
  return true;
}
bool Met(const Json& requirements, const std::set<std::string>& capabilities) {
  for (const auto& clause : requirements["any_of"]) {
    bool satisfied = true;
    for (const auto& name : clause["all_of"])
      satisfied &= capabilities.count(name.get<std::string>()) != 0;
    if (satisfied) return true;
  }
  return false;
}
// Validate every field consumed below before typed access. The host extractor
// owns full JSON-schema validation; this prevents malformed phone inputs from
// reaching JSON exceptions/abort paths in a no-exceptions executable.
bool ValidDatabase(const Json& db) {
  if (!db.is_object() || !db.contains("schema_version") || db["schema_version"] != 1 ||
      !db.contains("instruction_set") || db["instruction_set"] != "TOSA.001000.1") return false;
  for (const auto* key : {"extensions", "enums", "counts"})
    if (!db.contains(key) || !db[key].is_object()) return false;
  if (!db.contains("profiles") || !Strings(db["profiles"])) return false;
  if (!db.contains("status") || !db["status"].is_string() ||
      !db.contains("operators") || !db["operators"].is_array()) return false;
  std::set<std::string> names;
  for (const auto& op : db["operators"]) {
    if (!op.is_object() || !op.contains("name") || !op["name"].is_string() ||
        !names.insert(op["name"].get<std::string>()).second ||
        !op.contains("variants") || !op["variants"].is_array() ||
        !op.contains("arguments") || !op["arguments"].is_array()) return false;
    for (const auto& arg : op["arguments"])
      if (!arg.is_object() || !arg.contains("name") || !arg["name"].is_string() ||
          !arg.contains("tosa_type") || !arg["tosa_type"].is_string()) return false;
    for (const auto& row : op["variants"]) {
      if (!row.is_object() || !row.contains("id") || !row["id"].is_string() ||
          !row.contains("types") || !row["types"].is_object() ||
          !row.contains("parameters") || !row["parameters"].is_object() ||
          !row.contains("requirements") || !Requirements(row["requirements"])) return false;
      for (const auto& type : row["types"])
        if (!type.is_object() || !type.contains("element_type") ||
            !type["element_type"].is_string()) return false;
    }
  }
  for (const auto& definition : db["enums"]) {
    if (!definition.is_object() || !definition.contains("values") ||
        !definition["values"].is_array()) return false;
    for (const auto& v : definition["values"])
      if (!v.is_object() || !v.contains("name") || !v["name"].is_string() ||
          !v.contains("additional_requirements") || !Requirements(v["additional_requirements"])) return false;
  }
  return true;
}
Expected<std::map<std::string, std::string>> Assignments(const Options& options,
                                                     const std::string& key) {
  std::map<std::string, std::string> result;
  auto found = options.repeated.find(key);
  if (found == options.repeated.end()) return result;
  for (const auto& text : found->second) {
    auto equals = text.find('=');
    if (equals == std::string::npos || equals == 0 || equals + 1 == text.size())
      return Invalid("Expected --" + key + " argument=value");
    result[text.substr(0, equals)] = text.substr(equals + 1);
  }
  return result;
}
}  // namespace

Expected<void> QueryTosa(const Json& db, const Options& options, Json& result) {
  if (!ValidDatabase(db)) return Invalid("Malformed or unsupported TOSA database (expected schema 1, TOSA.001000.1)");
  std::string name = options.Get("op");
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  result = {{"instruction_set", db["instruction_set"]},
            {"query_scope", "specification_candidates_only"},
            {"shape_constraints", "not_evaluated"},
            {"lowering_support", "not_assessed"},
            {"device_support", "not_queried"}};
  if (name.empty()) {
    if (!options.repeated.empty()) return Invalid("Filters require --op");
    result["counts"] = db["counts"];
    result["database_status"] = db["status"];
    return {};
  }
  const Json* op = nullptr;
  for (const auto& entry : db["operators"]) if (entry["name"] == name) op = &entry;
  if (!op) return Invalid("Unknown mapped operator: " + name);
  LITERT_ASSIGN_OR_RETURN(auto types, Assignments(options, "type"));
  LITERT_ASSIGN_OR_RETURN(auto enums, Assignments(options, "enum"));
  std::set<std::string> columns, elements, capabilities;
  for (const auto& row : (*op)["variants"])
    for (auto it = row["types"].begin(); it != row["types"].end(); ++it) columns.insert(it.key());
  for (const auto& entry : db["operators"])
    for (const auto& row : entry["variants"])
      for (const auto& type : row["types"]) elements.insert(type["element_type"].get<std::string>());
  for (const auto& filter : types) {
    if (!columns.count(filter.first)) return Invalid("Type filter must name a registry table column: " + filter.first);
    if (!elements.count(filter.second)) return Invalid("Unknown TOSA element type: " + filter.second);
  }
  const auto caps = options.repeated.find("capability");
  const bool filter_caps = caps != options.repeated.end();
  if (filter_caps) for (const auto& cap : caps->second) {
    if (std::find(db["profiles"].begin(), db["profiles"].end(), Json(cap)) == db["profiles"].end() && !db["extensions"].contains(cap))
      return Invalid("Profile/extension is absent from this SPIR-V mapping: " + cap);
    capabilities.insert(cap);
  }
  std::vector<Json> enum_constraints;
  for (const auto& filter : enums) {
    const Json* arg = nullptr;
    for (const auto& candidate : (*op)["arguments"])
      if (candidate["name"] == filter.first) arg = &candidate;
    if (!arg || !db["enums"].contains((*arg)["tosa_type"].get<std::string>()))
      return Invalid("Not an enumerated argument: " + filter.first);
    const auto& definition = db["enums"][(*arg)["tosa_type"].get<std::string>()];
    const Json* value = nullptr;
    for (const auto& candidate : definition["values"])
      if (candidate["name"] == filter.second) value = &candidate;
    if (!value) return Invalid("Unknown enum value: " + filter.second);
    enum_constraints.push_back((*value)["additional_requirements"]);
  }
  result.update(*op);
  result["variants"] = Json::array();
  for (const auto& row : (*op)["variants"]) {
    bool matches = true;
    for (const auto& filter : types)
      matches &= row["types"].contains(filter.first) &&
          row["types"][filter.first]["element_type"] == filter.second;
    for (const auto& filter : enums)
      matches &= !row["parameters"].contains(filter.first) || row["parameters"][filter.first] == filter.second;
    if (filter_caps) {
      matches &= Met(row["requirements"], capabilities);
      for (const auto& constraint : enum_constraints) matches &= Met(constraint, capabilities);
    }
    if (matches) result["variants"].push_back(row);
  }
  result["candidate_count"] = result["variants"].size();
  return {};
}
Expected<void> TosaQuery(const Options& options, Json& report) {
  if (options.Get("database").empty()) return Invalid("tosa-query requires --database FILE");
  LITERT_ASSIGN_OR_RETURN(auto db, imagenet::ReadJson(options.Get("database")));
  Json result;
  LITERT_RETURN_IF_ERROR(QueryTosa(db, options, result));
  LITERT_ASSIGN_OR_RETURN(result["database_sha256"], imagenet::HashFile(options.Get("database")));
  report.update(result);
  return {};
}
void PrintTosaQuery(const Json& report) {
  if (!report.contains("name")) {
    std::cout << report["instruction_set"].get<std::string>() << " specification reference; not a support decision\n"
              << report["counts"].dump(2) << '\n'; return;
  }
  std::cout << report["name"].get<std::string>() << ": " << report["candidate_count"]
            << " specification candidates; not a support decision\n";
  for (const auto& row : report["variants"]) {
    std::cout << "  " << row["id"].get<std::string>() << ": ";
    bool first = true;
    for (auto it = row["types"].begin(); it != row["types"].end(); ++it) {
      if (!first) std::cout << ", ";
      std::cout << it.key() << '=' << it.value()["element_type"].get<std::string>();
      first = false;
    }
    std::cout << " parameters=" << row["parameters"].dump()
              << "\n    requirements: " << row["requirements"].dump() << '\n';
  }
  std::cout << "Shapes, constants and limits (not evaluated):\n";
  for (const auto& arg : report["arguments"]) {
    Json constraints;
    for (const auto* key : {"logical_shape", "rank_range", "constant_required", "level_limits"})
      if (arg.contains(key)) constraints[key] = arg[key];
    std::cout << "  " << arg["name"].get<std::string>() << ": " << constraints.dump() << '\n';
  }

}
}  // namespace litert::arm_vulkan_ml::tool
