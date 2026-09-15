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

// Local memory diagnostics only. No heap trimming or page residency changes.
#ifndef LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MEMORY_SNAPSHOT_H_
#define LITERT_TENSOR_EXAMPLES_GEMMA4_NATIVE_MEMORY_SNAPSHOT_H_
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <sys/resource.h>

namespace gemma_memory {
using Counters = std::map<std::string, uint64_t>;
inline std::string Quote(const std::string& s) {
  std::ostringstream out; out << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') out << '\\' << char(c);
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << char(c);
  }
  return out.str() + '"';
}
inline std::string Object(const Counters& values) {
  std::ostringstream out; out << '{'; bool first = true;
  for (const auto& [key, value] : values) {
    if (!first) out << ','; first = false;
    out << Quote(key) << ':' << value;
  }
  return out.str() + '}';
}
inline bool Metric(const std::string& line, std::string& key, uint64_t& bytes) {
  std::istringstream in(line); uint64_t value; std::string unit;
  if (!(in >> key >> value >> unit) || unit != "kB" || key.empty() || key.back() != ':') return false;
  key.pop_back(); bytes = value * 1024; return true;
}
// counters_json is a caller-authored JSON object of allocation counters. All
// smaps values are bytes; inaccessible proc files are explicitly identified.
inline bool AppendMemorySnapshot(const std::string& path, const std::string& phase,
    const std::string& case_id = "", int repetition = -1,
    const std::string& counters_json = "{}") {
  if (path.empty()) return true;
  Counters rollup; std::string line, key; uint64_t bytes;
  std::ifstream smaps_rollup("/proc/self/smaps_rollup");
  const bool rollup_available = smaps_rollup.is_open();
  while (std::getline(smaps_rollup, line)) if (Metric(line, key, bytes)) rollup[key] = bytes;
  std::ifstream smaps("/proc/self/smaps");
  const bool smaps_available = smaps.is_open();
  std::map<std::string, Counters> groups;
  std::string group = "anonymous_or_special";
  while (std::getline(smaps, line)) {
    std::istringstream header(line); std::string address, permissions, offset, device, inode;
    if ((header >> address >> permissions >> offset >> device >> inode) &&
        address.find('-') != std::string::npos && permissions.size() == 4) {
      std::string mapping; std::getline(header, mapping);
      const auto first = mapping.find_first_not_of(' ');
      mapping = first == std::string::npos ? "" : mapping.substr(first);
      if (mapping.find("/matched-bundle/") != std::string::npos) group = "native_bundle_files";
      else if (mapping.find(".litertlm") != std::string::npos) group = "litertlm_bundle_files";
      else if (!mapping.empty() && mapping.front() == '/') group = "other_file_mappings";
      else group = "anonymous_or_special";
    } else if (Metric(line, key, bytes)) groups[group][key] += bytes;
  }
  struct rusage usage{}; const bool peak_available = getrusage(RUSAGE_SELF, &usage) == 0;
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  std::ofstream out(path, std::ios::app);
  out << "{\"schema_version\":1,\"units\":\"bytes\",\"phase\":" << Quote(phase)
      << ",\"case_id\":" << Quote(case_id) << ",\"repetition\":" << repetition
      << ",\"host_epoch_ms\":" << std::chrono::duration_cast<std::chrono::milliseconds>(now).count()
      << ",\"smaps_rollup_available\":" << (rollup_available ? "true" : "false")
      << ",\"smaps_available\":" << (smaps_available ? "true" : "false")
      << ",\"peak_rss_available\":" << (peak_available ? "true" : "false")
      << ",\"peak_rss_bytes\":" << (peak_available ? uint64_t(usage.ru_maxrss) * 1024 : 0)
      << ",\"smaps_rollup\":" << Object(rollup) << ",\"mapping_groups\":{";
  bool first = true;
  for (const auto& [name, values] : groups) {
    if (!first) out << ','; first = false; out << Quote(name) << ':' << Object(values);
  }
  out << "},\"allocation_counters\":" << counters_json
      << ",\"timings_valid_for_benchmark\":false}\n";
  return bool(out);
}
}  // namespace gemma_memory
#endif
