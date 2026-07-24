// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_SCHEMA_SOC_TABLE_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_SCHEMA_SOC_TABLE_H_

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace qnn {

struct SocInfo {
  std::string_view soc_name;
  uint32_t soc_model;
  constexpr SocInfo(std::string_view soc_name, uint32_t soc_model)
      : soc_name(soc_name), soc_model(soc_model) {}
};

// kSocInfos is derived from the QAIRT 2.47 documentation and sorted by soc_name
// in ascending order.
inline constexpr std::array<SocInfo, 111> kSocInfos{{
    SocInfo{"AIC100", 63},   SocInfo{"IPQ5404", 80},  SocInfo{"IPQ5424", 81},
    SocInfo{"IPQ6018", 23},  SocInfo{"IPQ6028", 24},  SocInfo{"IPQ9574", 79},
    SocInfo{"QCM2290", 83},  SocInfo{"QCM4290", 28},  SocInfo{"QCM4325", 56},
    SocInfo{"QCM4490", 59},  SocInfo{"QCM6125", 19},  SocInfo{"QCM6490", 93},
    SocInfo{"QCM6690", 78},  SocInfo{"QCS2290", 83},  SocInfo{"QCS403", 20},
    SocInfo{"QCS405", 18},   SocInfo{"QCS410", 33},   SocInfo{"QCS610", 16},
    SocInfo{"QCS6125", 48},  SocInfo{"QCS7230", 51},  SocInfo{"QCS8275", 82},
    SocInfo{"QCS8300", 82},  SocInfo{"QCS8550", 66},  SocInfo{"QCS8625", 90},
    SocInfo{"QCS9075", 77},  SocInfo{"QCS9100", 77},  SocInfo{"QRB4210", 49},
    SocInfo{"QRB5165", 21},  SocInfo{"SA525M", 84},   SocInfo{"SA7255", 67},
    SocInfo{"SA8195", 26},   SocInfo{"SA8255", 52},   SocInfo{"SA8295", 39},
    SocInfo{"SA8540", 62},   SocInfo{"SA8610", 67},   SocInfo{"SA8620", 67},
    SocInfo{"SA8620P", 67},  SocInfo{"SA8630", 52},   SocInfo{"SA8650", 52},
    SocInfo{"SA8775", 52},   SocInfo{"SA8797", 72},   SocInfo{"SAR1130P", 58},
    SocInfo{"SAR2130P", 46}, SocInfo{"SAR2230P", 95}, SocInfo{"SC7280X", 44},
    SocInfo{"SC8280X", 37},  SocInfo{"SC8380XP", 60}, SocInfo{"SC8480XP", 88},
    SocInfo{"SDM625", 11},   SocInfo{"SDM630", 10},   SocInfo{"SDM632", 15},
    SocInfo{"SDM636", 9},    SocInfo{"SDM652", 8},    SocInfo{"SDM660", 7},
    SocInfo{"SDM670", 6},    SocInfo{"SDM710", 13},   SocInfo{"SDM820", 4},
    SocInfo{"SDM821", 3},    SocInfo{"SDM835", 2},    SocInfo{"SDM845", 1},
    SocInfo{"SDM855", 12},   SocInfo{"SDM865", 21},   SocInfo{"SM4250", 28},
    SocInfo{"SM4350", 31},   SocInfo{"SM4375", 55},   SocInfo{"SM4450", 59},
    SocInfo{"SM4635", 71},   SocInfo{"SM6115", 28},   SocInfo{"SM6125", 19},
    SocInfo{"SM6150", 16},   SocInfo{"SM6225", 40},   SocInfo{"SM6250", 27},
    SocInfo{"SM6350", 29},   SocInfo{"SM6375", 31},   SocInfo{"SM6450", 50},
    SocInfo{"SM6450Q", 65},  SocInfo{"SM6475", 76},   SocInfo{"SM6650", 74},
    SocInfo{"SM7150", 17},   SocInfo{"SM7225", 29},   SocInfo{"SM7250", 25},
    SocInfo{"SM7315", 38},   SocInfo{"SM7325", 35},   SocInfo{"SM7350", 32},
    SocInfo{"SM7435", 61},   SocInfo{"SM7450", 41},   SocInfo{"SM7475", 54},
    SocInfo{"SM7550", 64},   SocInfo{"SM7635", 73},   SocInfo{"SM7675", 70},
    SocInfo{"SM7750", 86},   SocInfo{"SM8325", 34},   SocInfo{"SM8350", 30},
    SocInfo{"SM8350P", 30},  SocInfo{"SM8450", 36},   SocInfo{"SM8475", 42},
    SocInfo{"SM8550", 43},   SocInfo{"SM8635", 68},   SocInfo{"SM8650", 57},
    SocInfo{"SM8735", 85},   SocInfo{"SM8750", 69},   SocInfo{"SM8845", 97},
    SocInfo{"SM8845P", 97},  SocInfo{"SM8850", 87},   SocInfo{"SSG2115P", 46},
    SocInfo{"SSG2125P", 58}, SocInfo{"STP6225P", 47}, SocInfo{"SW6100", 96},
    SocInfo{"SXR1230P", 45}, SocInfo{"SXR2230P", 53}, SocInfo{"SXR2330P", 75},
}};

static_assert(
    []() {
      for (std::size_t i = 1; i < kSocInfos.size(); ++i) {
        if (kSocInfos[i].soc_name <= kSocInfos[i - 1].soc_name) return false;
      }
      return true;
    }(),
    "kSocInfos must be sorted ascending and unique by soc_name for binary "
    "search");

// Returns the SocInfo whose soc_name exactly matches `soc_name`, or
// std::nullopt if `soc_name` is null or not in the kSocInfos table.
constexpr std::optional<SocInfo> FindSocInfo(const char* soc_name) {
  if (soc_name == nullptr) return std::nullopt;
  std::size_t lo = 0, hi = kSocInfos.size();
  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    if (kSocInfos[mid].soc_name < soc_name) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo < kSocInfos.size() && kSocInfos[lo].soc_name == soc_name) {
    return kSocInfos[lo];
  }
  return std::nullopt;
}

// Resolves a SocInfo from either a SoC name (e.g. "SM8750") or a numeric
// SoC model (e.g. "43"). Returns nullopt if the input is null or matches
// neither format.
inline std::optional<SocInfo> FindOrCreateSocInfo(const char* soc_name_or_model) {
  if (soc_name_or_model == nullptr) return std::nullopt;

  // Try parsing the whole string as a SoC name.
  if (auto soc_info = FindSocInfo(soc_name_or_model)) return soc_info;

  // Try parsing the whole string as a SoC model.
  std::string_view sv(soc_name_or_model);
  uint32_t soc_model = 0;
  auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), soc_model);
  if (ec == std::errc{} && ptr == sv.data() + sv.size()) {
    return SocInfo{"CUSTOM_SOC", soc_model};
  }
  return std::nullopt;
}
}  // namespace qnn
#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_SCHEMA_SOC_TABLE_H_
