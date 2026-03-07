// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/schema/soc_table.h"

#include <cstdint>
namespace qnn {
constexpr SocInfo kSocInfos[] = {
    {SocInfo("UNKNOWN_SDM", SnapdragonModel::UNKNOWN_SDM, DspArch::NONE,
             0  // vtcm_size_in_mb
             )},
    {SocInfo("SDM865", SnapdragonModel::SDM865, DspArch::V66,
             0  // vtcm_size_in_mb
             )},
    {SocInfo("SM6350", SnapdragonModel::SM6350, DspArch::V66,
             0  // vtcm_size_in_mb
             )},
    {SocInfo("SA8255", SnapdragonModel::SA8255, DspArch::V73,
             8  // vtcm_size_in_mb
             )},
    {SocInfo("SA8295", SnapdragonModel::SA8295, DspArch::V68,
             8  // vtcm_size_in_mb
             )},
    {SocInfo("SM8350", SnapdragonModel::SM8350, DspArch::V68,
             4  // vtcm_size_in_mb
             )},
    {SocInfo("SM8450", SnapdragonModel::SM8450, DspArch::V69,
             8  // vtcm_size_in_mb
             )},
    {SocInfo("SM8475", SnapdragonModel::SM8475, DspArch::V69,
             8  // vtcm_size_in_mb
             )},
    {SocInfo("SC8380XP", SnapdragonModel::SC8380XP, DspArch::V73,
             8  // vtcm_size_in_mb
             )},
    {SocInfo("SM8550", SnapdragonModel::SM8550, DspArch::V73,
             8  // vtcm_size_in_mb
             )},
    {SocInfo("SM8650", SnapdragonModel::SM8650, DspArch::V75,
             8  // vtcm_size_in_mb
             )},
    {SocInfo("SM8750", SnapdragonModel::SM8750, DspArch::V79,
             8  // vtcm_size_in_mb
             )},
    {SocInfo("SM8850", SnapdragonModel::SM8850, DspArch::V81,
             8  // vtcm_size_in_mb
             )},
    {SocInfo("SAR2230P", SnapdragonModel::SAR2230P, DspArch::V81,
             4  // vtcm_size_in_mb
             )},
    {SocInfo("SXR2230P", SnapdragonModel::SXR2230P, DspArch::V69,
             8  // vtcm_size_in_mb
             )},
    {SocInfo("SSG2125P", SnapdragonModel::SSG2125P, DspArch::V73,
             2  // vtcm_size_in_mb
             )},
};
constexpr uint64_t kNumSocInfos = sizeof(kSocInfos) / sizeof(kSocInfos[0]);
}  // namespace qnn
