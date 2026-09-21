// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.
#include "litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "QnnTypes.h"  // from @qairt

namespace qnn {

bool UndefinedQuantizeParamsWrapper::operator==(
    const UndefinedQuantizeParamsWrapper& other) const {
  return qnn_quantize_param_.encodingDefinition ==
             other.qnn_quantize_param_.encodingDefinition &&
         qnn_quantize_param_.quantizationEncoding ==
             other.qnn_quantize_param_.quantizationEncoding;
}

void UndefinedQuantizeParamsWrapper::CloneTo(Qnn_QuantizeParams_t& dst) {
  dst = qnn_quantize_param_;
}

ScaleOffsetQuantizeParamsWrapper::ScaleOffsetQuantizeParamsWrapper(
    const float scale, const std::int32_t zero_point) {
  qnn_quantize_param_.encodingDefinition = QNN_DEFINITION_DEFINED;
  qnn_quantize_param_.quantizationEncoding =
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  qnn_quantize_param_.scaleOffsetEncoding.scale = scale;
  qnn_quantize_param_.scaleOffsetEncoding.offset = -1 * zero_point;
}

ScaleOffsetQuantizeParamsWrapper::ScaleOffsetQuantizeParamsWrapper(
    const Qnn_ScaleOffset_t& scale_offset) {
  qnn_quantize_param_.encodingDefinition = QNN_DEFINITION_DEFINED;
  qnn_quantize_param_.quantizationEncoding =
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  qnn_quantize_param_.scaleOffsetEncoding = scale_offset;
}

bool ScaleOffsetQuantizeParamsWrapper::operator==(
    const ScaleOffsetQuantizeParamsWrapper& other) const {
  return qnn_quantize_param_.encodingDefinition ==
             other.qnn_quantize_param_.encodingDefinition &&
         qnn_quantize_param_.quantizationEncoding ==
             other.qnn_quantize_param_.quantizationEncoding &&
         qnn_quantize_param_.scaleOffsetEncoding.scale ==
             other.qnn_quantize_param_.scaleOffsetEncoding.scale &&
         qnn_quantize_param_.scaleOffsetEncoding.offset ==
             other.qnn_quantize_param_.scaleOffsetEncoding.offset;
}

void ScaleOffsetQuantizeParamsWrapper::CloneTo(Qnn_QuantizeParams_t& dst) {
  dst = qnn_quantize_param_;
}

AxisScaleOffsetQuantizeParamsWrapper::AxisScaleOffsetQuantizeParamsWrapper(
    std::int32_t axis, absl::Span<const float> scales,
    absl::Span<const std::int32_t> zero_points)
    : scale_offsets_(scales.size()) {
  assert(scales.size() == zero_points.size());
  for (size_t i = 0; i < scale_offsets_.size(); ++i) {
    scale_offsets_[i].scale = scales[i];
    scale_offsets_[i].offset = -1 * zero_points[i];
  }

  qnn_quantize_param_.encodingDefinition = QNN_DEFINITION_DEFINED;
  qnn_quantize_param_.quantizationEncoding =
      QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
  qnn_quantize_param_.axisScaleOffsetEncoding.axis = axis;
  qnn_quantize_param_.axisScaleOffsetEncoding.numScaleOffsets =
      scale_offsets_.size();
  qnn_quantize_param_.axisScaleOffsetEncoding.scaleOffset =
      scale_offsets_.data();
}

AxisScaleOffsetQuantizeParamsWrapper::AxisScaleOffsetQuantizeParamsWrapper(
    std::int32_t axis, std::uint32_t num_scale_offsets, float scale,
    std::int32_t zero_point)
    : scale_offsets_(num_scale_offsets) {
  const auto offset = -zero_point;
  for (auto& scale_offset : scale_offsets_) {
    scale_offset.scale = scale;
    scale_offset.offset = offset;
  }

  qnn_quantize_param_.encodingDefinition = QNN_DEFINITION_DEFINED;
  qnn_quantize_param_.quantizationEncoding =
      QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
  qnn_quantize_param_.axisScaleOffsetEncoding.axis = axis;
  qnn_quantize_param_.axisScaleOffsetEncoding.numScaleOffsets =
      scale_offsets_.size();
  qnn_quantize_param_.axisScaleOffsetEncoding.scaleOffset =
      scale_offsets_.data();
}

AxisScaleOffsetQuantizeParamsWrapper::AxisScaleOffsetQuantizeParamsWrapper(
    const Qnn_AxisScaleOffset_t& axis_scale_offset) {
  scale_offsets_.resize(axis_scale_offset.numScaleOffsets);
  for (size_t i = 0; i < scale_offsets_.size(); ++i) {
    scale_offsets_[i].scale = axis_scale_offset.scaleOffset[i].scale;
    scale_offsets_[i].offset = axis_scale_offset.scaleOffset[i].offset;
  }
  qnn_quantize_param_.encodingDefinition = QNN_DEFINITION_DEFINED;
  qnn_quantize_param_.quantizationEncoding =
      QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
  qnn_quantize_param_.axisScaleOffsetEncoding.axis = axis_scale_offset.axis;
  qnn_quantize_param_.axisScaleOffsetEncoding.numScaleOffsets =
      scale_offsets_.size();
  qnn_quantize_param_.axisScaleOffsetEncoding.scaleOffset =
      scale_offsets_.data();
}

AxisScaleOffsetQuantizeParamsWrapper::AxisScaleOffsetQuantizeParamsWrapper(
    const AxisScaleOffsetQuantizeParamsWrapper& rhs)
    : scale_offsets_{rhs.scale_offsets_},
      qnn_quantize_param_{rhs.qnn_quantize_param_} {
  qnn_quantize_param_.axisScaleOffsetEncoding.scaleOffset =
      scale_offsets_.data();
}

AxisScaleOffsetQuantizeParamsWrapper&
AxisScaleOffsetQuantizeParamsWrapper::operator=(
    const AxisScaleOffsetQuantizeParamsWrapper& rhs) {
  if (this != &rhs) {
    scale_offsets_ = rhs.scale_offsets_;
    qnn_quantize_param_ = rhs.qnn_quantize_param_;
    qnn_quantize_param_.axisScaleOffsetEncoding.scaleOffset =
        scale_offsets_.data();
  }
  return *this;
}

AxisScaleOffsetQuantizeParamsWrapper::AxisScaleOffsetQuantizeParamsWrapper(
    AxisScaleOffsetQuantizeParamsWrapper&& rhs) noexcept
    : scale_offsets_{std::move(rhs.scale_offsets_)},
      qnn_quantize_param_{rhs.qnn_quantize_param_} {
  qnn_quantize_param_.axisScaleOffsetEncoding.scaleOffset =
      scale_offsets_.data();
  rhs.qnn_quantize_param_ = QNN_QUANTIZE_PARAMS_INIT;
}

bool AxisScaleOffsetQuantizeParamsWrapper::operator==(
    const AxisScaleOffsetQuantizeParamsWrapper& other) const {
  if (qnn_quantize_param_.encodingDefinition !=
      other.qnn_quantize_param_.encodingDefinition)
    return false;
  if (qnn_quantize_param_.quantizationEncoding !=
      other.qnn_quantize_param_.quantizationEncoding)
    return false;
  if (qnn_quantize_param_.axisScaleOffsetEncoding.axis !=
      other.qnn_quantize_param_.axisScaleOffsetEncoding.axis)
    return false;
  if (qnn_quantize_param_.axisScaleOffsetEncoding.numScaleOffsets !=
      other.qnn_quantize_param_.axisScaleOffsetEncoding.numScaleOffsets)
    return false;

  return std::equal(
      qnn_quantize_param_.axisScaleOffsetEncoding.scaleOffset,
      qnn_quantize_param_.axisScaleOffsetEncoding.scaleOffset +
          qnn_quantize_param_.axisScaleOffsetEncoding.numScaleOffsets,
      other.qnn_quantize_param_.axisScaleOffsetEncoding.scaleOffset,
      [](const Qnn_ScaleOffset_t& a, const Qnn_ScaleOffset_t& b) {
        return a.scale == b.scale && a.offset == b.offset;
      });
}

AxisScaleOffsetQuantizeParamsWrapper&
AxisScaleOffsetQuantizeParamsWrapper::operator=(
    AxisScaleOffsetQuantizeParamsWrapper&& rhs) noexcept {
  if (this != &rhs) {
    qnn_quantize_param_ = rhs.qnn_quantize_param_;
    scale_offsets_ = std::move(rhs.scale_offsets_);
    qnn_quantize_param_.axisScaleOffsetEncoding.scaleOffset =
        scale_offsets_.data();
    rhs.qnn_quantize_param_ = QNN_QUANTIZE_PARAMS_INIT;
  }
  return *this;
}

void AxisScaleOffsetQuantizeParamsWrapper::CloneTo(Qnn_QuantizeParams_t& dst) {
  dst = qnn_quantize_param_;
}

std::int32_t AxisScaleOffsetQuantizeParamsWrapper::GetAxis() const {
  return qnn_quantize_param_.axisScaleOffsetEncoding.axis;
}

void AxisScaleOffsetQuantizeParamsWrapper::SetAxis(const std::int32_t axis) {
  qnn_quantize_param_.axisScaleOffsetEncoding.axis = axis;
}

std::vector<float> AxisScaleOffsetQuantizeParamsWrapper::GetScales() const {
  std::vector<float> scales;
  scales.reserve(scale_offsets_.size());
  for (size_t i = 0; i < scale_offsets_.size(); ++i) {
    scales.emplace_back(scale_offsets_[i].scale);
  }
  return scales;
}

std::vector<std::int32_t> AxisScaleOffsetQuantizeParamsWrapper::GetZeroPoints()
    const {
  std::vector<std::int32_t> zero_points;
  zero_points.reserve(scale_offsets_.size());
  for (size_t i = 0; i < scale_offsets_.size(); ++i) {
    zero_points.emplace_back(-1 * scale_offsets_[i].offset);
  }
  return zero_points;
}

BwScaleOffsetQuantizeParamsWrapper::BwScaleOffsetQuantizeParamsWrapper(
    const std::uint32_t bitwidth, const float scale,
    const std::int32_t zero_point) {
  qnn_quantize_param_.encodingDefinition = QNN_DEFINITION_DEFINED;
  qnn_quantize_param_.quantizationEncoding =
      QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET;
  qnn_quantize_param_.bwScaleOffsetEncoding.bitwidth = bitwidth;
  qnn_quantize_param_.bwScaleOffsetEncoding.scale = scale;
  qnn_quantize_param_.bwScaleOffsetEncoding.offset = -1 * zero_point;
}

bool BwScaleOffsetQuantizeParamsWrapper::operator==(
    const BwScaleOffsetQuantizeParamsWrapper& other) const {
  return qnn_quantize_param_.encodingDefinition ==
             other.qnn_quantize_param_.encodingDefinition &&
         qnn_quantize_param_.quantizationEncoding ==
             other.qnn_quantize_param_.quantizationEncoding &&
         qnn_quantize_param_.bwScaleOffsetEncoding.bitwidth ==
             other.qnn_quantize_param_.bwScaleOffsetEncoding.bitwidth &&
         qnn_quantize_param_.bwScaleOffsetEncoding.scale ==
             other.qnn_quantize_param_.bwScaleOffsetEncoding.scale &&
         qnn_quantize_param_.bwScaleOffsetEncoding.offset ==
             other.qnn_quantize_param_.bwScaleOffsetEncoding.offset;
}

void BwScaleOffsetQuantizeParamsWrapper::CloneTo(Qnn_QuantizeParams_t& dst) {
  dst = qnn_quantize_param_;
}

BwAxisScaleOffsetQuantizeParamsWrapper::BwAxisScaleOffsetQuantizeParamsWrapper(
    const std::uint32_t bitwidth, const std::int32_t axis,
    const absl::Span<const float> scales,
    const absl::Span<const std::int32_t> zero_points)
    : scales_(scales.size()), offsets_(zero_points.size()) {
  assert(scales.size() == zero_points.size());
  for (size_t i = 0; i < scales.size(); ++i) {
    scales_[i] = scales[i];
    offsets_[i] = -1 * zero_points[i];
  }
  qnn_quantize_param_.encodingDefinition = QNN_DEFINITION_DEFINED;
  qnn_quantize_param_.quantizationEncoding =
      QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET;
  qnn_quantize_param_.bwAxisScaleOffsetEncoding.bitwidth = bitwidth;
  qnn_quantize_param_.bwAxisScaleOffsetEncoding.axis = axis;
  qnn_quantize_param_.bwAxisScaleOffsetEncoding.numElements = scales_.size();
  qnn_quantize_param_.bwAxisScaleOffsetEncoding.scales = scales_.data();
  qnn_quantize_param_.bwAxisScaleOffsetEncoding.offsets = offsets_.data();
}

BwAxisScaleOffsetQuantizeParamsWrapper::BwAxisScaleOffsetQuantizeParamsWrapper(
    const BwAxisScaleOffsetQuantizeParamsWrapper& rhs)
    : scales_{rhs.scales_},
      offsets_{rhs.offsets_},
      qnn_quantize_param_{rhs.qnn_quantize_param_} {
  qnn_quantize_param_.bwAxisScaleOffsetEncoding.scales = scales_.data();
  qnn_quantize_param_.bwAxisScaleOffsetEncoding.offsets = offsets_.data();
}

BwAxisScaleOffsetQuantizeParamsWrapper&
BwAxisScaleOffsetQuantizeParamsWrapper::operator=(
    const BwAxisScaleOffsetQuantizeParamsWrapper& rhs) {
  if (this != &rhs) {
    scales_ = rhs.scales_;
    offsets_ = rhs.offsets_;
    qnn_quantize_param_ = rhs.qnn_quantize_param_;
    qnn_quantize_param_.bwAxisScaleOffsetEncoding.scales = scales_.data();
    qnn_quantize_param_.bwAxisScaleOffsetEncoding.offsets = offsets_.data();
  }
  return *this;
}

BwAxisScaleOffsetQuantizeParamsWrapper::BwAxisScaleOffsetQuantizeParamsWrapper(
    BwAxisScaleOffsetQuantizeParamsWrapper&& rhs) noexcept
    : scales_{std::move(rhs.scales_)},
      offsets_{std::move(rhs.offsets_)},
      qnn_quantize_param_{rhs.qnn_quantize_param_} {
  qnn_quantize_param_.bwAxisScaleOffsetEncoding.scales = scales_.data();
  qnn_quantize_param_.bwAxisScaleOffsetEncoding.offsets = offsets_.data();
  rhs.qnn_quantize_param_ = QNN_QUANTIZE_PARAMS_INIT;
}

BwAxisScaleOffsetQuantizeParamsWrapper&
BwAxisScaleOffsetQuantizeParamsWrapper::operator=(
    BwAxisScaleOffsetQuantizeParamsWrapper&& rhs) noexcept {
  if (this != &rhs) {
    qnn_quantize_param_ = rhs.qnn_quantize_param_;
    scales_ = std::move(rhs.scales_);
    offsets_ = std::move(rhs.offsets_);
    qnn_quantize_param_.bwAxisScaleOffsetEncoding.scales = scales_.data();
    qnn_quantize_param_.bwAxisScaleOffsetEncoding.offsets = offsets_.data();
    rhs.qnn_quantize_param_ = QNN_QUANTIZE_PARAMS_INIT;
  }
  return *this;
}

bool BwAxisScaleOffsetQuantizeParamsWrapper::operator==(
    const BwAxisScaleOffsetQuantizeParamsWrapper& other) const {
  if (qnn_quantize_param_.encodingDefinition !=
      other.qnn_quantize_param_.encodingDefinition)
    return false;
  if (qnn_quantize_param_.quantizationEncoding !=
      other.qnn_quantize_param_.quantizationEncoding)
    return false;
  if (qnn_quantize_param_.bwAxisScaleOffsetEncoding.bitwidth !=
      other.qnn_quantize_param_.bwAxisScaleOffsetEncoding.bitwidth)
    return false;
  if (qnn_quantize_param_.bwAxisScaleOffsetEncoding.axis !=
      other.qnn_quantize_param_.bwAxisScaleOffsetEncoding.axis)
    return false;
  if (qnn_quantize_param_.bwAxisScaleOffsetEncoding.numElements !=
      other.qnn_quantize_param_.bwAxisScaleOffsetEncoding.numElements)
    return false;

  return std::equal(
             qnn_quantize_param_.bwAxisScaleOffsetEncoding.scales,
             qnn_quantize_param_.bwAxisScaleOffsetEncoding.scales +
                 qnn_quantize_param_.bwAxisScaleOffsetEncoding.numElements,
             other.qnn_quantize_param_.bwAxisScaleOffsetEncoding.scales) &&
         std::equal(
             qnn_quantize_param_.bwAxisScaleOffsetEncoding.offsets,
             qnn_quantize_param_.bwAxisScaleOffsetEncoding.offsets +
                 qnn_quantize_param_.bwAxisScaleOffsetEncoding.numElements,
             other.qnn_quantize_param_.bwAxisScaleOffsetEncoding.offsets);
}

void BwAxisScaleOffsetQuantizeParamsWrapper::CloneTo(
    Qnn_QuantizeParams_t& dst) {
  dst = qnn_quantize_param_;
}

void BwAxisScaleOffsetQuantizeParamsWrapper::SetAxis(const std::int32_t axis) {
  qnn_quantize_param_.bwAxisScaleOffsetEncoding.axis = axis;
}

BwFloatBlockQuantizeParamsWrapper::BwFloatBlockQuantizeParamsWrapper(
    std::uint32_t bitwidth, absl::Span<const std::uint32_t> block_sizes,
    absl::Span<const float> scales, absl::Span<const std::int32_t> zero_points)
    : bitwidth_(bitwidth),
      block_sizes_(block_sizes.begin(), block_sizes.end()),
      scale_offsets_(scales.size()) {
  assert(zero_points.size() == scales.size());
  for (size_t i = 0; i < scales.size(); ++i) {
    scale_offsets_[i].scale = scales[i];
    scale_offsets_[i].offset = -zero_points[i] * scales[i];
  }
}

BwFloatBlockQuantizeParamsWrapper::BwFloatBlockQuantizeParamsWrapper(
    const Qnn_BwFloatBlockEncoding_t& encoding,
    absl::Span<const std::uint32_t> dimensions)
    : bitwidth_(encoding.bitwidth) {
  if (dimensions.empty() || !encoding.blockSize || !encoding.floatScaleOffset) {
    return;
  }
  block_sizes_.assign(encoding.blockSize,
                      encoding.blockSize + dimensions.size());
  size_t count = 1;
  for (size_t d = 0; d < dimensions.size(); ++d) {
    if (dimensions[d] == 0 || block_sizes_[d] == 0) {
      block_sizes_.clear();
      return;
    }
    const size_t blocks = 1 + (dimensions[d] - 1) / block_sizes_[d];
    if (blocks > scale_offsets_.max_size() / count) {
      block_sizes_.clear();
      return;
    }
    count *= blocks;
  }
  scale_offsets_.assign(encoding.floatScaleOffset,
                        encoding.floatScaleOffset + count);
}

void BwFloatBlockQuantizeParamsWrapper::CloneTo(Qnn_QuantizeParams_t& dst) {
  dst = QNN_QUANTIZE_PARAMS_INIT;
  dst.encodingDefinition = QNN_DEFINITION_DEFINED;
  dst.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK;
  dst.bwFloatBlockEncoding.bitwidth = bitwidth_;
  dst.bwFloatBlockEncoding.blockSize = block_sizes_.data();
  dst.bwFloatBlockEncoding.floatScaleOffset = scale_offsets_.data();
}

bool BwFloatBlockQuantizeParamsWrapper::operator==(
    const BwFloatBlockQuantizeParamsWrapper& other) const {
  return bitwidth_ == other.bitwidth_ && block_sizes_ == other.block_sizes_ &&
         scale_offsets_.size() == other.scale_offsets_.size() &&
         std::equal(scale_offsets_.begin(), scale_offsets_.end(),
                    other.scale_offsets_.begin(),
                    [](const auto& a, const auto& b) {
                      return a.scale == b.scale && a.offset == b.offset;
                    });
}

}  // namespace qnn
