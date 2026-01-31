// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_QUANTIZE_PARAMS_WRAPPER_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_QUANTIZE_PARAMS_WRAPPER_H_

#include <cstdint>
#include <variant>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "QnnTypes.h"  // from @qairt

namespace qnn {

class UndefinedQuantizeParamsWrapper final {
 public:
  bool operator==(const UndefinedQuantizeParamsWrapper& other) const;

  void CloneTo(Qnn_QuantizeParams_t& dst);

 private:
  Qnn_QuantizeParams_t qnn_quantize_param_ = QNN_QUANTIZE_PARAMS_INIT;
};

class ScaleOffsetQuantizeParamsWrapper final {
 public:
  explicit ScaleOffsetQuantizeParamsWrapper(const float scale,
                                            const std::int32_t zero_point);
  explicit ScaleOffsetQuantizeParamsWrapper(
      const Qnn_ScaleOffset_t& scale_offset);

  bool operator==(const ScaleOffsetQuantizeParamsWrapper& other) const;

  void CloneTo(Qnn_QuantizeParams_t& dst);

  float GetScale() const {
    return qnn_quantize_param_.scaleOffsetEncoding.scale;
  }

  std::int32_t GetZeroPoint() const {
    return -1 * qnn_quantize_param_.scaleOffsetEncoding.offset;
  }

  std::int32_t GetOffset() const {
    return qnn_quantize_param_.scaleOffsetEncoding.offset;
  }

 private:
  Qnn_QuantizeParams_t qnn_quantize_param_ = QNN_QUANTIZE_PARAMS_INIT;
};

class AxisScaleOffsetQuantizeParamsWrapper final {
 public:
  explicit AxisScaleOffsetQuantizeParamsWrapper(
      const std::int32_t axis, const absl::Span<const float> scales,
      const absl::Span<const std::int32_t> zero_points);

  explicit AxisScaleOffsetQuantizeParamsWrapper(
      const Qnn_AxisScaleOffset_t& axis_scale_offset);

  AxisScaleOffsetQuantizeParamsWrapper(
      const AxisScaleOffsetQuantizeParamsWrapper& rhs);

  AxisScaleOffsetQuantizeParamsWrapper& operator=(
      const AxisScaleOffsetQuantizeParamsWrapper& rhs);

  AxisScaleOffsetQuantizeParamsWrapper(
      AxisScaleOffsetQuantizeParamsWrapper&& rhs) noexcept;

  AxisScaleOffsetQuantizeParamsWrapper& operator=(
      AxisScaleOffsetQuantizeParamsWrapper&& rhs) noexcept;

  ~AxisScaleOffsetQuantizeParamsWrapper() = default;

  bool operator==(const AxisScaleOffsetQuantizeParamsWrapper& other) const;

  void CloneTo(Qnn_QuantizeParams_t& dst);

  std::int32_t GetAxis() const;

  void SetAxis(const std::int32_t axis);

  void GetScales(std::vector<float>& scales) const;

  void GetZeroPoints(std::vector<std::int32_t>& zero_points) const;

 private:
  std::vector<Qnn_ScaleOffset_t> scale_offsets_;
  Qnn_QuantizeParams_t qnn_quantize_param_ = QNN_QUANTIZE_PARAMS_INIT;
};

class BwScaleOffsetQuantizeParamsWrapper final {
 public:
  explicit BwScaleOffsetQuantizeParamsWrapper(const std::uint32_t bitwidth,
                                              const float scale,
                                              const std::int32_t zero_point);

  bool operator==(const BwScaleOffsetQuantizeParamsWrapper& other) const;

  void CloneTo(Qnn_QuantizeParams_t& dst);

  std::uint32_t GetBitwidth() const {
    return qnn_quantize_param_.bwScaleOffsetEncoding.bitwidth;
  }

  float GetScale() const {
    return qnn_quantize_param_.bwScaleOffsetEncoding.scale;
  }

  std::int32_t GetZeroPoint() const {
    return -qnn_quantize_param_.bwScaleOffsetEncoding.offset;
  }

 private:
  Qnn_QuantizeParams_t qnn_quantize_param_ = QNN_QUANTIZE_PARAMS_INIT;
};

class BwAxisScaleOffsetQuantizeParamsWrapper final {
 public:
  explicit BwAxisScaleOffsetQuantizeParamsWrapper(
      const std::uint32_t bitwidth, const std::int32_t axis,
      const absl::Span<const float> scales,
      const absl::Span<const std::int32_t> zero_points);

  BwAxisScaleOffsetQuantizeParamsWrapper(
      const BwAxisScaleOffsetQuantizeParamsWrapper& rhs);

  BwAxisScaleOffsetQuantizeParamsWrapper& operator=(
      const BwAxisScaleOffsetQuantizeParamsWrapper& rhs);

  BwAxisScaleOffsetQuantizeParamsWrapper(
      BwAxisScaleOffsetQuantizeParamsWrapper&& rhs) noexcept;

  BwAxisScaleOffsetQuantizeParamsWrapper& operator=(
      BwAxisScaleOffsetQuantizeParamsWrapper&& rhs) noexcept;

  ~BwAxisScaleOffsetQuantizeParamsWrapper() = default;

  bool operator==(const BwAxisScaleOffsetQuantizeParamsWrapper& other) const;

  void CloneTo(Qnn_QuantizeParams_t& dst);

  void SetAxis(const std::int32_t axis);

  std::uint32_t GetBitwidth() const {
    return qnn_quantize_param_.bwAxisScaleOffsetEncoding.bitwidth;
  }

 private:
  std::vector<float> scales_;
  std::vector<int32_t> offsets_;
  Qnn_QuantizeParams_t qnn_quantize_param_ = QNN_QUANTIZE_PARAMS_INIT;
};

using QuantizeParamsWrapperVariant = std::variant<
    UndefinedQuantizeParamsWrapper, ScaleOffsetQuantizeParamsWrapper,
    AxisScaleOffsetQuantizeParamsWrapper,
    BwAxisScaleOffsetQuantizeParamsWrapper, BwScaleOffsetQuantizeParamsWrapper>;

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_QUANTIZE_PARAMS_WRAPPER_H_
