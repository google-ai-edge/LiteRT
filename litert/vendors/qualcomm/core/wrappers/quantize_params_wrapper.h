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
  UndefinedQuantizeParamsWrapper();

  UndefinedQuantizeParamsWrapper(const UndefinedQuantizeParamsWrapper&);

  UndefinedQuantizeParamsWrapper(UndefinedQuantizeParamsWrapper&&);

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

  ScaleOffsetQuantizeParamsWrapper(const ScaleOffsetQuantizeParamsWrapper&);

  ScaleOffsetQuantizeParamsWrapper(ScaleOffsetQuantizeParamsWrapper&&);

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

  AxisScaleOffsetQuantizeParamsWrapper(
      AxisScaleOffsetQuantizeParamsWrapper&& rhs);

  void CloneTo(Qnn_QuantizeParams_t& dst);

  std::int32_t GetAxis() const;

  void SetAxis(const std::int32_t axis);

  void GetScales(std::vector<float>& scales) const;

  void GetZeroPoints(std::vector<std::int32_t>& zero_points) const;

 private:
  Qnn_QuantizeParams_t qnn_quantize_param_ = QNN_QUANTIZE_PARAMS_INIT;
  std::vector<Qnn_ScaleOffset_t> scale_offsets_;
};

class BwScaleOffsetQuantizeParamsWrapper final {
 public:
  explicit BwScaleOffsetQuantizeParamsWrapper(const std::uint32_t bitwidth,
                                              const float scale,
                                              const std::int32_t zero_point);

  BwScaleOffsetQuantizeParamsWrapper(
      const BwScaleOffsetQuantizeParamsWrapper& rhs);

  BwScaleOffsetQuantizeParamsWrapper(BwScaleOffsetQuantizeParamsWrapper&& rhs);

  void CloneTo(Qnn_QuantizeParams_t& dst);

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

  BwAxisScaleOffsetQuantizeParamsWrapper(
      BwAxisScaleOffsetQuantizeParamsWrapper&& rhs);

  void CloneTo(Qnn_QuantizeParams_t& dst);

  void SetAxis(const std::int32_t axis);

 private:
  Qnn_QuantizeParams_t qnn_quantize_param_ = QNN_QUANTIZE_PARAMS_INIT;
  std::vector<float> scales_;
  std::vector<int32_t> offsets_;
};

using QuantizeParamsWrapperVariant = std::variant<
    UndefinedQuantizeParamsWrapper, ScaleOffsetQuantizeParamsWrapper,
    AxisScaleOffsetQuantizeParamsWrapper,
    BwAxisScaleOffsetQuantizeParamsWrapper, BwScaleOffsetQuantizeParamsWrapper>;

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_QUANTIZE_PARAMS_WRAPPER_H_
