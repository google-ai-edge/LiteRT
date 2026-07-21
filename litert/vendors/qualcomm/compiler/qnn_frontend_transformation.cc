#include "litert/vendors/qualcomm/compiler/qnn_frontend_transformation.h"

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/cc/internal/litert_builder.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_matchers.h"
#include "litert/cc/internal/litert_op_options.h"

using litert::Builder;
using litert::Op;

LiteRtStatus QnnTransformation(LiteRtBuilder builder_ptr, LiteRtOp op) {
  Builder builder(builder_ptr);
  Op root_op(op);
  Op mean_op(nullptr);
  Op mul_op(nullptr);
  litert::Tensor sq_in(nullptr);

  // Match: Sqrt(Mean(Mul(x, x)))
  auto mul_op_matcher = litert::m_Op<kLiteRtOpCodeTflMul>(
      litert::m_CaptureOrSameAs(&sq_in, litert::m_Any()),
      litert::m_CaptureOrSameAs(&sq_in, litert::m_Any()));

  auto mean_input_matcher = litert::m_CaptureOrSameAs(
      &mul_op, litert::m_AllOf(litert::m_HasOneUse(), mul_op_matcher));

  auto mean_op_matcher = litert::m_Op<kLiteRtOpCodeTflMean>(
      mean_input_matcher, litert::m_Any());

  auto sqrt_input_matcher = litert::m_CaptureOrSameAs(
      &mean_op, litert::m_AllOf(litert::m_HasOneUse(), mean_op_matcher));

  if (!litert::Match(root_op,
                     litert::m_Op<kLiteRtOpCodeTflSqrt>(sqrt_input_matcher))) {
    return kLiteRtStatusPatternNoMatch;
  }

  LITERT_LOG(LITERT_INFO, "Pattern match!");

  // Read keep_dims from the original Mean op to propagate to Sum.
  litert::MeanOptions mean_options;
  LITERT_RETURN_IF_ERROR(mean_options.InitFromOp(mean_op.Get()));

  // mean_op inputs: [0] = Mul output, [1] = axes constant tensor.
  litert::Tensor mul_out = mean_op.Inputs()[0];
  litert::Tensor axes_tensor = mean_op.Inputs()[1];

  // Read axes values to compute the product of reduced dimension sizes.
  auto axes_data = axes_tensor.WeightsData<int32_t>();
  if (!axes_data) {
    return kLiteRtStatusPatternNoMatch;
  }

  auto ranked_type = mul_out.RankedTensorType();
  if (!ranked_type) {
    return kLiteRtStatusPatternNoMatch;
  }
  const auto dims = ranked_type->Layout().Dimensions();
  const int rank = static_cast<int>(dims.size());

  float dim_size = 1.0f;
  for (int32_t axis : *axes_data) {
    int32_t normalized = (axis < 0) ? axis + rank : axis;
    if (normalized < 0 || normalized >= rank || dims[normalized] <= 0) {
      return kLiteRtStatusPatternNoMatch;
    }
    dim_size *= static_cast<float>(dims[normalized]);
  }

  // Build the Sum op output tensor from scratch with the same type as Mean's
  // output. Do NOT use CloneTensor — it copies DefiningOp/user state from the
  // source tensor which becomes a stale pointer after ApplyChanges transfers
  // ops out of the builder's staging subgraph.
  auto mean_out_ranked_type = mean_op.Outputs()[0].RankedTensorType();
  if (!mean_out_ranked_type) {
    return kLiteRtStatusPatternNoMatch;
  }
  auto sum_out_tensor = builder.BuildTensor(
      litert::RankedTensorSpecBuilder(*mean_out_ranked_type).Build());
  if (!sum_out_tensor) {
    return sum_out_tensor.Error().Status();
  }

  // Build the Sum op directly so we can set its options.
  Op sum_op = builder.BuildOp(kLiteRtOpCodeTflSum,
                               {mul_out, axes_tensor}, {*sum_out_tensor});

  // Set Sum keep_dims option (inherited from Mean).
  litert::SumOptions sum_options;
  sum_options.keep_dims = mean_options.keep_dims;
  auto opts_result = builder.SetOpOptions(sum_op, std::move(sum_options));
  if (!opts_result) {
    return opts_result.Error().Status();
  }

  // Build a [1] float constant tensor holding dim_size for the Div denominator.
  auto dim_tensor = builder.BuildTensor(litert::TensorType<float>({1}));
  if (!dim_tensor) {
    return dim_tensor.Error().Status();
  }
  auto weights = builder.BuildWeights<float>(
      absl::MakeConstSpan(&dim_size, 1), *dim_tensor);
  if (!weights) {
    return weights.Error().Status();
  }

  // Replace Mean with Div(sum_out, dim_tensor), reusing Mean's output tensor.
  Op div_op = builder.ReplaceOp(mean_op, kLiteRtOpCodeTflDiv,
                                {*sum_out_tensor, *dim_tensor});

  // Set Div options (no fused activation).
  litert::DivOptions div_options;
  div_options.fused_activation_function = litert::kActivationFunctionTypeNone;
  auto div_opts_result = builder.SetOpOptions(div_op, std::move(div_options));
  if (!div_opts_result) {
    return div_opts_result.Error().Status();
  }

  return kLiteRtStatusOk;
}
