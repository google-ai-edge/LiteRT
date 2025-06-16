//==============================================================================
// Auto Generated Code for LiteRtQualcommOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

BEGIN_PKG_OP_DEFINITION(PKG_ElementWiseAdd);

// op execute function declarations
template <typename TensorType>
GraphStatus elementwiseaddImpl(TensorType& out_0, const TensorType& in_0,
                               const TensorType& in_1);

// forward declaration of sample cost function
static float elementwiseaddCostFunc(const Op* op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default
 * flag (Flags::RESOURCE_HVX) syntax: DEF_PACKAGE_OP(F,OP) e.g.
 * DEF_PACKAGE_OP((elementwiseaddImpl<Tensor>), "ElementWiseAdd")
 */
DEF_PACKAGE_OP((elementwiseaddImpl<Tensor>), "ElementWiseAdd")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL,
 * FAST, FREE) and provided flags syntax:
 * DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...) can use zero or more flags,
 * FLAG options are IS_CONST, INHIBIT_CONST_PROP, RESOURCE_HVX, RESOURCE_HMX(not
 * supported in external op packages) e.g.
 * DEF_PACKAGE_OP_AND_COST_AND_FLAGS((elementwiseaddImpl<PlainFloatTensor>),
 * "ElementWiseAdd", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g.
 * DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((elementwiseaddImpl<PlainFloatTensor>),
 * "ElementWiseAdd", elementwiseaddCostFunc, Flags::RESOURCE_HVX)
 */

/*
 * optimization definitions
 * need to be global in the package
 * one definition per optimization
 * syntax:
 * DEF_PACKAGE_OPTIMIZATION(PRIORITY,MATCHCODE,CONSTRAINTCODE,REPLACECODE)
 * PRIORITY predefined values include EARLY(2000), MIDDLE(3000), LATE(4000)
 * HTP core provides some replacement functions for op package to use
 * for more information about optimization rules, please refer to HTP core
 * documentations
 */

/*
 * op parameter order definitions
 * need to be global in the package
 * one definition per op, and this is optional
 * syntax:
 * DEF_PACKAGE_PARAM_ORDER(OP,PARAM1,MANDATORY1,DEFAULT1,PARAM2,MANDATORY2,DEFAULT2...)
 * one or more parameters can be specified for each op
 * order of parameters listed determines the order of parameters passed into op
 * execution functions if an op does not have a parameter order definition,
 * parameter order passed into Qnn_addNode will be passed into op execution
 * functions if an op has a parameter order definition, any parameter passed
 * into Qnn_addNode with unlisted name will be abandoned if two or more op
 * packages with the same package name will be registered, they cannot list
 *   conflicting parameter orders
 * PARAM refers to parameter name as a string literal
 * MANDATORY refers to whether this parameter is required to be provided at
 * Qnn_addNode DEFAULT is used when MANDATORY is false if provided as
 * Qnn_Param_t*, DEFAULT will be used for graph construction when this parameter
 * is not provided at Qnn_addNode if provided as nullptr, graph construction
 * will skip this parameter when this parameter is not provided at Qnn_addNode
 */

/* execute functions for ops */

namespace {
template <typename TensorType, typename DataType>
void elementwiseAdd(const size_t num_elements, TensorType& output_0,
                    const TensorType& input_0, const TensorType& input_1) {
  const auto* input_data_0 =
      static_cast<const DataType*>(input_0.raw_data_const());
  const auto* input_data_1 =
      static_cast<const DataType*>(input_1.raw_data_const());
  auto* output_data_0 = static_cast<DataType*>(output_0.raw_data());
  for (size_t i = 0; i < num_elements; ++i) {
    output_data_0[i] = input_data_0[i] + input_data_1[i];
  }
}

}  // namespace

template <typename TensorType>
GraphStatus elementwiseaddImpl(TensorType& output_0, const TensorType& input_0,
                               const TensorType& input_1) {
  /*
   * add code here
   * */
  /*
   * To have good performance and stability, it is required to avoid heap memory
   * allocation in this function. The heap memory allocation includes but not
   * limited to calling malloc, operator new, constructing STL container objects
   * like std::vector with default allocator, and adding items like calling
   * std::vector::push_back to STL container objects with default allocator.
   *
   * Please check in SDK documentation for more information.
   */

  const size_t input_num_elements_0 = input_0.total_storage_elements();
  const size_t input_num_elements_1 = input_1.total_storage_elements();
  const size_t output_num_elements_0 = output_0.total_storage_elements();
  if (input_num_elements_0 != input_num_elements_1 ||
      input_num_elements_0 != output_num_elements_0) {
    return GraphStatus::ErrorBadInput;
  }

  const DTypeScaleOff input_dtype_0 = input_0.get_dtype_intfc();
  const DTypeScaleOff input_dtype_1 = input_1.get_dtype_intfc();
  const DTypeScaleOff output_dtype_0 = output_0.get_dtype_intfc();
  if (input_dtype_0.dtype != input_dtype_1.dtype ||
      input_dtype_0.dtype != output_dtype_0.dtype) {
    return GraphStatus::ErrorBadInput;
  }

  // Handle float types.
  if (input_dtype_0.dtype == DType::Float32) {
    elementwiseAdd<TensorType, float>(input_num_elements_0, output_0, input_0,
                                      input_1);
    return GraphStatus::Success;
  }

  // Currently we only support quantized types with the same offset and scale.
  if (input_dtype_0.scale != input_dtype_1.scale ||
      input_dtype_0.scale != output_dtype_0.scale) {
    return GraphStatus::ErrorBadInput;
  }
  if (input_dtype_0.offset != input_dtype_1.offset ||
      input_dtype_0.offset != output_dtype_0.offset) {
    return GraphStatus::ErrorBadInput;
  }

  // Handle quantized types.
  if (input_dtype_0.dtype == DType::QInt8) {
    elementwiseAdd<TensorType, std::int8_t>(input_num_elements_0, output_0,
                                            input_0, input_1);
  } else if (input_dtype_0.dtype == DType::QUInt8) {
    elementwiseAdd<TensorType, std::uint8_t>(input_num_elements_0, output_0,
                                             input_0, input_1);
  } else if (input_dtype_0.dtype == DType::QInt16) {
    elementwiseAdd<TensorType, std::int16_t>(input_num_elements_0, output_0,
                                             input_0, input_1);
  } else if (input_dtype_0.dtype == DType::QUInt16) {
    elementwiseAdd<TensorType, std::uint16_t>(input_num_elements_0, output_0,
                                              input_0, input_1);
  } else {
    return GraphStatus::ErrorBadInput;
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float elementwiseaddCostFunc(const Op* op) {
  /*
   * add code here
   * */

  float cost = 0.0;  // add cost computation here
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_ElementWiseAdd);