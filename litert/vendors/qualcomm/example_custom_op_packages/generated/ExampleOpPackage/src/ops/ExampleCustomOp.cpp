//==============================================================================
// Auto Generated Code for ExampleCustomOp - QHPI Implementation
// Multiple kernels generated for different data type combinations
//==============================================================================

#include "HTP/core/constraints.h"
#include <string>

// Plugin/QHPI includes - using correct header from hexnn_qhpi.h
#include "HTP/core/qhpi.h"
#include "flatbuffers/flexbuffers.h"


// Forward declarations for ExampleCustomOp kernel examplecustomop_float_32_
static uint32_t examplecustomop_float_32_Execute(QHPI_RuntimeHandle *handle,
                                      uint32_t num_outputs, QHPI_Tensor **outputs,
                                      uint32_t num_inputs, const QHPI_Tensor *const *inputs);
static float examplecustomop_float_32_CostFunc(const uint32_t num_inputs, const QHPI_Tensor *const *inputs);

// Common forward declarations for ExampleCustomOp
static const QHPI_Op* examplecustomopEarlyRewrite(const QHPI_Op *op);
static QHPI_Shape examplecustomopShapeRequired(const QHPI_Op *op);
static QHPI_Shape examplecustomopShapeLegal(const QHPI_Op *op, const QHPI_Shape* shape);
static const QHPI_Op* examplecustomopBuildTile(const QHPI_Op *op, const QHPI_Shape* start, const QHPI_Shape* extent);
static const QHPI_Op* examplecustomopLateRewrite(const QHPI_Op *op);

/*
 * QHPI Registration using hexnn_ffi.h API for ExampleCustomOp
 * Multiple kernels for different data type combinations
 */


// Input tensor signatures for ExampleCustomOp kernel examplecustomop_float_32_
// Includes both regular inputs and parameters as inputs
static QHPI_Tensor_Signature_v1 examplecustomop_float_32_InputSignatures[] = {

    {
        .element_type = QHPI_Float32,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    }

    ,{
        .element_type = QHPI_QUInt8,
        .layout = QHPI_Layout_Any,
        .storage = QHPI_Storage_Direct_OR_Indirect,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    }
};

static QHPI_Tensor_Signature_v1 examplecustomop_float_32_OutputSignatures[] = {

    {
        .element_type = QHPI_Float32,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM
    }
};

// Kernel definition for ExampleCustomOp kernel examplecustomop_float_32_
static QHPI_Kernel_v1 examplecustomop_float_32_Kernel = {
    .function_name = "examplecustomop_float_32_Execute",
    .function = examplecustomop_float_32_Execute,
    .resources = QHPI_RESOURCE_HVX,
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 2,
    .input_signature = examplecustomop_float_32_InputSignatures,
    .min_outputs = 1,
    .output_signature = examplecustomop_float_32_OutputSignatures,
    .cost_function = examplecustomop_float_32_CostFunc,
    .sync_block_size = 0,
    .precomputed_data_size = 0,
    .do_precomputation_function = nullptr,
    .function_with_precomputed_data = nullptr,
    .predicate = nullptr
};

// Array of all kernels for ExampleCustomOp
static QHPI_Kernel_v1 examplecustomopKernels[] = {

    examplecustomop_float_32_Kernel
};

// Operator info for ExampleCustomOp - exported for package registration
QHPI_OpInfo_v1 examplecustomopOpInfo = {
    .name = THIS_PKG_NAME_STR "::" "ExampleCustomOp",
    .num_kernels = 1,
    .kernels = examplecustomopKernels,
    .early_rewrite = examplecustomopEarlyRewrite,
    .shape_required = examplecustomopShapeRequired,
    .shape_legalized = examplecustomopShapeLegal,
    .build_tile = examplecustomopBuildTile,
    .late_rewrite = examplecustomopLateRewrite
};


/* QHPI execute function implementation for ExampleCustomOp kernel examplecustomop_float_32_ */
static uint32_t examplecustomop_float_32_Execute(QHPI_RuntimeHandle *handle,
                                      uint32_t num_outputs, QHPI_Tensor **outputs,
                                      uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
  /*
   * QHPI implementation code for ExampleCustomOp kernel examplecustomop_float_32_
   * This kernel handles the following data type combination:
   *   Input 0: QnnDatatype.QNN_DATATYPE_FLOAT_32 -> QHPI_Float32

   *   Parameter 0 (as input 1): CustomInitialData -> QHPI_QUInt8
   *   Output 0: QnnDatatype.QNN_DATATYPE_FLOAT_32 -> QHPI_Float32
   *
   * Input parameters:
   * - handle: Runtime handle for accessing runtime context
   * - num_outputs: Number of output tensors
   * - outputs: Array of output tensor pointers
   * - num_inputs: Number of input tensors (includes regular inputs + parameters)
   * - inputs: Array of input tensor pointers (regular inputs first, then parameters)
   */

  if (num_inputs != 2 || num_outputs != 1) {
    return QHPI_ErrorFatal;
  }

  const QHPI_Tensor* input = inputs[0];
  const QHPI_Tensor* custom_initial_data = inputs[1];
  QHPI_Tensor* output = outputs[0];

  QHPI_Shape input_shape = qhpi_tensor_shape(input);
  QHPI_Shape output_shape = qhpi_tensor_shape(output);

  // Validate input/output element counts match.
  uint32_t input_num_elements = 1;
  for (uint32_t i = 0; i < input_shape.rank; ++i) {
    input_num_elements *= input_shape.dims[i];
  }
  uint32_t output_num_elements = 1;
  for (uint32_t i = 0; i < output_shape.rank; ++i) {
    output_num_elements *= output_shape.dims[i];
  }
  if (input_num_elements != output_num_elements) {
    return QHPI_ErrorFatal;
  }

  // Parse alpha from CustomInitialData (flexbuffer-encoded).
  QHPI_Shape custom_data_shape = qhpi_tensor_shape(custom_initial_data);
  uint32_t custom_data_size = custom_data_shape.dims[custom_data_shape.rank - 1];
  const auto* custom_data_ptr =
      static_cast<const uint8_t*>(qhpi_tensor_raw_data(custom_initial_data));
  if (!custom_data_ptr) {
    return QHPI_ErrorFatal;
  }

  const flexbuffers::Map& custom_option_map =
      flexbuffers::GetRoot(custom_data_ptr, custom_data_size).AsMap();
  if (!custom_option_map["alpha"].IsFloat()) {
    return QHPI_ErrorFatal;
  }
  const float alpha = custom_option_map["alpha"].AsFloat();

  // Execute leaky ReLU: out = (in < 0) ? in * alpha : in.
  const auto* p_input =
      static_cast<const float*>(qhpi_tensor_raw_data(input));
  auto* p_output = static_cast<float*>(qhpi_tensor_raw_data(output));
  if (!p_input || !p_output) {
    return QHPI_ErrorFatal;
  }

  for (uint32_t i = 0; i < input_num_elements; ++i) {
    p_output[i] = (p_input[i] < 0.0f) ? p_input[i] * alpha : p_input[i];
  }

  return QHPI_Success;
}

static float examplecustomop_float_32_CostFunc(const uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
  /*
   * Cost estimation function for ExampleCustomOp kernel examplecustomop_float_32_
   * Return approximate number of cycles needed for this operation
   * with the specific data type combination. Used for estimating cycle
   * performance of a graph.
   *
   * Parameters:
   * - num_inputs: Number of input tensors
   * - inputs: Array of input tensor pointers
   */

  float cost = 1000.0;  // add cost computation here based on tensor sizes and data types
  return cost;
}

/*
 * Common stub implementations for ExampleCustomOp QHPI_OpInfo functions
 * These are shared across all kernels and provide default no-op implementations
 */

static const QHPI_Op* examplecustomopEarlyRewrite(const QHPI_Op *op)
{
  /*
   * Early rewrite function for ExampleCustomOp
   * Called during graph optimization phase
   * Return the original op if no rewriting is needed, or a new op if rewriting is required
   */
  return op;  // No rewriting by default
}

static QHPI_Shape examplecustomopShapeRequired(const QHPI_Op *op)
{
  /*
   * Shape required function for ExampleCustomOp
   * Specifies required input shapes for the operation
   * Return empty shape if no specific shape requirements
   */
  QHPI_Shape empty_shape = {0};  // Empty shape by default
  return empty_shape;
}

static QHPI_Shape examplecustomopShapeLegal(const QHPI_Op *op, const QHPI_Shape* shape)
{
  /*
   * Shape legal function for ExampleCustomOp
   * Validates if a given shape is legal for this operation
   * Return the shape if legal, or modified shape if not legal
   */
  return *shape;  // Accept the provided shape by default
}

static const QHPI_Op* examplecustomopBuildTile(const QHPI_Op *op, const QHPI_Shape* start, const QHPI_Shape* extent)
{
  /*
   * Build tile function for ExampleCustomOp
   * Creates a tiled version of the operation
   * Return a new op that operates on the specified tile, or op if tiling is not supported
   */
  return op;  // No tiling support by default
}

static const QHPI_Op* examplecustomopLateRewrite(const QHPI_Op *op)
{
  /*
   * Late rewrite function for ExampleCustomOp
   * Called during late optimization phase
   * Return the original op if no rewriting is needed, or a new op if rewriting is required
   */
  return op;  // No rewriting by default
}

// Array of all ExampleCustomOp operations for registration
static QHPI_OpInfo_v1 examplecustomop_ops[] = {
    examplecustomopOpInfo
};

// Registration function for ExampleCustomOp operations
extern "C" void register_examplecustomop_ops()
{
    qhpi_register_ops_v1(sizeof(examplecustomop_ops) / sizeof(examplecustomop_ops[0]), examplecustomop_ops, THIS_PKG_NAME_STR);
}
