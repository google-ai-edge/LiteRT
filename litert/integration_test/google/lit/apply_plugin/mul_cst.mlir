// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o %t && apply_plugin_main_for_test  --model=%t partition | flatbuffer_translate -tflite-flatbuffer-to-mlir | FileCheck %s

func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
	%cst = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %0 = tfl.mul %arg0, %cst {fused_activation_function = "NONE"} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// NOTE: Constant moved into partition.

// CHECK: func.func @main(%arg0: tensor<4xf32> {tf_saved_model.index_path = ["arg0"]}) -> (tensor<4xf32> {tf_saved_model.index_path = ["tfl.mul"]}) attributes {tf.entry_function = {inputs = "arg0", outputs = "tfl.mul"}, tf_saved_model.exported_names = ["<placeholder signature>"]} {
// CHECK:   %0 = "tfl.custom"(%arg0) <{custom_code = "DISPATCH_OP", custom_option = #tfl<const_bytes : "0x">}> : (tensor<4xf32>) -> tensor<4xf32>
// CHECK:   return %0 : tensor<4xf32>

// CHECK: func.func private @fn_1(%arg0: tensor<4xf32>) -> tensor<4xf32>
// CHECK:   %0 = "tfl.pseudo_const"() <{value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>}> : () -> tensor<4xf32>
// CHECK:   %1 = tfl.mul %arg0, %0 {fused_activation_function = "NONE"} : tensor<4xf32>
// CHECK:   return %1 : tensor<4xf32>

