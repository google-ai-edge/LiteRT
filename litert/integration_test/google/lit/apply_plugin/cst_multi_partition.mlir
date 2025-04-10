// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o %t && apply_plugin_main_for_test  --model=%t partition | flatbuffer_translate -tflite-flatbuffer-to-mlir | FileCheck %s

func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
	%cst = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %0 = tfl.mul %arg0, %cst {fused_activation_function = "NONE"} : tensor<4xf32>
  %1 = tfl.add %0, %0 {fused_activation_function = "NONE"} : tensor<4xf32>
  %2 = tfl.mul %1, %cst {fused_activation_function = "NONE"} : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// CHECK-LABEL: @main
// CHECK:       "tfl.custom"(%arg0) <{custom_code = "DISPATCH_OP", custom_option = #tfl<const_bytes : "0x">}> : (tensor<4xf32>) -> tensor<4xf32>
// CHECK:       "tfl.custom"(%1) <{custom_code = "DISPATCH_OP", custom_option = #tfl<const_bytes : "0x">}> : (tensor<4xf32>) -> tensor<4xf32>

// CHECK-LABEL: @fn_1(%arg0: tensor<4xf32>)
// CHECK:       const
// CHECK:       tfl.mul
// CHECK:       return

// CHECK-LABEL: @fn_2(%arg0: tensor<4xf32>)
// CHECK:       const
// CHECK:       tfl.mul
// CHECK:       return
