// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o %t && apply_plugin_main_for_test  --model=%t partition | flatbuffer_translate -tflite-flatbuffer-to-mlir | FileCheck %s

func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    %1 = tfl.mul %0, %0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    %2 = tfl.sub %1, %0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    %3 = tfl.sub %2, %0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    return %3 : tensor<2x2xf32>
}

// CHECK:  func.func @main(%arg0: tensor<2x2xf32> {tf_saved_model.index_path = ["arg0"]}) -> (tensor<2x2xf32> {tf_saved_model.index_path = ["tfl.sub1"]}) attributes {tf.entry_function = {inputs = "arg0", outputs = "tfl.sub1"}, tf_saved_model.exported_names = ["<placeholder signature>"]} {
// CHECK:    %0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
// CHECK:    %1 = "tfl.custom"(%0) <{custom_code = "DISPATCH_OP", custom_option = #tfl<const_bytes : "0x">}> : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK:    %2 = "tfl.custom"(%1, %0) <{custom_code = "DISPATCH_OP", custom_option = #tfl<const_bytes : "0x">}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK:    return %2 : tensor<2x2xf32>

// CHECK-DAG:  func.func private @fn_{{[1-2]}}(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-DAG-NEXT:    %0 = tfl.mul %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
// CHECK-DAG-NEXT:    return %0 : tensor<2x2xf32>

// CHECK-DAG:  func.func private @fn_{{[1-2]}}(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-DAG-NEXT:    %0 = tfl.sub %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
// CHECK-DAG-NEXT:    %1 = tfl.sub %0, %arg1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
// CHECK-DAG-NEXT:    return %1 : tensor<2x2xf32>