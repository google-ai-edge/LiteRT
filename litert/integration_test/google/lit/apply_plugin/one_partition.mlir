// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o %t && apply_plugin_main_for_test  --model=%t partition | flatbuffer_translate -tflite-flatbuffer-to-mlir | FileCheck %s

func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK: func.func @main(%arg0: tensor<4xf32> {tf_saved_model.index_path = ["arg0"]}, %arg1: tensor<4xf32> {tf_saved_model.index_path = ["arg1"]}) -> (tensor<4xf32> {tf_saved_model.index_path = ["tfl.mul"]}) attributes {tf.entry_function = {inputs = "arg0,arg1", outputs = "tfl.mul"}, tf_saved_model.exported_names = ["<placeholder signature>"]} {
// CHECK:   %0 = "tfl.custom"(%arg0, %arg1) <{custom_code = {{.*}}, custom_option = #tfl<const_bytes : "{{.*}}">}> : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK:   return %0 : tensor<4xf32>

// CHECK: func.func private @fn_1(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// CHECK:   %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<4xf32>
// CHECK:   return %0 : tensor<4xf32>
