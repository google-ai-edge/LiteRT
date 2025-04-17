// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o %t && apply_plugin_main_for_test  --model=%t partition | flatbuffer_translate -tflite-flatbuffer-to-mlir | FileCheck %s

func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    %1 = tfl.mul %0, %0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    %2 = tfl.add %1, %1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    %3 = tfl.mul %2, %2 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    return %3 : tensor<2x2xf32>
}

// CHECK: func.func @main(%arg0: tensor<2x2xf32> {tf_saved_model.index_path = ["arg0"]}) -> (tensor<2x2xf32> {tf_saved_model.index_path = ["tfl.mul1"]}) attributes {tf.entry_function = {inputs = "arg0", outputs = "tfl.mul1"}, tf_saved_model.exported_names = ["<placeholder signature>"]} {
// CHECK:     %0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
// CHECK:     %1 = "tfl.custom"(%0) <{custom_code = {{.*}}, custom_option = #tfl<const_bytes : "{{.*}}">}> : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK:     %2 = tfl.add %1, %1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
// CHECK:     %3 = "tfl.custom"(%2) <{custom_code = {{.*}}, custom_option = #tfl<const_bytes : "{{.*}}">}> : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK:     return %3 : tensor<2x2xf32>

// CHECK: func.func private @fn_1(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK:     %0 = tfl.mul %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
// CHECK:     return %0 : tensor<2x2xf32>

// CHECK: func.func private @fn_2(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK:     %0 = tfl.mul %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
// CHECK:     return %0 : tensor<2x2xf32>
