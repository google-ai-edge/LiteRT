module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.metadata = {buffer_location = "outside flatbuffers", min_runtime_version = "2.17.0\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<2x4xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<2x4xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "serving_default_x:0", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tfl.pseudo_const"() <{value = dense<1.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
    %1 = "tfl.pseudo_const"() <{value = dense<0.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
    %2 = stablehlo.composite "odml.group_norm" %arg0, %0, %1 {composite_attributes = {channel_axis = -1 : i64, epsilon = 9.99999997E-7 : f32, num_groups = 2 : i64}, decomposition = @XlaCallModule_odml.group_norm.impl_0} : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    return %2 : tensor<2x4xf32>
  }
  func.func private @XlaCallModule_odml.group_norm.impl_0(%arg0: tensor<2x4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<2x4xf32> {
    %0 = tfl.sub %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2x4xf32>
    %1 = "tfl.pseudo_const"() <{value = dense<[2, 2, 2]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %2 = "tfl.reshape"(%0, %1) : (tensor<2x4xf32>, tensor<3xi32>) -> tensor<2x2x2xf32>
    %3 = tfl.mul %2, %2 {fused_activation_function = "NONE"} : tensor<2x2x2xf32>
    %4 = "tfl.pseudo_const"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "tfl.sum"(%3, %4) <{keep_dims = false}> : (tensor<2x2x2xf32>, tensor<1xi32>) -> tensor<2x2xf32>
    %6 = "tfl.pseudo_const"() <{value = dense<9.99999997E-7> : tensor<f32>}> : () -> tensor<f32>
    %7 = tfl.add(%5, %6) <{fused_activation_function = "NONE"}> : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
    %8 = "tfl.pseudo_const"() <{value = dense<[2, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %9 = "tfl.reshape"(%7, %8) : (tensor<2x2xf32>, tensor<3xi32>) -> tensor<2x2x1xf32>
    %10 = "tfl.rsqrt"(%9) : (tensor<2x2x1xf32>) -> tensor<2x2x1xf32>
    %11 = tfl.mul(%2, %10) <{fused_activation_function = "NONE"}> : (tensor<2x2x2xf32>, tensor<2x2x1xf32>) -> tensor<2x2x2xf32>
    %12 = "tfl.pseudo_const"() <{value = dense<[2, 4]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %13 = "tfl.reshape"(%11, %12) : (tensor<2x2x2xf32>, tensor<2xi32>) -> tensor<2x4xf32>
    %14 = tfl.mul(%13, %arg1) <{fused_activation_function = "NONE"}> : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    %15 = tfl.add(%14, %arg2) <{fused_activation_function = "NONE"}> : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    return %15 : tensor<2x4xf32>
  }
}
