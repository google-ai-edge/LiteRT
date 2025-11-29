module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.metadata = {buffer_location = "outside flatbuffers", min_runtime_version = "2.17.0\00\00\00\00\00\00\00\00\00\00"}, tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<1x2x3x4xf32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1x2x3x4xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "serving_default_x:0", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tfl.pseudo_const"() <{value = dense<1.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
    %1 = "tfl.pseudo_const"() <{value = dense<0.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
    %2 = stablehlo.composite "odml.group_norm" %arg0, %0, %1 {composite_attributes = {channel_axis = -1 : i64, epsilon = 9.99999997E-7 : f32, num_groups = 2 : i64}, decomposition = @XlaCallModule_odml.group_norm.impl_0} : (tensor<1x2x3x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x2x3x4xf32>
    return %2 : tensor<1x2x3x4xf32>
  }
  func.func private @XlaCallModule_odml.group_norm.impl_0(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<1x2x3x4xf32> {
    %0 = "tfl.pseudo_const"() <{value = dense<[1, 2, 3, 2, 2]> : tensor<5xi32>}> : () -> tensor<5xi32>
    %1 = "tfl.reshape"(%arg0, %0) : (tensor<1x2x3x4xf32>, tensor<5xi32>) -> tensor<1x2x3x2x2xf32>
    %2 = "tfl.pseudo_const"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "tfl.sum"(%1, %2) <{keep_dims = true}> : (tensor<1x2x3x2x2xf32>, tensor<1xi32>) -> tensor<1x2x3x1x2xf32>
    %4 = "tfl.pseudo_const"() <{value = dense<5.000000e-01> : tensor<f32>}> : () -> tensor<f32>
    %5 = tfl.mul(%3, %4) <{fused_activation_function = "NONE"}> : (tensor<1x2x3x1x2xf32>, tensor<f32>) -> tensor<1x2x3x1x2xf32>
    %6 = tfl.sub(%1, %5) <{fused_activation_function = "NONE"}> : (tensor<1x2x3x2x2xf32>, tensor<1x2x3x1x2xf32>) -> tensor<1x2x3x2x2xf32>
    %7 = tfl.mul %6, %6 {fused_activation_function = "NONE"} : tensor<1x2x3x2x2xf32>
    %8 = "tfl.sum"(%7, %2) <{keep_dims = true}> : (tensor<1x2x3x2x2xf32>, tensor<1xi32>) -> tensor<1x2x3x1x2xf32>
    %9 = tfl.mul(%8, %4) <{fused_activation_function = "NONE"}> : (tensor<1x2x3x1x2xf32>, tensor<f32>) -> tensor<1x2x3x1x2xf32>
    %10 = "tfl.pseudo_const"() <{value = dense<9.99999997E-7> : tensor<f32>}> : () -> tensor<f32>
    %11 = tfl.add(%9, %10) <{fused_activation_function = "NONE"}> : (tensor<1x2x3x1x2xf32>, tensor<f32>) -> tensor<1x2x3x1x2xf32>
    %12 = "tfl.rsqrt"(%11) : (tensor<1x2x3x1x2xf32>) -> tensor<1x2x3x1x2xf32>
    %13 = tfl.mul(%6, %12) <{fused_activation_function = "NONE"}> : (tensor<1x2x3x2x2xf32>, tensor<1x2x3x1x2xf32>) -> tensor<1x2x3x2x2xf32>
    %14 = "tfl.pseudo_const"() <{value = dense<[1, 2, 3, 4]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %15 = "tfl.reshape"(%13, %14) : (tensor<1x2x3x2x2xf32>, tensor<4xi32>) -> tensor<1x2x3x4xf32>
    %16 = tfl.mul(%15, %arg1) <{fused_activation_function = "NONE"}> : (tensor<1x2x3x4xf32>, tensor<4xf32>) -> tensor<1x2x3x4xf32>
    %17 = tfl.add(%16, %arg2) <{fused_activation_function = "NONE"}> : (tensor<1x2x3x4xf32>, tensor<4xf32>) -> tensor<1x2x3x4xf32>
    return %17 : tensor<1x2x3x4xf32>
  }
}
