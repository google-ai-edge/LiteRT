module {
  func.func @main(%arg0: tensor<1x768xf32>) -> tensor<1x768xf32> {
    %0 = stablehlo.composite "odml.l2_norm" %arg0 {composite_attributes = {axis = -1 : i64, epsilon = 9.99999997E-7 : f32}, decomposition = @odml.l2_norm.impl} : (tensor<1x768xf32>) -> tensor<1x768xf32>
    return %0 : tensor<1x768xf32>
  }
  func.func @odml.l2_norm.impl(%arg0: tensor<1x768xf32>) -> tensor<1x768xf32> {
    %0 = tfl.mul %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<1x768xf32>
    %1 = "tfl.pseudo_const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = "tfl.sum"(%0, %1) <{keep_dims = true}> : (tensor<1x768xf32>, tensor<1xi32>) -> tensor<1x1xf32>
    %3 = "tfl.pseudo_const"() <{value = dense<9.99999997E-7> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
    %4 = tfl.add %2, %3 {fused_activation_function = "NONE"} : tensor<1x1xf32>
    %5 = "tfl.rsqrt"(%4) : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %6 = tfl.mul(%arg0, %5) <{fused_activation_function = "NONE"}> : (tensor<1x768xf32>, tensor<1x1xf32>) -> tensor<1x768xf32>
    return %6 : tensor<1x768xf32>
  }
}

