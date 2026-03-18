module {
  func.func @main(%arg0: tensor<1x128x2304xf32>, %arg1: tensor<2304xf32>) -> tensor<1x128x2304xf32> {
    %0 = stablehlo.composite "odml.rms_norm" %arg0, %arg1 {composite_attributes = {epsilon = 9.99999997E-7 : f32}, decomposition = @odml.rms_norm.impl} : (tensor<1x128x2304xf32>, tensor<2304xf32>) -> tensor<1x128x2304xf32>
    return %0 : tensor<1x128x2304xf32>
  }
  func.func @odml.rms_norm.impl(%arg0: tensor<1x128x2304xf32>, %arg1: tensor<2304xf32>) -> tensor<1x128x2304xf32> {
    %0 = tfl.mul %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<1x128x2304xf32>
    return %0 : tensor<1x128x2304xf32>
  }
}
