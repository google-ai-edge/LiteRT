module {
  func.func @main(%arg0: tensor<1x128x128xf32>, %arg1: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
    %0 = stablehlo.composite "odml.softmax" %arg0, %arg1 {composite_attributes = {}, decomposition = @odml.softmax.impl}: (tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
    return %0 : tensor<1x128x128xf32>
  }
  func.func @odml.softmax.impl(%arg0: tensor<1x128x128xf32>, %arg1: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
    %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<1x128x128xf32>
    return %0 : tensor<1x128x128xf32>
  }
}
