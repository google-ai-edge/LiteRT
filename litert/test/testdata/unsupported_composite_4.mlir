module {

func.func @main(%arg0: tensor<1x128x128xf32>, %arg1: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
  %0 = tfl.div %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<1x128x128xf32>
  %1 = stablehlo.composite "unsupported.composite" %0, %arg1 {composite_attributes = {}, decomposition = @unsupported.composite.impl}: (tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
  %2 = tfl.div %1, %0 {fused_activation_function = "NONE"} : tensor<1x128x128xf32>
  return %2 : tensor<1x128x128xf32>
}

func.func @unsupported.composite.impl(%arg0: tensor<1x128x128xf32>, %arg1: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
    %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<1x128x128xf32>
    %1 = tfl.mul %arg0, %0 {fused_activation_function = "NONE"} : tensor<1x128x128xf32>
    %2 = tfl.mul %0, %1 {fused_activation_function = "NONE"} : tensor<1x128x128xf32>
    return %2 : tensor<1x128x128xf32>
  }
}