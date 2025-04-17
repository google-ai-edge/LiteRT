module {
func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<1x128xf32>, %arg2: tensor<1xi64>) -> tensor<1x128xf32> {
  %1 = "tfl.cast"(%arg2) : (tensor<1xi64>) -> tensor<1xf32>
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<1x128xf32>
  %2 = "tfl.add"(%1, %0) {fused_activation_function = "NONE"} : (tensor<1xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
  return %2 : tensor<1x128xf32>
}
}