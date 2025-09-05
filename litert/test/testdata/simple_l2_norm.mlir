module {
func.func @main(%arg0: tensor<1x768xf32>) -> tensor<1x768xf32> {
  %0 = "tfl.l2_normalization"(%arg0) {fused_activation_function = "NONE"} : (tensor<1x768xf32>) -> tensor<1x768xf32>
  return %0 : tensor<1x768xf32>
}
}
