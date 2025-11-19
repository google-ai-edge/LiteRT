module {
func.func @main(%arg0: tensor<1x96x96x16xf32>, %arg1: tensor<1x1x16xf32>) -> tensor<1x96x96x16xf32> {
  %0 = "tfl.prelu"(%arg0, %arg1) : (tensor<1x96x96x16xf32>, tensor<1x1x16xf32>) -> tensor<1x96x96x16xf32>
  return %0 : tensor<1x96x96x16xf32>
}
}
