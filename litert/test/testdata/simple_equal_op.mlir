module {
func.func @main(%arg0: tensor<1x128x8x128xf32>, %arg1: tensor<1x128x8x128xf32>) -> tensor<1x128x8x128xi1> {
  %0 = "tfl.equal"(%arg0, %arg1) : (tensor<1x128x8x128xf32>, tensor<1x128x8x128xf32>) -> tensor<1x128x8x128xi1>
  return %0 : tensor<1x128x8x128xi1>
}
}
