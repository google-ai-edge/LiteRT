module {
func.func @main(%arg0: tensor<1x128x1xf32>) -> tensor<1x128x1xf32> {
  %0 = "tfl.ceil"(%arg0) : (tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
  return %0 : tensor<1x128x1xf32>
}
}
