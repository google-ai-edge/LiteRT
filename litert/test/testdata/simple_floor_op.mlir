module {
func.func @main(%arg0: tensor<1x128x1xf32>) -> tensor<1x128x1xf32> {
  %0 = "tfl.floor"(%arg0) : (tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
  return %0 : tensor<1x128x1xf32>
}
}
