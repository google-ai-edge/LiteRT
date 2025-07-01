module {
func.func @main(%arg0: tensor<1x128x64x32xf32>) -> tensor<1x128x64x32xf32> {
  %0 = "tfl.neg"(%arg0) : (tensor<1x128x64x32xf32>) -> tensor<1x128x64x32xf32>
  return %0 : tensor<1x128x64x32xf32>
}
}
