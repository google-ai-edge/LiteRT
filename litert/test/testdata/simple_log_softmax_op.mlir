module {
func.func @main(%arg0: tensor<8x128xf32>) -> tensor<8x128xf32> {
  %0 = "tfl.log_softmax"(%arg0) : (tensor<8x128xf32>) -> tensor<8x128xf32>
  return %0 : tensor<8x128xf32>
}
}
