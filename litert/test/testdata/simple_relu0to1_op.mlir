module {
func.func @main(%arg0: tensor<8x100x1xf32>) -> tensor<8x100x1xf32> {
  %0 = "tfl.relu_0_to_1"(%arg0) : (tensor<8x100x1xf32>) -> tensor<8x100x1xf32>
  return %0 : tensor<8x100x1xf32>
}
}
