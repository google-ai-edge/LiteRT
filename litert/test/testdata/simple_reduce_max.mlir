module {
func.func @main(%arg0: tensor<256x32x32x3xf32>, %arg1: tensor<1xi32>) -> tensor<256x32x32xf32> {
  %0 = "tfl.reduce_max"(%arg0, %arg1) {keep_dims = false}: (tensor<256x32x32x3xf32>, tensor<1xi32>) -> tensor<256x32x32xf32>
  func.return %0 : tensor<256x32x32xf32>
}
}