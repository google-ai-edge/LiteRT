module {
func.func @main(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "tfl.squeeze"(%arg0) <{squeeze_dims = [1]}> : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}
}