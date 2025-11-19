module {
  func.func @main(%arg0: tensor<?x2x3xf32>, %arg1: tensor<?x2x3xf32>) -> tensor<?x2x3xf32> {
    %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<?x2x3xf32>, tensor<?x2x3xf32>) -> tensor<?x2x3xf32>
    return %0 : tensor<?x2x3xf32>
  }
}