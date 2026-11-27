module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x1x3xf32> {
    %axis = "tfl.pseudo_const"() {
      value = dense<1> : tensor<i32>
    } : () -> tensor<i32>
    %0 = "tfl.expand_dims"(%arg0, %axis) : (tensor<2x3xf32>, tensor<i32>) -> tensor<2x1x3xf32>
    return %0 : tensor<2x1x3xf32>
  }
}
