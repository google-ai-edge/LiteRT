module {
  func.func @main(%indices: tensor<2x4xi32>) -> tensor<1x128x4x128xf32> {
    %updates = "tfl.pseudo_const"() <{value = dense<[1.0, 2.0]> : tensor<2xf32>}> : () -> tensor<2xf32>
    %shape = "tfl.pseudo_const"() <{value = dense<[1,128,4,128]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %output = "tfl.scatter_nd"(%indices, %updates, %shape) : (tensor<2x4xi32>, tensor<2xf32>, tensor<4xi32>) -> tensor<1x128x4x128xf32>
    return %output : tensor<1x128x4x128xf32>
  }
}
