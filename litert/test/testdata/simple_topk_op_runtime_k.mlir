
module {
  func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<i32>) -> (tensor<1x?xf32>, tensor<1x?xi32>) {
    %values, %indices = "tfl.topk_v2"(%arg0, %arg1) : (tensor<1x128xf32>, tensor<i32>) -> (tensor<1x?xf32>, tensor<1x?xi32>)
    return %values, %indices : tensor<1x?xf32>, tensor<1x?xi32>
  }
}
