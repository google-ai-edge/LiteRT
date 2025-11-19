
module {
  func.func @main(%arg0: tensor<1x128xf32>) -> (tensor<1x10xf32>, tensor<1x10xi32>) {
    %k = "tfl.pseudo_const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
    %values, %indices = "tfl.topk_v2"(%arg0, %k) : (tensor<1x128xf32>, tensor<i32>) -> (tensor<1x10xf32>, tensor<1x10xi32>)
    return %values, %indices : tensor<1x10xf32>, tensor<1x10xi32>
  }
}
