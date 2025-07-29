module {
  func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2xf32> {
    %0 = "tfl.mul"(%arg0, %arg0) {fused_activation_function = "NONE"} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %cst = "tfl.pseudo_const"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "tfl.mean"(%0, %cst) <{keep_dims = false}> : (tensor<2x2xf32>, tensor<1xi32>) -> tensor<2xf32>
    %2 = "tfl.sqrt"(%1) : (tensor<2xf32>) -> tensor<2xf32>
    return %2 : tensor<2xf32>
  }
}