module {
func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<3xf32>
  %cst01 = "tfl.pseudo_const"() <{value = dense<[2.0, 1.0, 0.0]> : tensor<3xf32>}> : () -> tensor<3xf32>
  %cst02 = "tfl.pseudo_const"() <{value = dense<[5.0, 4.0, 1.0]> : tensor<3xf32>}> : () -> tensor<3xf32>
  %1 = tfl.mul %0, %cst01 {fused_activation_function = "NONE"} : tensor<3xf32>
  %2 = tfl.mul %0, %cst02 {fused_activation_function = "NONE"} : tensor<3xf32>
  %cst1 = "tfl.pseudo_const"() <{value = dense<[2.0, 1.0, 6.0]> : tensor<3xf32>}> : () -> tensor<3xf32>
  %3 = tfl.add %cst1, %2 {fused_activation_function = "NONE"} : tensor<3xf32>
  %4 = tfl.mul %1, %3 {fused_activation_function = "NONE"} : tensor<3xf32>
  return %4 : tensor<3xf32>
}
}