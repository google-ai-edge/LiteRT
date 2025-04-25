module {
func.func @main(%arg0: tensor<2x1x3xi32>) -> tensor<2x1x3xi32> {
  %cst = arith.constant dense<[[[10, 20, 30]], [[40, 50, 60]]]> : tensor<2x1x3xi32>
  %0 = tfl.add %arg0, %cst {fused_activation_function = "NONE"} : tensor<2x1x3xi32>
  return %0 : tensor<2x1x3xi32>
}
}