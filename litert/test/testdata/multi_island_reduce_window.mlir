module {
func.func @main(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32> {
  %0 = tfl.mul %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<1x16x16x4xf32>
  %1 = tfl.add %0, %arg0 {fused_activation_function = "NONE"} : tensor<1x16x16x4xf32>
  %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %2 = "stablehlo.reduce_window"(%1, %cst) <{padding = dense<0> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 2, 2, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %m = stablehlo.maximum %a, %b : tensor<f32>
    stablehlo.return %m : tensor<f32>
  }) : (tensor<1x16x16x4xf32>, tensor<f32>) -> tensor<1x8x8x4xf32>
  %3 = tfl.mul %2, %arg1 {fused_activation_function = "NONE"} : tensor<1x8x8x4xf32>
  return %3 : tensor<1x8x8x4xf32>
}
}
