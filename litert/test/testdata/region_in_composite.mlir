module {
func.func @main(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32> {
  %0 = stablehlo.composite "odml.unsupported_pool" %arg0 {decomposition = @pool_decomp} : (tensor<1x16x16x4xf32>) -> tensor<1x8x8x4xf32>
  %1 = tfl.mul %0, %arg1 {fused_activation_function = "NONE"} : tensor<1x8x8x4xf32>
  return %1 : tensor<1x8x8x4xf32>
}

func.func private @pool_decomp(%arg0: tensor<1x16x16x4xf32>) -> tensor<1x8x8x4xf32> {
  %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<0> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 2, 2, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %m = stablehlo.maximum %a, %b : tensor<f32>
    stablehlo.return %m : tensor<f32>
  }) : (tensor<1x16x16x4xf32>, tensor<f32>) -> tensor<1x8x8x4xf32>
  return %0 : tensor<1x8x8x4xf32>
}
}
