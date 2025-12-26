module {
  func.func @main(%arg0: tensor<1x4x4x1xf32>) -> tensor<1x3x3x1xf32> {
    %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>

    %result = "stablehlo.reduce_window"(%arg0, %init) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %sum = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) {
      window_dimensions = array<i64: 1, 2, 2, 1>,
      window_strides = array<i64: 1, 1, 1, 1>,
      base_dilations = array<i64: 1, 1, 1, 1>,
      window_dilations = array<i64: 1, 1, 1, 1>,
      padding = dense<[[0, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>
    } : (tensor<1x4x4x1xf32>, tensor<f32>) -> tensor<1x3x3x1xf32>

    return %result : tensor<1x3x3x1xf32>
  }
}