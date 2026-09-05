module {
func.func @main(%arg0: tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32> {
  %0 = "tfl.local_response_normalization"(%arg0) <{radius = 5 : i32, bias = 9.000000e+00 : f32, alpha = 4.000000e+00 : f32, beta = 7.500000e-01 : f32}> : (tensor<1x4x4x8xf32>) -> tensor<1x4x4x8xf32>
  return %0 : tensor<1x4x4x8xf32>
}
}
