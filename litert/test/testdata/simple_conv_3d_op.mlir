module {
func.func @main(%input: tensor<1x216x288x24x24xf32>, %filter: tensor<3x3x3x24x24xf32>, %bias: tensor<24xf32>) -> tensor<1x216x288x24x24xf32> {
  %0 = "tfl.conv_3d"(%input, %filter, %bias) <{
    padding = "SAME",
    stride_d = 1 : i32,
    stride_h = 1 : i32,
    stride_w = 1 : i32,
    dilation_d_factor = 1 : i32,
    dilation_h_factor = 1 : i32,
    dilation_w_factor = 1 : i32,
    fused_activation_function = "NONE"
    }> : (tensor<1x216x288x24x24xf32>, tensor<3x3x3x24x24xf32>, tensor<24xf32>) -> tensor<1x216x288x24x24xf32>
  return %0 : tensor<1x216x288x24x24xf32>
}
}
