module {
  func.func @main(%input: tensor<1x3x1x1x1xf32>) -> tensor<1x3x3x2x2xf32> {
    %output_shape = "tfl.pseudo_const"() {value = dense<[1, 3, 3, 2, 2]> : tensor<5xi32>} : () -> tensor<5xi32>
    %filter = "tfl.pseudo_const"() {value = dense<1.0> : tensor<1x2x2x2x1xf32>} : () -> tensor<1x2x2x2x1xf32>
    %bias = "tfl.no_value"() {value = unit} : () -> none
    %0 = "tfl.conv_3d_transpose"(%output_shape, %filter, %input, %bias) {
      data_format = "NDHWC",
      dilation_d_factor = 1 : i32,
      dilation_h_factor = 2 : i32,
      dilation_w_factor = 1 : i32,
      fused_activation_function = "NONE",
      padding = "VALID",
      stride_d = 1 : i32,
      stride_h = 1 : i32,
      stride_w = 1 : i32
    } : (tensor<5xi32>, tensor<1x2x2x2x1xf32>, tensor<1x3x1x1x1xf32>, none) -> tensor<1x3x3x2x2xf32>
    return %0 : tensor<1x3x3x2x2xf32>
  }
}
