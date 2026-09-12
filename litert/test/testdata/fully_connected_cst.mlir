module {
  func.func @main(%arg0: tensor<1x4xf32>) -> tensor<1x2xf32> {
    %weights = "tfl.pseudo_const"() <{value = dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>}> : () -> tensor<2x4xf32>
    %bias = "tfl.pseudo_const"() <{value = dense<[0.1, 0.2]> : tensor<2xf32>}> : () -> tensor<2xf32>
    %0 = "tfl.fully_connected"(%arg0, %weights, %bias) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true, weights_format = "DEFAULT"}> : (tensor<1x4xf32>, tensor<2x4xf32>, tensor<2xf32>) -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }
}
