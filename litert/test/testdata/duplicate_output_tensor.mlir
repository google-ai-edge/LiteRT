module {
  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)
      attributes {tf.entry_function = {inputs = "x,y", outputs = "product,product_again"}} {

    %product = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2xf32>

    // The same value is returned twice, so one tensor is listed twice as
    // output.
    return %product, %product : tensor<2xf32>, tensor<2xf32>
  }
}
