module {
  func.func @main(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<4xf32>)
      attributes {tf.entry_function = {inputs = "input", outputs = "normal_output,constant_output"}} {

    // Normal output: input + input
    %normal = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2xf32>

    // Constant output: always returns [1.0, 2.0, 3.0, 4.0]
    %const = "tfl.pseudo_const"() {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : () -> tensor<4xf32>

    return %normal, %const : tensor<2xf32>, tensor<4xf32>
  }
}
