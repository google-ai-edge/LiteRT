module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<2xf32> {tf_saved_model.index_path = ["x"]}, %arg1: tensor<2xf32> {tf_saved_model.index_path = ["y"]}) -> (tensor<2xf32> {tf_saved_model.index_path = ["product"]}, tensor<2xf32> {tf_saved_model.index_path = ["same_as_y"]})
      attributes {tf.entry_function = {inputs = "x,y", outputs = "product,y"}, tf_saved_model.exported_names = ["serving_default"]} {

    // Normal output: x * y
    %product = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2xf32>

    // Pass-through output: the input y itself, so the output tensor is also an
    // input tensor.
    return %product, %arg1 : tensor<2xf32>, tensor<2xf32>
  }
}
