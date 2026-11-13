module {
func.func @main(%arg0: tensor<8x4xf32>, %arg1: tensor<2x1xi32>, %arg2: tensor<2x4xf32>) -> tensor<8x4xf32> {
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %s = stablehlo.multiply %a, %b : tensor<f32>
    stablehlo.return %s : tensor<f32>
  }) : (tensor<8x4xf32>, tensor<2x1xi32>, tensor<2x4xf32>) -> tensor<8x4xf32>
  %1 = tfl.mul %0, %arg0 {fused_activation_function = "NONE"} : tensor<8x4xf32>
  return %1 : tensor<8x4xf32>
}
}
