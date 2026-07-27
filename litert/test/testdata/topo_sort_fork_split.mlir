module {
func.func @main(%arg0: tensor<2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
    %cst = "tfl.pseudo_const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %0 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2x4xf32>
    %1:2 = "tfl.split"(%cst, %0) {num_splits = 2 : i32} : (tensor<i32>, tensor<2x4xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)
    %2 = tfl.mul %1#0, %1#0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    %3 = tfl.add %2, %2 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    %4 = tfl.add %1#1, %1#1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    return %3, %4 : tensor<2x2xf32>, tensor<2x2xf32>
}
}
