module {
func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
    // Path 1
    %a1 = tfl.add %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    %b1 = tfl.mul %a1, %a1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    %c1 = tfl.add %b1, %b1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    
    // Path 2
    %a2 = tfl.add %arg1, %arg1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    %b2 = tfl.mul %a2, %a2 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    %c2 = tfl.add %b2, %b2 {fused_activation_function = "NONE"} : tensor<2x2xf32>
    
    return %c1, %c2 : tensor<2x2xf32>, tensor<2x2xf32>
}
}
