module {
func.func @main(%arg0: tensor<i1>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "tfl.if"(%arg0) ({
    %1 = func.call @if_then(%arg1, %arg2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "tfl.yield"(%1) : (tensor<4xf32>) -> ()
  }, {
    %1 = func.call @if_else(%arg1, %arg2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "tfl.yield"(%1) : (tensor<4xf32>) -> ()
  }) : (tensor<i1>) -> tensor<4xf32>
  %2 = tfl.mul %0, %arg1 {fused_activation_function = "NONE"} : tensor<4xf32>
  return %2 : tensor<4xf32>
}

func.func private @if_then(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func private @if_else(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
}
