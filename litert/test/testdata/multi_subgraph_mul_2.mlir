module {

func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

func.func @func1(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

func.func @func2(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<i32>
  return %0 : tensor<i32>
}

func.func @func3(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<f32>
  return %0 : tensor<f32>
}

}