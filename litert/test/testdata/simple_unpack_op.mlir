module {
func.func @main(%arg0: tensor<1x4x4x3xf32>) -> (tensor<1x4x3xf32>, tensor<1x4x3xf32>, tensor<1x4x3xf32>, tensor<1x4x3xf32>) {
  %0:4 = "tfl.unpack"(%arg0) {axis = 2 : i32, num = 4 : i32} : (tensor<1x4x4x3xf32>) -> (tensor<1x4x3xf32>, tensor<1x4x3xf32>, tensor<1x4x3xf32>, tensor<1x4x3xf32>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<1x4x3xf32>, tensor<1x4x3xf32>, tensor<1x4x3xf32>, tensor<1x4x3xf32>
}
}
