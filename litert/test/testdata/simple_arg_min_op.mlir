module {
func.func @main(%arg0: tensor<1x128x64x32xf32>) -> tensor<1x128x64xi32> {
  %axis = "tfl.pseudo_const"() {
    value = dense<3> : tensor<i32>
  } : () -> tensor<i32>
  %0 = "tfl.arg_min"(%arg0, %axis) {
    output_type = i32
  } : (tensor<1x128x64x32xf32>, tensor<i32>) -> tensor<1x128x64xi32>
  return %0 : tensor<1x128x64xi32>
}
}
