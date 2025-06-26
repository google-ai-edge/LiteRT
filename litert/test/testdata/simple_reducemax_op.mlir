module {
func.func @main(%arg0: tensor<1x12x1x1xf32>) -> tensor<1x12x1xf32> {
  %axis = "tfl.pseudo_const"() {value = dense<3> : tensor<1xi32> } : () -> tensor<1xi32>
  %0 = "tfl.reduce_max"(%arg0, %axis) {keep_dims = false} : (tensor<1x12x1x1xf32>, tensor<1xi32>) -> tensor<1x12x1xf32>
  return %0 : tensor<1x12x1xf32>
}
}
