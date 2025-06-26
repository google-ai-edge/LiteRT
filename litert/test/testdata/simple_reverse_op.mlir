module {
func.func @main(%arg0: tensor<1x128x1xf32>) -> tensor<1x128x1xf32> {
  %axis = "tfl.pseudo_const"() {value = dense<1> : tensor<1xi32> } : () -> tensor<1xi32>
  %0 = "tfl.reverse_v2"(%arg0, %axis) : (tensor<1x128x1xf32>, tensor<1xi32>) -> tensor<1x128x1xf32>
  return %0 : tensor<1x128x1xf32>
}
}
