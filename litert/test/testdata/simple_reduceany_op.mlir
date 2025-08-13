module {
func.func @main(%arg0: tensor<1x32x64x128xi1>) -> tensor<1x32x64xi1> {
  %axis = "tfl.pseudo_const"() {value = dense<3> : tensor<1xi32> } : () -> tensor<1xi32>
  %0 = "tfl.reduce_any"(%arg0, %axis) {keep_dims = false} : (tensor<1x32x64x128xi1>, tensor<1xi32>) -> tensor<1x32x64xi1>
  return %0 : tensor<1x32x64xi1>
}
}
