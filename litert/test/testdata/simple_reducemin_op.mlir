module {
func.func @main(%arg0: tensor<1x32x64x128xf32>) -> tensor<1x32x64xf32> {
  %axis = "tfl.pseudo_const"() {value = dense<3> : tensor<1xi32> } : () -> tensor<1xi32>
  %0 = "tfl.reduce_min"(%arg0, %axis) {keep_dims = false} : (tensor<1x32x64x128xf32>, tensor<1xi32>) -> tensor<1x32x64xf32>
  return %0 : tensor<1x32x64xf32>
}
}
