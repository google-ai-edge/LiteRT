module {
func.func @main(%arg0: tensor<1x78xi1>) -> tensor<1x77xi1> {
  %cst_0 = "tfl.pseudo_const"() <{value = dense<[0, 1]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %cst_1 = "tfl.pseudo_const"() <{value = dense<[1, 77]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %0 = "tfl.slice"(%arg0, %cst_0, %cst_1) : (tensor<1x78xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x77xi1>
  return %0 : tensor<1x77xi1>
}
}
