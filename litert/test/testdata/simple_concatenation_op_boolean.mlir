module {
func.func @main(%arg0: tensor<1x77xi1>) -> tensor<1x78xi1> {
  %cst = "tfl.pseudo_const"() <{value = dense<true> : tensor<1x1xi1>}> : () -> tensor<1x1xi1>
  %0 = "tfl.concatenation"(%arg0, %cst) <{axis = 1 : i32, fused_activation_function = "NONE"}> : (tensor<1x77xi1>, tensor<1x1xi1>) -> tensor<1x78xi1>
  return %0 : tensor<1x78xi1>
}
}
