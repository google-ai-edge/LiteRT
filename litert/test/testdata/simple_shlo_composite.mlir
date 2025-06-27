module {
func.func @main(%arg0 : tensor<i64>) -> tensor<i64> {
  %0 = stablehlo.composite "stablehlo.add_n" %arg0 {
  composite_attributes = { an_attribute = "foo", meaning_of_life = 42 },
  version = 3: i32,
  decomposition = @add_n.impl
  } : (tensor<i64>) -> tensor<i64>
  func.return %0 : tensor<i64>
}

func.func @add_n.impl(%arg0: tensor<i64>) -> tensor<i64> {
  %0 = stablehlo.constant dense<2> : tensor<i64>
  %1 = stablehlo.add %arg0, %0 : tensor<i64>
  func.return %1 : tensor<i64>
}
}
