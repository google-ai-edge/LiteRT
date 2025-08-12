module {
func.func @main(%arg0: tensor<128x4x2x128xf32>) -> tensor<128x4x4x128xf32> {
  %repeats = arith.constant dense<[1, 1, 2, 1]> : tensor<4xi32>
  %0 = "tfl.tile"(%arg0, %repeats) : 
        (tensor<128x4x2x128xf32>, tensor<4xi32>) -> tensor<128x4x4x128xf32>
  return %0 : tensor<128x4x4x128xf32>
}
}
