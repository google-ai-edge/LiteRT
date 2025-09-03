module {
func.func @main(%arg0: tensor<1x4096x320x1xf32>) -> tensor<1x4096x320x2xf32> {
  %repeats = arith.constant dense<[1, 1, 1, 2]> : tensor<4xi32>
  %0 = "tfl.tile"(%arg0, %repeats) :
        (tensor<1x4096x320x1xf32>, tensor<4xi32>) -> tensor<1x4096x320x2xf32>
  return %0 : tensor<1x4096x320x2xf32>
}
}
