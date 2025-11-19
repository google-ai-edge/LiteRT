module {
func.func @main(%arg0: tensor<?x128x4xf32>, %arg1: tensor<?x128x4xf32>) -> tensor<?x128x4xf32> {
  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<?x128x4xf32>
  return %0 : tensor<?x128x4xf32>
}
}