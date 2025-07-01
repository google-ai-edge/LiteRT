module {
func.func @main(%arg0: tensor<2x5x5x1xf32>, %arg1: tensor<2x1x1x1xf32>) -> tensor<2x5x5x1xf32> {
  %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "simple_op", custom_option = #tfl<const_bytes : "0x00">} : (tensor<2x5x5x1xf32>, tensor<2x1x1x1xf32>) -> tensor<2x5x5x1xf32>
  return %0 : tensor<2x5x5x1xf32>
}
}