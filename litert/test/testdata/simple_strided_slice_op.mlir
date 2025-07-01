module {
func.func @main(%arg0: tensor<1x128x4x256xf32>) -> tensor<1x128x4x128xf32> {
  %begin = "tfl.pseudo_const"() {
    value = dense<[0, 0, 0, 0]> : tensor<4xi32>
  } : () -> tensor<4xi32>
  %end = "tfl.pseudo_const"() {
    value = dense<[1, 128, 4, 256]> : tensor<4xi32>
  } : () -> tensor<4xi32>
  %strides = "tfl.pseudo_const"() {
    value = dense<[1, 1, 1, 2]> : tensor<4xi32>
  } : () -> tensor<4xi32>
  %0 = "tfl.strided_slice"(%arg0, %begin, %end, %strides) {
    begin_mask = 0 : i32,
    end_mask = 0 : i32,
    ellipsis_mask = 0 : i32,
    new_axis_mask = 0 : i32,
    shrink_axis_mask = 0 : i32,
    offset = false
  } : (tensor<1x128x4x256xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>) -> tensor<1x128x4x128xf32>
  return %0 : tensor<1x128x4x128xf32>
}
}
