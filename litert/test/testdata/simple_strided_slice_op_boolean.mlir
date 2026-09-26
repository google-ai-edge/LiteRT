module {
func.func @main(%arg0: tensor<1x78xi1>) -> tensor<1x77xi1> {
  %begin = "tfl.pseudo_const"() {
    value = dense<[0, 1]> : tensor<2xi32>
  } : () -> tensor<2xi32>
  %end = "tfl.pseudo_const"() {
    value = dense<[1, 78]> : tensor<2xi32>
  } : () -> tensor<2xi32>
  %strides = "tfl.pseudo_const"() {
    value = dense<[1, 1]> : tensor<2xi32>
  } : () -> tensor<2xi32>
  %0 = "tfl.strided_slice"(%arg0, %begin, %end, %strides) {
    begin_mask = 0 : i32,
    end_mask = 0 : i32,
    ellipsis_mask = 0 : i32,
    new_axis_mask = 0 : i32,
    shrink_axis_mask = 0 : i32,
    offset = false
  } : (tensor<1x78xi1>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x77xi1>
  return %0 : tensor<1x77xi1>
}
}
