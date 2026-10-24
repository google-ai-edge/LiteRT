// STABLEHLO_SORT has no TFLite kernel and is only serialized from VHLO, so
// this model is generated with:
//   flatbuffer_translate -mlir-to-tflite-flatbuffer --emit-stablehlo-ops=true
module {
func.func @main(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
  %0 = "vhlo.sort_v1"(%arg0) <{dimension = #vhlo.integer_v1<0 : i64>, is_stable = #vhlo.bool_v1<true>}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %c = "vhlo.compare_v1"(%a, %b) <{compare_type = #vhlo<comparison_type_v1 TOTALORDER>, comparison_direction = #vhlo<comparison_direction_v1 GT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "vhlo.return_v1"(%c) : (tensor<i1>) -> ()
  }) : (tensor<16xf32>) -> tensor<16xf32>
  %1 = tfl.mul %0, %arg1 {fused_activation_function = "NONE"} : tensor<16xf32>
  return %1 : tensor<16xf32>
}
}
