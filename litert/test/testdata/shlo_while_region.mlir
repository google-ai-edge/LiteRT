// STABLEHLO_WHILE has no TFLite kernel and is only serialized from VHLO, so
// this model is generated with:
//   flatbuffer_translate -mlir-to-tflite-flatbuffer --emit-stablehlo-ops=true
module {
func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  %0:3 = "vhlo.while_v1"(%arg0, %arg1, %arg2) ({
  ^bb0(%i: tensor<i32>, %n: tensor<i32>, %x: tensor<4xf32>):
    %lt = "vhlo.compare_v1"(%i, %n) <{compare_type = #vhlo<comparison_type_v1 NOTYPE>, comparison_direction = #vhlo<comparison_direction_v1 LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "vhlo.return_v1"(%lt) : (tensor<i1>) -> ()
  }, {
  ^bb0(%i: tensor<i32>, %n: tensor<i32>, %x: tensor<4xf32>):
    %ni = "vhlo.add_v1"(%i, %n) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %nx = tfl.mul %x, %x {fused_activation_function = "NONE"} : tensor<4xf32>
    "vhlo.return_v1"(%ni, %n, %nx) : (tensor<i32>, tensor<i32>, tensor<4xf32>) -> ()
  }) : (tensor<i32>, tensor<i32>, tensor<4xf32>) -> (tensor<i32>, tensor<i32>, tensor<4xf32>)
  %1 = tfl.mul %0#2, %arg2 {fused_activation_function = "NONE"} : tensor<4xf32>
  return %1 : tensor<4xf32>
}
}
