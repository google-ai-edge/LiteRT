// STABLEHLO_REDUCE has no TFLite kernel and is only serialized from VHLO, so
// this model is generated with:
//   flatbuffer_translate -mlir-to-tflite-flatbuffer --emit-stablehlo-ops=true
module {
func.func @main(%arg0: tensor<4x8xf32>, %arg1: tensor<f32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "vhlo.reduce_v1"(%arg0, %arg1) <{dimensions = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %s = "vhlo.multiply_v1"(%a, %b) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "vhlo.return_v1"(%s) : (tensor<f32>) -> ()
  }) : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  %1 = tfl.mul %0, %arg2 {fused_activation_function = "NONE"} : tensor<4xf32>
  return %1 : tensor<4xf32>
}
}
