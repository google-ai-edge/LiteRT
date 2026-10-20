// STABLEHLO_REDUCE has no TFLite kernel and is only serialized from VHLO, so
// this model is generated with:
//   flatbuffer_translate -mlir-to-tflite-flatbuffer --emit-stablehlo-ops=true
module {
func.func @main(%arg0: tensor<i32>, %arg1: tensor<4x8xf32>, %arg2: tensor<f32>) -> tensor<4x8xf32> {
  %0:3 = "tfl.while"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<4x8xf32>, %arg5: tensor<f32>):
    %1 = func.call @while_cond(%arg3, %arg4, %arg5) : (tensor<i32>, tensor<4x8xf32>, tensor<f32>) -> tensor<i1>
    "tfl.yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<4x8xf32>, %arg5: tensor<f32>):
    %1:3 = func.call @while_body(%arg3, %arg4, %arg5) : (tensor<i32>, tensor<4x8xf32>, tensor<f32>) -> (tensor<i32>, tensor<4x8xf32>, tensor<f32>)
    "tfl.yield"(%1#0, %1#1, %1#2) : (tensor<i32>, tensor<4x8xf32>, tensor<f32>) -> ()
  }) : (tensor<i32>, tensor<4x8xf32>, tensor<f32>) -> (tensor<i32>, tensor<4x8xf32>, tensor<f32>)
  return %0#1 : tensor<4x8xf32>
}

func.func private @while_cond(%arg0: tensor<i32>, %arg1: tensor<4x8xf32>, %arg2: tensor<f32>) -> tensor<i1> {
  %cst = arith.constant dense<0> : tensor<i32>
  %0 = "tfl.greater"(%arg0, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %0 : tensor<i1>
}

func.func private @while_body(%arg0: tensor<i32>, %arg1: tensor<4x8xf32>, %arg2: tensor<f32>) -> (tensor<i32>, tensor<4x8xf32>, tensor<f32>) {
  %cst = arith.constant dense<1> : tensor<i32>
  %0 = tfl.sub %arg0, %cst {fused_activation_function = "NONE"} : tensor<i32>
  %1 = "vhlo.reduce_v1"(%arg1, %arg2) <{dimensions = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %m = tfl.mul %a, %b {fused_activation_function = "NONE"} : tensor<f32>
    "vhlo.return_v1"(%m) : (tensor<f32>) -> ()
  }) : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  %shape = arith.constant dense<[4, 1]> : tensor<2xi32>
  %2 = "tfl.reshape"(%1, %shape) : (tensor<4xf32>, tensor<2xi32>) -> tensor<4x1xf32>
  %3 = "tfl.mul"(%arg1, %2) {fused_activation_function = "NONE"} : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
  return %0, %3, %arg2 : tensor<i32>, tensor<4x8xf32>, tensor<f32>
}
}
