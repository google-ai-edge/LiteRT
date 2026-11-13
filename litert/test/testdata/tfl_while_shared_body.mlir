// tf_tfl_translate duplicates functions called from several ops, so to keep
// both WHILE ops referencing the same cond/body subgraphs this model is
// generated with:
//   flatbuffer_translate -mlir-to-tflite-flatbuffer --emit-stablehlo-ops=true
module {
func.func @main(%arg0: tensor<i32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0:2 = "tfl.while"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<4xf32>):
    %1 = func.call @while_cond(%arg2, %arg3) : (tensor<i32>, tensor<4xf32>) -> tensor<i1>
    "tfl.yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<4xf32>):
    %1:2 = func.call @while_body(%arg2, %arg3) : (tensor<i32>, tensor<4xf32>) -> (tensor<i32>, tensor<4xf32>)
    "tfl.yield"(%1#0, %1#1) : (tensor<i32>, tensor<4xf32>) -> ()
  }) : (tensor<i32>, tensor<4xf32>) -> (tensor<i32>, tensor<4xf32>)
  %2 = tfl.mul %0#1, %arg1 {fused_activation_function = "NONE"} : tensor<4xf32>
  %3:2 = "tfl.while"(%arg0, %2) ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<4xf32>):
    %1 = func.call @while_cond(%arg2, %arg3) : (tensor<i32>, tensor<4xf32>) -> tensor<i1>
    "tfl.yield"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<4xf32>):
    %1:2 = func.call @while_body(%arg2, %arg3) : (tensor<i32>, tensor<4xf32>) -> (tensor<i32>, tensor<4xf32>)
    "tfl.yield"(%1#0, %1#1) : (tensor<i32>, tensor<4xf32>) -> ()
  }) : (tensor<i32>, tensor<4xf32>) -> (tensor<i32>, tensor<4xf32>)
  return %3#1 : tensor<4xf32>
}

func.func private @while_cond(%arg0: tensor<i32>, %arg1: tensor<4xf32>) -> tensor<i1> {
  %cst = arith.constant dense<0> : tensor<i32>
  %0 = "tfl.greater"(%arg0, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %0 : tensor<i1>
}

func.func private @while_body(%arg0: tensor<i32>, %arg1: tensor<4xf32>) -> (tensor<i32>, tensor<4xf32>) {
  %cst = arith.constant dense<1> : tensor<i32>
  %0 = tfl.sub %arg0, %cst {fused_activation_function = "NONE"} : tensor<i32>
  %1 = tfl.mul %arg1, %arg1 {fused_activation_function = "NONE"} : tensor<4xf32>
  return %0, %1 : tensor<i32>, tensor<4xf32>
}
}
