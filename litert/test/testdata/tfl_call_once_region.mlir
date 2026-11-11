// tf_tfl_translate crashes on tfl.call_once, so this model is generated with:
//   flatbuffer_translate -mlir-to-tflite-flatbuffer --emit-stablehlo-ops=true
module {
func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  "tfl.call_once"() {session_init_function = "init_fn"} : () -> ()
  %0 = tfl.mul %arg0, %arg0 {fused_activation_function = "NONE"} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func private @init_fn() {
  return
}
}
