module {
func.func @main(%arg0: tensor<2x1x3xf32>, %arg1: tensor<3x2xi32>) -> tensor<? x f32> {
  %0 = "tf.MirrorPad"(%arg0, %arg1) { mode = "SYMMETRIC" }: (tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32>
  func.return %0 : tensor<? x f32>
}
}
