module {
  func.func public @main(%arg0: tensor<1x8x8x128xf32>, %arg1: tensor<1x8x8x128xi32>) -> tensor<1x8x8x128xf32> {
    %0 = "tfl.custom"(%arg0, %arg1) {custom_option = #tfl<const_bytes : "0x01000000020000000200000002000000020000000000000000000000000000000000000000000000">, custom_code = "LiteRtMaxUnpooling2D"} : (tensor<1x8x8x128xf32>, tensor<1x8x8x128xi32>) -> (tensor<1x8x8x128xf32>)
    func.return %0 : tensor<1x8x8x128xf32>
  }
}
