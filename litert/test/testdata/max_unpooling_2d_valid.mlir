module {
  func.func public @main(%arg0: tensor<2x2x4x2xf32>, %arg1: tensor<2x2x4x2xi32>) -> tensor<2x4x11x2xf32> {
    %0 = "tfl.custom"(%arg0, %arg1) {custom_option = #tfl<const_bytes : "0x02000000030000000200000002000000020000000000000000000000000000000000000000000000">, custom_code = "LiteRtMaxUnpooling2D"} : (tensor<2x2x4x2xf32>, tensor<2x2x4x2xi32>) -> (tensor<2x4x11x2xf32>)
    func.return %0 : tensor<2x4x11x2xf32>
  }
}
