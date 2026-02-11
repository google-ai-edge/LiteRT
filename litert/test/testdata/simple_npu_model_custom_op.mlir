module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = tfl.mul %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<4xf32>
    %1 = "tfl.custom"(%0, %arg1) {custom_code = "litert_cust", custom_option = #tfl<const_bytes: "">} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    func.return %1 : tensor<4xf32>
  }
}
