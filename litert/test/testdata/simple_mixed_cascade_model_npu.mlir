// A mixed cascade of NPU and CPU ops.
module {
  func.func @main(%x1: tensor<2xf32>, %x2: tensor<2xf32>, %x3: tensor<2xf32>, %x4: tensor<2xf32>, %x5: tensor<2xf32>) -> tensor<2xf32> {
    %t1 = "tfl.custom"(%x1, %x2) {custom_code = "DISPATCH_OP_1", custom_option = #tfl<const_bytes: "">} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %t2 = "tfl.custom"(%t1, %x3) {custom_code = "DISPATCH_OP_2", custom_option = #tfl<const_bytes: "">} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %t3 = tfl.add %t2, %x4 {fused_activation_function = "NONE"} : tensor<2xf32> 
    %t4 = "tfl.custom"(%t3, %x5) {custom_code = "DISPATCH_OP_3", custom_option = #tfl<const_bytes: "">} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    func.return %t4 : tensor<2xf32>
  }
}
