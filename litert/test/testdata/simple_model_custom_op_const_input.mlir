module {
  func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {
    %cst = "tfl.pseudo_const"() {value = dense<[10.0, 20.0]> : tensor<2xf32>} : () -> tensor<2xf32>
    %out = "tfl.custom"(%x, %cst) {custom_code = "MyCustomOp", custom_option = #tfl<const_bytes: "">} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    func.return %out : tensor<2xf32>
  }
}
