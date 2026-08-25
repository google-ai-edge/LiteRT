module {
  func.func @main(%input: tensor<3x!tf_type.string>) -> tensor<3x!tf_type.string> {
    %normalized = "tfl.custom"(%input) {
      custom_code = "Normalize",
      custom_option = #tfl<const_bytes: "">
    } : (tensor<3x!tf_type.string>) -> tensor<3x!tf_type.string>

    func.return %normalized : tensor<3x!tf_type.string>
  }
}
