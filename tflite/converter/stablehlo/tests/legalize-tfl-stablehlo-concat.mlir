// RUN: odml-to-stablehlo-opt %s -tfl-parse-stablehlo-ops | FileCheck %s

module {
func.func @main(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
  %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "stablehlo.concatenate", custom_option = #tfl<const_bytes : "0x64696D656E73696F6E00010B0101010004022401">} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>
  func.return %0 : tensor<6x3xf32>
}
}

// CHECK:       module {
// CHECK-NEXT:  func @main(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
// CHECK-NEXT:    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>
// CHECK-NEXT:    return %0 : tensor<6x3xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  }
