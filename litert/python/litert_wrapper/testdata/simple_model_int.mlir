// Simple model that adds a constant value of 1 to each element of an int32 tensor.
//
// Inputs:
//   - arg0: tensor<4xi32> - Input tensor of 4 int32 values
// Output:
//   - tensor<4xi32> - Result tensor where each element is incremented by 1
//
// This model demonstrates basic tensor operations with integer data types.
// For example, input [5, 10, 15, 20] produces output [6, 11, 16, 21].

module {
  func.func @main(%arg0: tensor<4xi32>) -> tensor<4xi32> {
    %cst = arith.constant dense<1> : tensor<4xi32>
    %0 = tfl.add %arg0, %cst {fused_activation_function = "NONE"} : tensor<4xi32>
    return %0 : tensor<4xi32>
  }
}
