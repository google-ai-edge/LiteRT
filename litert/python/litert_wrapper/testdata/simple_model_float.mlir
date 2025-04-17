// Simple model that performs element-wise addition on two float32 tensors.
//
// Inputs:
//   - arg0: tensor<4xf32> - First input tensor
//   - arg1: tensor<4xf32> - Second input tensor
// Output:
//   - tensor<4xf32> - Element-wise sum of the inputs
//
// Example: [1,2,3,4] + [10,20,30,40] = [11,22,33,44]
module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    // Perform element-wise addition with no activation function
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<4xf32>
    // Return the addition result
    func.return %0 : tensor<4xf32>
  }
}
