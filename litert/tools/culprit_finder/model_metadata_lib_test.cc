// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/tools/culprit_finder/model_metadata_lib.h"

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "litert/cc/litert_expected.h"
#include "litert/tools/culprit_finder/interpreter_handler.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/interpreter.h"
#include "tflite/tools/delegates/delegate_provider.h"

namespace litert::tools {
namespace {
static constexpr char kModelPath[] =
    "litert/test/testdata/"
    "mobilenet_v2_1.0_224.tflite";
}  // namespace

class ModelMetadataTest : public ::testing::Test {
 protected:
  std::unique_ptr<InterpreterHandler> interpreter_handler_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<ModelMetadata> model_metadata_;

  void SetUp() override {
    // Create interpreter handler for kModelPath.
    litert::Expected<std::unique_ptr<InterpreterHandler>>
        expected_interpreter_handler = InterpreterHandler::Create(kModelPath);
    ASSERT_TRUE(expected_interpreter_handler);
    interpreter_handler_ = std::move(*expected_interpreter_handler);

    // Create interpreter.
    litert::Expected<std::unique_ptr<tflite::Interpreter>>
        expected_interpreter = interpreter_handler_->PrepareInterpreter(
            tflite::tools::CreateNullDelegate());
    ASSERT_TRUE(expected_interpreter);
    interpreter_ = std::move(*expected_interpreter);

    // Create model metadata class.
    litert::Expected<std::unique_ptr<ModelMetadata>> expected_model_metadata =
        ModelMetadata::Create(interpreter_.get());
    ASSERT_TRUE(expected_model_metadata);
    model_metadata_ = std::move(*expected_model_metadata);
  }
};

TEST_F(ModelMetadataTest, GetNodeIdentifierTest) {
  // Node id 0 is the first convolution node.
  EXPECT_EQ(model_metadata_->GetNodeIdentifier(0), "CONV_2D");
  EXPECT_EQ(model_metadata_->GetNodeIdentifier(0, /*with_index=*/true),
            "[CONV_2D]:0");
  // Node id 33 is the depthwise convolution node.
  EXPECT_EQ(model_metadata_->GetNodeIdentifier(33), "DEPTHWISE_CONV_2D");
  EXPECT_EQ(model_metadata_->GetNodeIdentifier(33, /*with_index=*/true),
            "[DEPTHWISE_CONV_2D]:33");
  // Node id 65 is the softmax node.
  EXPECT_EQ(model_metadata_->GetNodeIdentifier(65), "SOFTMAX");
  EXPECT_EQ(model_metadata_->GetNodeIdentifier(65, /*with_index=*/true),
            "[SOFTMAX]:65");
}

TEST_F(ModelMetadataTest, GetTensorIdentifierTest) {
  // Tensor id 173 is the input tensor.
  EXPECT_EQ(model_metadata_->GetTensorIdentifier(173), "(INPUT)->173");
  // Tensor id 62 is the output tensor of the last SOFTMAX node.
  EXPECT_EQ(model_metadata_->GetTensorIdentifier(62), "([SOFTMAX]:65)->62");
  // Tensor id 100 is the output tensor of the first CONV_2D node.
  EXPECT_EQ(model_metadata_->GetTensorIdentifier(167),
            "([DEPTHWISE_CONV_2D]:33)->167");
}

TEST_F(ModelMetadataTest, GetNodeShapesTest) {
  // Node id 0 is the first convolution node.
  EXPECT_EQ(model_metadata_->GetNodeShapes(0),
            "(FLOAT32[1,224,224,3],FLOAT32[32,3,3,3],FLOAT32[32]) -> "
            "(FLOAT32[1,112,112,32])");
  // Node id 33 is the depthwise convolution node.
  EXPECT_EQ(model_metadata_->GetNodeShapes(33),
            "(FLOAT32[1,14,14,384],FLOAT32[1,3,3,384],FLOAT32[384]) -> "
            "(FLOAT32[1,14,14,384])");
  // Node id 65 is the softmax node.
  EXPECT_EQ(model_metadata_->GetNodeShapes(65),
            "(FLOAT32[1,1001]) -> (FLOAT32[1,1001])");
}

TEST_F(ModelMetadataTest, GetOutputTensorsOfNodeTest) {
  // Node id 0 is the first convolution node.
  EXPECT_EQ(model_metadata_->GetOutputTensorsOfNode(0).size(), 1);
  EXPECT_EQ(model_metadata_->GetOutputTensorsOfNode(0)[0], 54);
  // Node id 33 is the depthwise convolution node.
  EXPECT_EQ(model_metadata_->GetOutputTensorsOfNode(33).size(), 1);
  EXPECT_EQ(model_metadata_->GetOutputTensorsOfNode(33)[0], 167);
  // Node id 65 is the softmax node.
  EXPECT_EQ(model_metadata_->GetOutputTensorsOfNode(65).size(), 1);
  EXPECT_EQ(model_metadata_->GetOutputTensorsOfNode(65)[0], 62);
}

TEST_F(ModelMetadataTest, GetNodeIdsInRangeTest) {
  // Default execution plan size is 66.
  EXPECT_EQ(model_metadata_->GetNodeIdsInRange(0, 65).size(), 66);
  // Execution plan in range [0, 10] is [0, 33].
  std::vector<int> expected_node_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  EXPECT_EQ(model_metadata_->GetNodeIdsInRange(0, 10), expected_node_ids);

  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  // Modify the graph with the xnnpack delegate.
  interpreter_->ModifyGraphWithDelegate(std::move(xnnpack_delegate));
  auto expected_model_metadata = ModelMetadata::Create(interpreter_.get());
  ASSERT_TRUE(expected_model_metadata);
  auto model_metadata_with_delegate = std::move(*expected_model_metadata);

  // Since the model is fully delegated, only the xnnpack node 66 remains.
  EXPECT_EQ(model_metadata_with_delegate->GetNodeIdsInRange(0, 66).size(), 1);
  EXPECT_EQ(model_metadata_with_delegate->GetNodeIdsInRange(0, 66)[0], 66);
  EXPECT_EQ(
      model_metadata_with_delegate->GetNodeIdentifier(66, /*with_index=*/true),
      "[TfLiteXNNPackDelegate]:66");
}

}  // namespace litert::tools
