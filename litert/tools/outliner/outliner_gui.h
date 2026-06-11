// Copyright 2024 Google LLC.
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

#ifndef ODML_LITERT_LITERT_TOOLS_OUTLINER_OUTLINER_GUI_H_
#define ODML_LITERT_LITERT_TOOLS_OUTLINER_OUTLINER_GUI_H_

#include <cstddef>
#include <string>
#include <vector>

#include "litert/cc/litert_expected.h"
#include "litert/core/model/model.h"
#include "litert/tools/outliner/outliner_util.h"

namespace litert::tools {

// Handles the interactive terminal GUI for the LiteRT Graph Outliner.
class OutlinerGui {
 public:
  explicit OutlinerGui(LiteRtModelT& model);

  // Runs the interactive loop. Returns OK if a subgraph was successfully
  // outlined.
  litert::Expected<void> Run();

  // Get the resulting options if the user confirmed an outlining.
  const OutlinerOptions& GetFinalOptions() const { return final_options_; }
  size_t GetFinalSubgraphIndex() const { return final_subgraph_index_; }
  bool WasConfirmed() const { return confirmed_; }

 private:
  void Render();
  void HandleInput();
  void UpdateSelection();

  // Terminal utilities
  void ClearScreen();
  void MoveCursor(int row, int col);
  void SetColor(int color_code);
  void ResetColor();

  LiteRtModelT& model_;
  OutlinerOptions final_options_;
  size_t final_subgraph_index_ = 0;
  bool confirmed_ = false;
  bool running_ = true;

  // GUI state
  size_t current_sg_idx_ = 0;
  int scroll_offset_ = 0;
  int cursor_pos_ = 0;
  std::vector<int> identified_op_indices_;
  std::string error_message_;
  std::string status_message_;

  // Search state
  bool search_mode_ = false;
  std::string search_query_;
};

}  // namespace litert::tools

#endif  // ODML_LITERT_LITERT_TOOLS_OUTLINER_OUTLINER_GUI_H_
