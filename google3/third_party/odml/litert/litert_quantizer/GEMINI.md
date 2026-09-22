
## Copybara-friendly code

 * Do not manipulate files with open in google3. Always use gfile instead. No
   exceptions to this rule.
 * Always use gfile.Open() instead of the gfile.GFile() constructor. No
   exceptions to this rule.
 * Do not use os.path.join or gpathlib. Only use pathlib for compatibility
   across GitHub and google3. No exceptions to this rule.
 * Do not use google3.testing.pybase. Always use absl.testing. No exceptions
   to this rule.

## Testing that copybara is working correctly

 * Check for the piper to GitHub copybara transformations with the following
   command:
   ```
   /google/data/ro/teams/copybara/copybara third_party/odml/model_customization/quantization/copy.bara.sky g3folder_to_gitfolder $(g4 g4d)/.. --folder-dir /tmp/aeq_copybara_out --force
   ```

## Testing OSS code
Follow these steps to confirm that all tests pass in the OSS build:

 1. Create a gitfolder from google3 with:
     ```
     /google/data/ro/teams/copybara/copybara third_party/odml/model_customization/quantization/copy.bara.sky g3folder_to_gitfolder $(g4 g4d)/.. --folder-dir /tmp/aeq_copybara_out --force
     ```
 2. Switch to the gitfolder (and run all commands from within that folder):
     ```
     echo '[[index]]
     url = "https://pypi.org/simple"
     default = true' > /tmp/aeq_copybara_out/uv.toml && pushd /tmp/aeq_copybara_out/ && $HOME/.local/bin/uv sync && $HOME/.local/bin/uv run pytest && popd
     ```
