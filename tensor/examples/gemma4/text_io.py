# Copyright 2026 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline Hugging Face tokenization for the C++ example's --token_ids mode.

Google's mobile snapshots contain tokenizer.json (BPE), not tokenizer.model
(SentencePiece). Keep tokenization on the host and run the same IDs on either
Linux or Android. Requires transformers and its tokenizer dependencies.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="Local model snapshot directory; never downloads files.")
    commands = parser.add_subparsers(dest="command", required=True)
    encode = commands.add_parser("encode")
    encode.add_argument("--prompt", required=True)
    encode.add_argument("--chat", action="store_true",
                        help="Apply the checkpoint's user chat template with thinking disabled.")
    decode = commands.add_parser("decode")
    decode.add_argument("--report", type=Path, required=True,
                        help="JSON report emitted by --dump_logits.")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    if args.command == "encode":
        if args.chat:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False,
                return_dict=False,
            )
        else:
            if tokenizer.bos_token_id is None:
                parser.error("Tokenizer has no BOS token for raw completion")
            ids = [tokenizer.bos_token_id] + tokenizer.encode(
                args.prompt, add_special_tokens=False
            )
        print(",".join(map(str, ids)))
    else:
        report = json.loads(args.report.read_text())
        print(tokenizer.decode(report["generated_token_ids"], skip_special_tokens=True))


if __name__ == "__main__":
    main()
