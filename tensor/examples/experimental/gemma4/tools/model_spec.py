# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Architecture shared by published-bundle extraction and Core graph authoring."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    variant: str
    embed_dim: int
    hidden_dim: int
    num_layers: int
    num_kv_heads: int
    num_kv_owners: int
    attention_pattern_size: int
    num_heads: int = 8
    vocab_size: int = 262144
    head_dim: int = 256
    global_key_size: int = 512
    per_layer_input_dim: int = 256
    sliding_window_size: int = 512
    rms_norm_eps: float = 1e-6
    final_logit_softcap: float = 30.0
    local_base_frequency: float = 10000.0
    global_base_frequency: float = 1000000.0
    global_rope_proportion: float = 0.25
    local_rope_proportion: float = 1.0

    def is_global(self, layer):
        return (layer + 1) % self.attention_pattern_size == 0

    def owner(self, layer):
        if layer < self.num_kv_owners:
            return layer
        return max(
            i
            for i in range(self.num_kv_owners)
            if self.is_global(i) == self.is_global(layer)
        )

    @property
    def owners(self):
        return tuple(self.owner(i) for i in range(self.num_layers))

    @property
    def final_prefill_owner(self):
        return max(self.owners)

    def to_dict(self):
        return asdict(self)


SPECS = {
    "e2b": ModelSpec("e2b", 1536, 6144, 35, 1, 15, 5),
    "e4b": ModelSpec("e4b", 2560, 10240, 42, 2, 24, 6),
}


def from_manifest(manifest):
    config = manifest.get("model_config")
    if config is None:
        if manifest.get("schema_version") != 1:
            raise ValueError("Missing model_config")
        return SPECS["e2b"]
    spec = SPECS[config["variant"]]
    if config != spec.to_dict():
        raise ValueError("Unsupported or inconsistent Gemma4 architecture")
    if manifest.get("kv_owners") != list(spec.owners):
        raise ValueError("Incorrect KV owner mapping")
    return spec
