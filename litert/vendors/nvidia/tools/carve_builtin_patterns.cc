// Copyright 2026 Google LLC.
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

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/vendors/nvidia/tools/carve_pattern.h"

namespace litert::tools {

// The patterns the tool knows without a description file, in the format a
// description file has (carve_pattern.h): `carve --print_patterns` writes this
// text, which is the template for a new pattern.
absl::string_view BuiltinCarvePatternText() {
  return R"carve(# odml.factorized_mlp: a feed-forward whose up and down
# projections are each the product of two factors. Two up projections take one
# half of the INT8 input each, a block-diagonal product (one K x N matrix for
# each of G groups) takes the regrouped result to the hidden width, SiLU, and
# the same three steps lead back down.
#
#   inputs   x F16 [1, M, F], then the six weights in graph order: up 1 first
#            half and second half [U, F/2], up 2 [G, K, N], down 1 [G, N, K],
#            down 2 first half and second half [F/2, U] (a leading dimension
#            of 1 is possible on up 2 and down 1). The weights stay constants
#            of the graph, with their quantization.
#   output   y F16 [1, M, F]
#   scale_*  the scale of each INT8 stage (zero point 0), in order: the
#            quantized input, up 1, up 2, the activation, down 1, the output.
pattern odml.factorized_mlp

in  x        : f16 [1, M, F]
in  w_up1a   : const [U, H]
in  w_up1b   : const [U, H]
in  w_up2    : const
in  w_down1  : const
in  w_down2a : const [H, U]
in  w_down2b : const [H, U]
out y

x2 = RESHAPE(x, _)                                                 : f16 [M, F]
q0 = QUANTIZE(x2)                                                  : i8<Sin> [M, F]
h0 = SLICE(q0, begin0, _)                                          : i8<Sin> [M, H]
u0 = FULLY_CONNECTED(h0, w_up1a) {fused_activation_function = "NONE"} : i8<Sup1> [M, U]
r0 = RESHAPE(u0, _)                                                : i8<Sup1> [M, G, 1, C]
h1 = SLICE(q0, begin1, _)                                          : i8<Sin> [M, H]
u1 = FULLY_CONNECTED(h1, w_up1b) {fused_activation_function = "NONE"} : i8<Sup1> [M, U]
r1 = RESHAPE(u1, _)                                                : i8<Sup1> [M, G, 1, C]
c0 = CONCATENATION(r0, r1) {axis = 2, fused_activation_function = "NONE"} : i8<Sup1> [M, G, 2, C]
r2 = RESHAPE(c0, _)                                                : i8<Sup1> [M, G, 1, K]
m0 = BATCH_MATMUL(r2, w_up2) {adj_x = false, adj_y = false}        : i8<Sup2> [M, G, 1, N]
r3 = RESHAPE(m0, _)                                                : i8<Sup2> [M, D]
d0 = DEQUANTIZE(r3)                                                : f16 [M, D]
s0 = LOGISTIC(d0)                                                  : f16 [M, D]
a0 = MUL(d0, s0) {fused_activation_function = "NONE"}              : f16 [M, D]
q1 = QUANTIZE(a0)                                                  : i8<Sact> [M, D]
r4 = RESHAPE(q1, _)                                                : i8<Sact> [M, G, 1, N]
m1 = BATCH_MATMUL(r4, w_down1) {adj_x = false, adj_y = false}      : i8<Sdown1> [M, G, 1, K]
r5 = RESHAPE(m1, _)                                                : i8<Sdown1> [M, G, 2, C]
p0 = SLICE(r5, begin2, _)                                          : i8<Sdown1> [M, G, 1, C]
r6 = RESHAPE(p0, _)                                                : i8<Sdown1> [M, U]
f0 = FULLY_CONNECTED(r6, w_down2a) {fused_activation_function = "NONE"} : i8<Sout> [M, H]
p1 = SLICE(r5, begin3, _)                                          : i8<Sdown1> [M, G, 1, C]
r7 = RESHAPE(p1, _)                                                : i8<Sdown1> [M, U]
f1 = FULLY_CONNECTED(r7, w_down2b) {fused_activation_function = "NONE"} : i8<Sout> [M, H]
c1 = CONCATENATION(f0, f1) {axis = 1, fused_activation_function = "NONE"} : i8<Sout> [M, F]
r8 = RESHAPE(c1, _)                                                : i8<Sout> [1, M, F]
y  = DEQUANTIZE(r8)                                                : f16 [1, M, F]

check F == 2 * H
check K == 2 * C
check U == G * C
check D == G * N
check begin0 == [0, 0]
check begin1 == [0, H]
check begin2 == [0, 0, 0, 0]
check begin3 == [0, 0, 1, 0]

attr version         = 1
attr features        = F
attr blocks          = 2
attr groups          = G
attr up1_features    = U
attr hidden_features = D
attr activation      = "silu"
attr scale_in        = Sin
attr scale_up1       = Sup1
attr scale_up2       = Sup2
attr scale_act       = Sact
attr scale_down1     = Sdown1
attr scale_out       = Sout
)carve";
}

}  // namespace litert::tools
