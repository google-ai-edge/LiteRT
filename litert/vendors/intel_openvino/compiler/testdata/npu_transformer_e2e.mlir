// Multi-subgraph transformer-style TFLite MLIR fixture for end-to-end testing
// of the Intel OpenVINO NPU compiler pipeline:
//   - @main (prefill, S_q = 4) and @decode (decode, S_q = 2) share four weight
//     constants (%w_q, %w_out, %fallback_bias, and the 16-byte boolean constant
//     %shared_bool_gate), which tf_tfl_translate deduplicates into shared
//     flatbuffer buffers across both subgraphs.
//   - Split-KV-cache attention (K_cache [1,2,8,8], K_new [1,2,4,8], transposed
//     V_cache [1,2,8,8] and V_new [1,2,8,4], unaligned merged S_kv = 12 padded
//     to 16) with reversed V-branch Add (%ctx_new + %ctx_cache) and both
//     tfl.slice (@main) and tfl.strided_slice (@decode).
//   - tfl.reduce_any with both keep_dims = true and keep_dims = false, combined
//     with %shared_bool_gate via tfl.logical_and + tfl.select_v2.
module {
  func.func @main(
      %hidden: tensor<1x2x4x8xf32>,
      %k_cache: tensor<1x2x8x8xf32>,
      %k_new: tensor<1x2x4x8xf32>,
      %v_cache: tensor<1x2x8x8xf32>,
      %v_new: tensor<1x2x8x4xf32>,
      %attn_mask: tensor<1x1x4x12xf32>)
      -> (tensor<1x2x4x8xf32>, tensor<1x2x4x1xi1>, tensor<1x2x4xi1>) {
    %w_q = "tfl.pseudo_const"() <{
      value = dense<[[
        [
          [ 0.25, -0.05,  0.10,  0.00, -0.10,  0.05,  0.15, -0.20],
          [ 0.05,  0.30, -0.10,  0.15,  0.00, -0.05,  0.10,  0.05],
          [-0.10,  0.05,  0.35, -0.05,  0.10,  0.15, -0.05,  0.00],
          [ 0.00, -0.15,  0.05,  0.25, -0.05,  0.10,  0.20, -0.10],
          [ 0.10,  0.00, -0.15,  0.05,  0.30, -0.10,  0.05,  0.15],
          [-0.05,  0.10,  0.00, -0.10,  0.05,  0.35, -0.15,  0.05],
          [ 0.15, -0.05,  0.10,  0.05, -0.15,  0.00,  0.25, -0.05],
          [-0.20,  0.10, -0.05,  0.15,  0.05, -0.10,  0.05,  0.30]
        ],
        [
          [ 0.20,  0.10, -0.05,  0.05, -0.15,  0.00,  0.10, -0.05],
          [-0.05,  0.25,  0.15, -0.10,  0.05,  0.10, -0.05,  0.00],
          [ 0.10, -0.10,  0.30,  0.05, -0.05,  0.15,  0.00, -0.10],
          [ 0.05,  0.15, -0.05,  0.35,  0.10, -0.05,  0.15,  0.05],
          [-0.15,  0.05,  0.10, -0.05,  0.25,  0.05, -0.10,  0.15],
          [ 0.00, -0.05,  0.15,  0.10, -0.10,  0.30,  0.05, -0.05],
          [ 0.10,  0.00, -0.10,  0.15,  0.05, -0.05,  0.20,  0.10],
          [-0.05,  0.10,  0.05, -0.15,  0.10,  0.00, -0.05,  0.25]
        ]
      ]]> : tensor<1x2x8x8xf32>
    }> : () -> tensor<1x2x8x8xf32>

    %w_out = "tfl.pseudo_const"() <{
      value = dense<[[
        [
          [ 0.30,  0.05, -0.10,  0.15,  0.00, -0.05,  0.10, -0.15],
          [-0.10,  0.25,  0.05, -0.05,  0.15,  0.10, -0.10,  0.05],
          [ 0.05, -0.15,  0.35,  0.10, -0.05,  0.00,  0.15, -0.05],
          [ 0.15,  0.05, -0.05,  0.20,  0.10, -0.15,  0.05,  0.10],
          [ 0.00,  0.10,  0.15, -0.10,  0.30,  0.05, -0.05, -0.10],
          [-0.05,  0.15, -0.10,  0.05, -0.05,  0.25,  0.10,  0.15],
          [ 0.10, -0.05,  0.05,  0.15, -0.10,  0.05,  0.35, -0.05],
          [-0.15,  0.00,  0.10, -0.05,  0.15, -0.10,  0.05,  0.20]
        ],
        [
          [ 0.25, -0.10,  0.05,  0.10, -0.05,  0.15,  0.00, -0.10],
          [ 0.10,  0.30, -0.15,  0.05,  0.10, -0.05,  0.15,  0.00],
          [-0.05,  0.15,  0.20, -0.10,  0.05,  0.10, -0.15,  0.05],
          [ 0.05, -0.05,  0.10,  0.35, -0.15,  0.05,  0.10, -0.05],
          [-0.10,  0.10, -0.05,  0.15,  0.25, -0.10,  0.05,  0.15],
          [ 0.15, -0.05,  0.15, -0.05,  0.10,  0.30, -0.05,  0.05],
          [ 0.00,  0.15, -0.10,  0.05, -0.05,  0.05,  0.25, -0.10],
          [-0.10,  0.05,  0.05, -0.10,  0.15,  0.00, -0.10,  0.35]
        ]
      ]]> : tensor<1x2x8x8xf32>
    }> : () -> tensor<1x2x8x8xf32>

    %fallback_bias = "tfl.pseudo_const"() <{
      value = dense<[[
        [[-0.25,  0.15, -0.10,  0.30, -0.20,  0.05, -0.15,  0.25]],
        [[ 0.10, -0.30,  0.20, -0.05,  0.15, -0.25,  0.35, -0.10]]
      ]]> : tensor<1x2x1x8xf32>
    }> : () -> tensor<1x2x1x8xf32>

    %shared_bool_gate = "tfl.pseudo_const"() <{
      value = dense<[[
        [[true, false, true, true, false, true, false, true]],
        [[false, true, true, false, true, false, true, false]]
      ]]> : tensor<1x2x1x8xi1>
    }> : () -> tensor<1x2x1x8xi1>

    %q = "tfl.batch_matmul"(%hidden, %w_q) <{
      adj_x = false, adj_y = false, asymmetric_quantize_inputs = false
    }> : (tensor<1x2x4x8xf32>, tensor<1x2x8x8xf32>) -> tensor<1x2x4x8xf32>

    %qk_cache = "tfl.batch_matmul"(%q, %k_cache) <{
      adj_x = false, adj_y = true, asymmetric_quantize_inputs = false
    }> : (tensor<1x2x4x8xf32>, tensor<1x2x8x8xf32>) -> tensor<1x2x4x8xf32>

    %qk_new = "tfl.batch_matmul"(%q, %k_new) <{
      adj_x = false, adj_y = true, asymmetric_quantize_inputs = false
    }> : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x4x4xf32>

    %scores = "tfl.concatenation"(%qk_cache, %qk_new) <{
      axis = 3 : i32, fused_activation_function = "NONE"
    }> : (tensor<1x2x4x8xf32>, tensor<1x2x4x4xf32>) -> tensor<1x2x4x12xf32>

    %masked_scores = "tfl.add"(%scores, %attn_mask) <{
      fused_activation_function = "NONE"
    }> : (tensor<1x2x4x12xf32>, tensor<1x1x4x12xf32>) -> tensor<1x2x4x12xf32>

    %probs = "tfl.softmax"(%masked_scores) <{
      beta = 1.000000e+00 : f32
    }> : (tensor<1x2x4x12xf32>) -> tensor<1x2x4x12xf32>

    %slice_cache_begin = "tfl.pseudo_const"() <{
      value = dense<[0, 0, 0, 0]> : tensor<4xi32>
    }> : () -> tensor<4xi32>
    %slice_cache_size = "tfl.pseudo_const"() <{
      value = dense<[1, 2, 4, 8]> : tensor<4xi32>
    }> : () -> tensor<4xi32>
    %probs_cache = "tfl.slice"(%probs, %slice_cache_begin, %slice_cache_size)
        : (tensor<1x2x4x12xf32>, tensor<4xi32>, tensor<4xi32>)
        -> tensor<1x2x4x8xf32>

    %slice_new_begin = "tfl.pseudo_const"() <{
      value = dense<[0, 0, 0, 8]> : tensor<4xi32>
    }> : () -> tensor<4xi32>
    %slice_new_size = "tfl.pseudo_const"() <{
      value = dense<[1, 2, 4, 4]> : tensor<4xi32>
    }> : () -> tensor<4xi32>
    %probs_new = "tfl.slice"(%probs, %slice_new_begin, %slice_new_size)
        : (tensor<1x2x4x12xf32>, tensor<4xi32>, tensor<4xi32>)
        -> tensor<1x2x4x4xf32>

    %ctx_cache = "tfl.batch_matmul"(%probs_cache, %v_cache) <{
      adj_x = false, adj_y = true, asymmetric_quantize_inputs = false
    }> : (tensor<1x2x4x8xf32>, tensor<1x2x8x8xf32>) -> tensor<1x2x4x8xf32>

    %ctx_new = "tfl.batch_matmul"(%probs_new, %v_new) <{
      adj_x = false, adj_y = true, asymmetric_quantize_inputs = false
    }> : (tensor<1x2x4x4xf32>, tensor<1x2x8x4xf32>) -> tensor<1x2x4x8xf32>

    // Reversed operand order (%ctx_new first, %ctx_cache second) relative to
    // the score concatenation (%qk_cache, %qk_new) to exercise V-branch order
    // recovery in FuseSplitAttentionToSDPA.
    %attn_out = "tfl.add"(%ctx_new, %ctx_cache) <{
      fused_activation_function = "NONE"
    }> : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32>

    %proj_out = "tfl.batch_matmul"(%attn_out, %w_out) <{
      adj_x = false, adj_y = false, asymmetric_quantize_inputs = false
    }> : (tensor<1x2x4x8xf32>, tensor<1x2x8x8xf32>) -> tensor<1x2x4x8xf32>

    %fallback_out = "tfl.add"(%proj_out, %fallback_bias) <{
      fused_activation_function = "NONE"
    }> : (tensor<1x2x4x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x4x8xf32>

    %zero = "tfl.pseudo_const"() <{
      value = dense<0.000000e+00> : tensor<f32>
    }> : () -> tensor<f32>
    %is_pos = "tfl.greater"(%proj_out, %zero)
        : (tensor<1x2x4x8xf32>, tensor<f32>) -> tensor<1x2x4x8xi1>

    %axis_3 = "tfl.pseudo_const"() <{
      value = dense<3> : tensor<1xi32>
    }> : () -> tensor<1xi32>
    %any_keep = "tfl.reduce_any"(%is_pos, %axis_3) <{keep_dims = true}>
        : (tensor<1x2x4x8xi1>, tensor<1xi32>) -> tensor<1x2x4x1xi1>
    %any_drop = "tfl.reduce_any"(%is_pos, %axis_3) <{keep_dims = false}>
        : (tensor<1x2x4x8xi1>, tensor<1xi32>) -> tensor<1x2x4xi1>

    %gate_cond = "tfl.logical_and"(%any_keep, %shared_bool_gate)
        : (tensor<1x2x4x1xi1>, tensor<1x2x1x8xi1>) -> tensor<1x2x4x8xi1>

    %gated_out = "tfl.select_v2"(%gate_cond, %proj_out, %fallback_out)
        : (tensor<1x2x4x8xi1>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>)
        -> tensor<1x2x4x8xf32>

    return %gated_out, %any_keep, %any_drop
        : tensor<1x2x4x8xf32>, tensor<1x2x4x1xi1>, tensor<1x2x4xi1>
  }

  func.func @decode(
      %hidden: tensor<1x2x2x8xf32>,
      %k_cache: tensor<1x2x8x8xf32>,
      %k_new: tensor<1x2x4x8xf32>,
      %v_cache: tensor<1x2x8x8xf32>,
      %v_new: tensor<1x2x8x4xf32>,
      %attn_mask: tensor<1x1x2x12xf32>)
      -> (tensor<1x2x2x8xf32>, tensor<1x2x2x1xi1>, tensor<1x2x2xi1>) {
    %w_q = "tfl.pseudo_const"() <{
      value = dense<[[
        [
          [ 0.25, -0.05,  0.10,  0.00, -0.10,  0.05,  0.15, -0.20],
          [ 0.05,  0.30, -0.10,  0.15,  0.00, -0.05,  0.10,  0.05],
          [-0.10,  0.05,  0.35, -0.05,  0.10,  0.15, -0.05,  0.00],
          [ 0.00, -0.15,  0.05,  0.25, -0.05,  0.10,  0.20, -0.10],
          [ 0.10,  0.00, -0.15,  0.05,  0.30, -0.10,  0.05,  0.15],
          [-0.05,  0.10,  0.00, -0.10,  0.05,  0.35, -0.15,  0.05],
          [ 0.15, -0.05,  0.10,  0.05, -0.15,  0.00,  0.25, -0.05],
          [-0.20,  0.10, -0.05,  0.15,  0.05, -0.10,  0.05,  0.30]
        ],
        [
          [ 0.20,  0.10, -0.05,  0.05, -0.15,  0.00,  0.10, -0.05],
          [-0.05,  0.25,  0.15, -0.10,  0.05,  0.10, -0.05,  0.00],
          [ 0.10, -0.10,  0.30,  0.05, -0.05,  0.15,  0.00, -0.10],
          [ 0.05,  0.15, -0.05,  0.35,  0.10, -0.05,  0.15,  0.05],
          [-0.15,  0.05,  0.10, -0.05,  0.25,  0.05, -0.10,  0.15],
          [ 0.00, -0.05,  0.15,  0.10, -0.10,  0.30,  0.05, -0.05],
          [ 0.10,  0.00, -0.10,  0.15,  0.05, -0.05,  0.20,  0.10],
          [-0.05,  0.10,  0.05, -0.15,  0.10,  0.00, -0.05,  0.25]
        ]
      ]]> : tensor<1x2x8x8xf32>
    }> : () -> tensor<1x2x8x8xf32>

    %w_out = "tfl.pseudo_const"() <{
      value = dense<[[
        [
          [ 0.30,  0.05, -0.10,  0.15,  0.00, -0.05,  0.10, -0.15],
          [-0.10,  0.25,  0.05, -0.05,  0.15,  0.10, -0.10,  0.05],
          [ 0.05, -0.15,  0.35,  0.10, -0.05,  0.00,  0.15, -0.05],
          [ 0.15,  0.05, -0.05,  0.20,  0.10, -0.15,  0.05,  0.10],
          [ 0.00,  0.10,  0.15, -0.10,  0.30,  0.05, -0.05, -0.10],
          [-0.05,  0.15, -0.10,  0.05, -0.05,  0.25,  0.10,  0.15],
          [ 0.10, -0.05,  0.05,  0.15, -0.10,  0.05,  0.35, -0.05],
          [-0.15,  0.00,  0.10, -0.05,  0.15, -0.10,  0.05,  0.20]
        ],
        [
          [ 0.25, -0.10,  0.05,  0.10, -0.05,  0.15,  0.00, -0.10],
          [ 0.10,  0.30, -0.15,  0.05,  0.10, -0.05,  0.15,  0.00],
          [-0.05,  0.15,  0.20, -0.10,  0.05,  0.10, -0.15,  0.05],
          [ 0.05, -0.05,  0.10,  0.35, -0.15,  0.05,  0.10, -0.05],
          [-0.10,  0.10, -0.05,  0.15,  0.25, -0.10,  0.05,  0.15],
          [ 0.15, -0.05,  0.15, -0.05,  0.10,  0.30, -0.05,  0.05],
          [ 0.00,  0.15, -0.10,  0.05, -0.05,  0.05,  0.25, -0.10],
          [-0.10,  0.05,  0.05, -0.10,  0.15,  0.00, -0.10,  0.35]
        ]
      ]]> : tensor<1x2x8x8xf32>
    }> : () -> tensor<1x2x8x8xf32>

    %fallback_bias = "tfl.pseudo_const"() <{
      value = dense<[[
        [[-0.25,  0.15, -0.10,  0.30, -0.20,  0.05, -0.15,  0.25]],
        [[ 0.10, -0.30,  0.20, -0.05,  0.15, -0.25,  0.35, -0.10]]
      ]]> : tensor<1x2x1x8xf32>
    }> : () -> tensor<1x2x1x8xf32>

    %shared_bool_gate = "tfl.pseudo_const"() <{
      value = dense<[[
        [[true, false, true, true, false, true, false, true]],
        [[false, true, true, false, true, false, true, false]]
      ]]> : tensor<1x2x1x8xi1>
    }> : () -> tensor<1x2x1x8xi1>

    %q = "tfl.batch_matmul"(%hidden, %w_q) <{
      adj_x = false, adj_y = false, asymmetric_quantize_inputs = false
    }> : (tensor<1x2x2x8xf32>, tensor<1x2x8x8xf32>) -> tensor<1x2x2x8xf32>

    %qk_cache = "tfl.batch_matmul"(%q, %k_cache) <{
      adj_x = false, adj_y = true, asymmetric_quantize_inputs = false
    }> : (tensor<1x2x2x8xf32>, tensor<1x2x8x8xf32>) -> tensor<1x2x2x8xf32>

    %qk_new = "tfl.batch_matmul"(%q, %k_new) <{
      adj_x = false, adj_y = true, asymmetric_quantize_inputs = false
    }> : (tensor<1x2x2x8xf32>, tensor<1x2x4x8xf32>) -> tensor<1x2x2x4xf32>

    %scores = "tfl.concatenation"(%qk_cache, %qk_new) <{
      axis = 3 : i32, fused_activation_function = "NONE"
    }> : (tensor<1x2x2x8xf32>, tensor<1x2x2x4xf32>) -> tensor<1x2x2x12xf32>

    %masked_scores = "tfl.add"(%scores, %attn_mask) <{
      fused_activation_function = "NONE"
    }> : (tensor<1x2x2x12xf32>, tensor<1x1x2x12xf32>) -> tensor<1x2x2x12xf32>

    %probs = "tfl.softmax"(%masked_scores) <{
      beta = 1.000000e+00 : f32
    }> : (tensor<1x2x2x12xf32>) -> tensor<1x2x2x12xf32>

    // Split %probs using tfl.strided_slice (including begin_mask = 15 on the
    // cache branch) to exercise the v1::StridedSlice path in
    // LastAxisSliceStart.
    %ss_cache_begin = "tfl.pseudo_const"() <{
      value = dense<[0, 0, 0, 0]> : tensor<4xi32>
    }> : () -> tensor<4xi32>
    %ss_cache_end = "tfl.pseudo_const"() <{
      value = dense<[1, 2, 2, 8]> : tensor<4xi32>
    }> : () -> tensor<4xi32>
    %ss_strides = "tfl.pseudo_const"() <{
      value = dense<[1, 1, 1, 1]> : tensor<4xi32>
    }> : () -> tensor<4xi32>
    %probs_cache = "tfl.strided_slice"(
        %probs, %ss_cache_begin, %ss_cache_end, %ss_strides) <{
      begin_mask = 15 : i32,
      ellipsis_mask = 0 : i32,
      end_mask = 7 : i32,
      new_axis_mask = 0 : i32,
      offset = false,
      shrink_axis_mask = 0 : i32
    }> : (tensor<1x2x2x12xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>)
        -> tensor<1x2x2x8xf32>

    %ss_new_begin = "tfl.pseudo_const"() <{
      value = dense<[0, 0, 0, 8]> : tensor<4xi32>
    }> : () -> tensor<4xi32>
    %ss_new_end = "tfl.pseudo_const"() <{
      value = dense<[1, 2, 2, 12]> : tensor<4xi32>
    }> : () -> tensor<4xi32>
    %probs_new = "tfl.strided_slice"(
        %probs, %ss_new_begin, %ss_new_end, %ss_strides) <{
      begin_mask = 7 : i32,
      ellipsis_mask = 0 : i32,
      end_mask = 15 : i32,
      new_axis_mask = 0 : i32,
      offset = false,
      shrink_axis_mask = 0 : i32
    }> : (tensor<1x2x2x12xf32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>)
        -> tensor<1x2x2x4xf32>

    %ctx_cache = "tfl.batch_matmul"(%probs_cache, %v_cache) <{
      adj_x = false, adj_y = true, asymmetric_quantize_inputs = false
    }> : (tensor<1x2x2x8xf32>, tensor<1x2x8x8xf32>) -> tensor<1x2x2x8xf32>

    %ctx_new = "tfl.batch_matmul"(%probs_new, %v_new) <{
      adj_x = false, adj_y = true, asymmetric_quantize_inputs = false
    }> : (tensor<1x2x2x4xf32>, tensor<1x2x8x4xf32>) -> tensor<1x2x2x8xf32>

    %attn_out = "tfl.add"(%ctx_new, %ctx_cache) <{
      fused_activation_function = "NONE"
    }> : (tensor<1x2x2x8xf32>, tensor<1x2x2x8xf32>) -> tensor<1x2x2x8xf32>

    %proj_out = "tfl.batch_matmul"(%attn_out, %w_out) <{
      adj_x = false, adj_y = false, asymmetric_quantize_inputs = false
    }> : (tensor<1x2x2x8xf32>, tensor<1x2x8x8xf32>) -> tensor<1x2x2x8xf32>

    %fallback_out = "tfl.add"(%proj_out, %fallback_bias) <{
      fused_activation_function = "NONE"
    }> : (tensor<1x2x2x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x2x8xf32>

    %zero = "tfl.pseudo_const"() <{
      value = dense<0.000000e+00> : tensor<f32>
    }> : () -> tensor<f32>
    %is_pos = "tfl.greater"(%proj_out, %zero)
        : (tensor<1x2x2x8xf32>, tensor<f32>) -> tensor<1x2x2x8xi1>

    %axis_3 = "tfl.pseudo_const"() <{
      value = dense<3> : tensor<1xi32>
    }> : () -> tensor<1xi32>
    %any_keep = "tfl.reduce_any"(%is_pos, %axis_3) <{keep_dims = true}>
        : (tensor<1x2x2x8xi1>, tensor<1xi32>) -> tensor<1x2x2x1xi1>
    %any_drop = "tfl.reduce_any"(%is_pos, %axis_3) <{keep_dims = false}>
        : (tensor<1x2x2x8xi1>, tensor<1xi32>) -> tensor<1x2x2xi1>

    %gate_cond = "tfl.logical_and"(%any_keep, %shared_bool_gate)
        : (tensor<1x2x2x1xi1>, tensor<1x2x1x8xi1>) -> tensor<1x2x2x8xi1>

    %gated_out = "tfl.select_v2"(%gate_cond, %proj_out, %fallback_out)
        : (tensor<1x2x2x8xi1>, tensor<1x2x2x8xf32>, tensor<1x2x2x8xf32>)
        -> tensor<1x2x2x8xf32>

    return %gated_out, %any_keep, %any_drop
        : tensor<1x2x2x8xf32>, tensor<1x2x2x1xi1>, tensor<1x2x2xi1>
  }
}
