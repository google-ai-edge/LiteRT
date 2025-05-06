#version 310 es
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

uniform int mask_width;
uniform int mask_height;

// Input SSBO
layout(std430, binding = 0) buffer InputBuffer {
  float data[];
} input_ssbo;
layout(std430, binding = 1) buffer MaskBuffer0 {
  float data[];
} mask0;
layout(std430, binding = 2) buffer MaskBuffer1 {
  float data[];
} mask1;
layout(std430, binding = 3) buffer MaskBuffer2 {
  float data[];
} mask2;
layout(std430, binding = 4) buffer MaskBuffer3 {
  float data[];
} mask3;
layout(std430, binding = 5) buffer MaskBuffer4 {
  float data[];
} mask4;
layout(std430, binding = 6) buffer MaskBuffer5 {
  float data[];
} mask5;

void main() {
  uint gx = gl_GlobalInvocationID.x;
  uint gy = gl_GlobalInvocationID.y;

  if(gx >= uint(mask_width) || gy >= uint(mask_height)){
    return;
  }

  uint output_1d_index = gy * uint(mask_width) + gx;

  uint input_1d_index_masks0to3 = (gy * uint(mask_width) * 4u) + (gx * 4u);

  if(input_1d_index_masks0to3 + 3u < uint(input_ssbo.data.length())) {
    mask0.data[output_1d_index] = input_ssbo.data[input_1d_index_masks0to3 + 0u];
    mask1.data[output_1d_index] = input_ssbo.data[input_1d_index_masks0to3 + 1u];
    mask2.data[output_1d_index] = input_ssbo.data[input_1d_index_masks0to3 + 2u];
    mask3.data[output_1d_index] = input_ssbo.data[input_1d_index_masks0to3 + 3u];
  }

  uint offset_for_masks4and5 = uint(mask_width) * uint(mask_height) * 4u;
  uint base_index_current_pixel_masks4and5 = offset_for_masks4and5 + (output_1d_index * 4u);

  if (base_index_current_pixel_masks4and5 + 1u < uint(input_ssbo.data.length())) {
    mask4.data[output_1d_index] = input_ssbo.data[base_index_current_pixel_masks4and5 + 0u];
    mask5.data[output_1d_index] = input_ssbo.data[base_index_current_pixel_masks4and5 + 1u];
  }
}