#version 310 es
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Output SSBO - target is RGBA float [0,1] data
layout(std430, binding = 0) buffer OutputBlendBuffer { // Output buffer at binding 0
    vec4 data[];
} output_blend_buffer;

// Input samplers starting from binding 1
layout (binding = 1) uniform sampler2D baseTexture;

// Mask data from SSBOs, starting from binding 2
layout(std430, binding = 2) buffer MaskBuffer0 { float mask_data0[]; };
layout(std430, binding = 3) buffer MaskBuffer1 { float mask_data1[]; };
layout(std430, binding = 4) buffer MaskBuffer2 { float mask_data2[]; };
layout(std430, binding = 5) buffer MaskBuffer3 { float mask_data3[]; };
layout(std430, binding = 6) buffer MaskBuffer4 { float mask_data4[]; };
layout(std430, binding = 7) buffer MaskBuffer5 { float mask_data5[]; };

uniform vec4 maskColor0;
uniform vec4 maskColor1;
uniform vec4 maskColor2;
uniform vec4 maskColor3;
uniform vec4 maskColor4;
uniform vec4 maskColor5;

void main() {
    ivec2 store_pos = ivec2(gl_GlobalInvocationID.xy);

    // Assuming output dimensions match baseTexture dimensions
    ivec2 base_dims = textureSize(baseTexture, 0);

    if (store_pos.x >= base_dims.x || store_pos.y >= base_dims.y) {
        return;
    }

    vec2 tex_coord = vec2(float(store_pos.x) / float(base_dims.x - 1),
                          float(store_pos.y) / float(base_dims.y - 1));

    vec4 base_color = texture(baseTexture, tex_coord);
    vec4 final_color = base_color;

    // Calculate pixel index for SSBOs.
    // Masks are assumed to be 256x256. We need to sample them using tex_coord
    // that corresponds to the original image's coordinate system.
    ivec2 mask_dims = ivec2(256, 256); // Fixed size of the masks
    ivec2 mask_sample_pos = ivec2(tex_coord * vec2(mask_dims - 1)); // Scale tex_coord to mask dimensions
    int mask_pixel_idx = clamp(mask_sample_pos.y, 0, mask_dims.y -1) * mask_dims.x + clamp(mask_sample_pos.x, 0, mask_dims.x -1) ;

    float mask_value0 = mask_data0[mask_pixel_idx];
    if (mask_value0 > 0.5) {
        final_color = mix(final_color, maskColor0, maskColor0.a * mask_value0);
    }

    float mask_value1 = mask_data1[mask_pixel_idx];
    if (mask_value1 > 0.5) {
        final_color = mix(final_color, maskColor1, maskColor1.a * mask_value1);
    }

    float mask_value2 = mask_data2[mask_pixel_idx];
    if (mask_value2 > 0.5) {
        final_color = mix(final_color, maskColor2, maskColor2.a * mask_value2);
    }

    float mask_value3 = mask_data3[mask_pixel_idx];
    if (mask_value3 > 0.5) {
        final_color = mix(final_color, maskColor3, maskColor3.a * mask_value3);
    }

    float mask_value4 = mask_data4[mask_pixel_idx];
    if (mask_value4 > 0.5) {
        final_color = mix(final_color, maskColor4, maskColor4.a * mask_value4);
    }

    float mask_value5 = mask_data5[mask_pixel_idx];
    if (mask_value5 > 0.5) {
        final_color = mix(final_color, maskColor5, maskColor5.a * mask_value5);
    }

    final_color.a = base_color.a;

    int output_pixel_idx = store_pos.y * base_dims.x + store_pos.x;
    output_blend_buffer.data[output_pixel_idx] = final_color;
}