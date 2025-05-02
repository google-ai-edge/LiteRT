#version 310 es
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (binding = 0) uniform sampler2D inputTexture;
layout (binding = 1, rgba8) uniform writeonly highp image2D outputImage;

void main() {
    ivec2 store_pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 out_dims = imageSize(outputImage);

    if (store_pos.x < out_dims.x && store_pos.y < out_dims.y) {
        vec2 tex_coord = vec2(float(store_pos.x) / float(out_dims.x - 1),
                              float(store_pos.y) / float(out_dims.y - 1));
        vec4 color = texture(inputTexture, tex_coord);
        imageStore(outputImage, store_pos, color);
    }
}
