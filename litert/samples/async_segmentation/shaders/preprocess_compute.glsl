#version 310 es
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Input texture (sampler) - typically RGBA8 from loaded image
layout (binding = 0) uniform sampler2D inputTexture;

// Output SSBO - target is RGB float data
// Each pixel will take 3 float values.
layout(std430, binding = 1) buffer PreprocessedOutput {
    float data[]; // Stores R0, G0, B0, 0.0, R1, G1, B1, 0.0, ...
} preprocessed_output;

// Number of channels in the input SSBO.
uniform int num_channels;

void main() {
    ivec2 store_pos_2d = ivec2(gl_GlobalInvocationID.xy); // x, y position in the 2D output grid
    const int out_width = 256;
    const int out_height = 256;


    if (store_pos_2d.x < out_width && store_pos_2d.y < out_height) {
      vec2 sample_coord_norm = vec2(float(store_pos_2d.x) / float(out_width - 1),
                                    float(store_pos_2d.y) / float(out_height - 1));

      vec4 color_0_1 = texture(inputTexture, sample_coord_norm);

      vec3 color_neg1_1 = (color_0_1.rgb * 2.0) - 1.0;

      int base_index = (store_pos_2d.y * out_width + store_pos_2d.x) * num_channels;
      preprocessed_output.data[base_index + 0] = color_neg1_1.r;
      preprocessed_output.data[base_index + 1] = color_neg1_1.g;
      preprocessed_output.data[base_index + 2] = color_neg1_1.b;
      if(num_channels == 4){
        // If preprocess requires 4 channels (i.e. for GL-CL interop),
        // we set the last channel to 0.0.
        preprocessed_output.data[base_index + 3] = 0.0;
      }
    }
}
