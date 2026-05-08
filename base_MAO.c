#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "bmp_utils.h"

void convolve_optimized(const unsigned char* input, unsigned char* output, int width, int height, float kernel[3][3]) {
    int channels = 3;
    memcpy(output, input, width * height * channels);
    int block_size = 32;

    for (int y0 = 1; y0 < height - 1; y0 += block_size) {
        for (int x0 = 1; x0 < width - 1; x0 += block_size) {
            int y_end = (y0 + block_size < height - 1) ? y0 + block_size : height - 1;
            int x_end = (x0 + block_size < width - 1) ? x0 + block_size : width - 1;

            for (int y = y0; y < y_end; y++) {
                for (int x = x0; x < x_end; x++) {
                    float sum_r = 0, sum_g = 0, sum_b = 0;

                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            int pixel_idx = ((y + ky) * width + (x + kx)) * channels;
                            float k_val = kernel[ky + 1][kx + 1];

                            sum_r += input[pixel_idx]     * k_val;
                            sum_g += input[pixel_idx + 1] * k_val;
                            sum_b += input[pixel_idx + 2] * k_val;
                        }
                    }

                    int out_idx = (y * width + x) * channels;
                    output[out_idx]     = clamp(sum_r);
                    output[out_idx + 1] = clamp(sum_g);
                    output[out_idx + 2] = clamp(sum_b);
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    // Default to "output1.bmp" if no argument is provided
    const char* input_filename = (argc > 1) ? argv[1] : "output1.bmp";
    
    FILE *f_in = fopen(input_filename, "rb");
    if (!f_in) { 
        printf("Error: Could not open '%s'.\n", input_filename); 
        printf("Usage: %s <input_image.bmp>\n", argv[0]);
        return 1; 
    }

    BMPHeader header; BMPInfoHeader info;
    fread(&header, sizeof(BMPHeader), 1, f_in);
    fread(&info, sizeof(BMPInfoHeader), 1, f_in);

    if (info.bits != 24) { printf("Error: Not 24-bit.\n"); fclose(f_in); return 1; }

    int img_size = info.width * info.height * 3;
    unsigned char* input_img = (unsigned char*)malloc(img_size);
    unsigned char* output_img = (unsigned char*)malloc(img_size);
    
    fseek(f_in, header.offset, SEEK_SET);
    fread(input_img, 1, img_size, f_in);
    fclose(f_in);
    
    float kernel[3][3] = {{ 0.0f, -1.0f,  0.0f}, 
                        {-1.0f,  5.0f, -1.0f}, 
                        { 0.0f, -1.0f,  0.0f}};

    int iterations = 100; 
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        convolve_optimized(input_img, output_img, info.width, info.height, kernel);
    }
    clock_t end = clock();
    
    printf("Memory Optimized Time: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    FILE *f_out = fopen("out_memory.bmp", "wb");
    fwrite(&header, sizeof(BMPHeader), 1, f_out);
    fwrite(&info, sizeof(BMPInfoHeader), 1, f_out);
    fseek(f_out, header.offset, SEEK_SET);
    fwrite(output_img, 1, img_size, f_out);
    
    fclose(f_out); free(input_img); free(output_img);
    return 0;
}