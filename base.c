#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "bmp_utils.h"

void convolve_baseline(const unsigned char* input, unsigned char* output, int width, int height, float kernel[3][3]) {
    int channels = 3;

    // use Zero padding for Out-of-bounds handling 
    for (int y = 0; y < height - 1; y++) {
        for (int x = 0; x < width - 1; x++) {
            float sum_r = 0, sum_g = 0, sum_b = 0;

            // Kernel Processing
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int ny = y + ky; // Y coord of neighbor of the pixel's we are on
                    int nx = x + kx; // X coord of neighbor of the pixel's we are on
                    
                    // Check for boundary case
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        int pixel_idx = ((y + ky) * width + (x + kx)) * channels;
                        float k_val = kernel[ky + 1][kx + 1];

                        // Convolution Accumulation
                        sum_r += input[pixel_idx] * k_val;
                        sum_g += input[pixel_idx + 1] * k_val;
                        sum_b += input[pixel_idx + 2] * k_val;
                    }
                }
            }

            // Clamp and Write the calculated sums to the output pixel
            int out_idx = (y * width + x) * channels;
            output[out_idx] = clamp(sum_r);
            output[out_idx + 1] = clamp(sum_g);
            output[out_idx + 2] = clamp(sum_b);
        }
    }
}

int main(int argc, char* argv[]) {
    /* run at cmd as:
        `base input.bmp`
    
    Default to "input.bmp" if no argument is provided */
    
    const char* input_filename = (argc > 1) ? argv[1] : "input.bmp";
    
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

    // Defaults to Laplacian Edge Detection
    // Change Value to desired Kernel Value
    float kernel[3][3] = {{ 0.0f, -1.0f,  0.0f}, 
                        {-1.0f,  5.0f, -1.0f}, 
                        { 0.0f, -1.0f,  0.0f}};

    int iterations = 100; 
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        convolve_baseline(input_img, output_img, info.width, info.height, kernel);
    }
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    double total_bytes = 2.0 * img_size * iterations;

    double bandwidth_gbs = (total_bytes / 1000000000.0) / time_taken;
    
    printf("Processed '%s' (%dx%d)\n", input_filename, info.width, info.height);
    printf("Time Taken: %f seconds (over %d iterations)\n", time_taken, iterations);
    printf("Effective Bandwidth: %.3f GB/s\n", bandwidth_gbs);

    FILE *f_out = fopen("out_baseline.bmp", "wb");
    fwrite(&header, sizeof(BMPHeader), 1, f_out);
    fwrite(&info, sizeof(BMPInfoHeader), 1, f_out);
    fseek(f_out, header.offset, SEEK_SET);
    fwrite(output_img, 1, img_size, f_out);
    
    fclose(f_out); free(input_img); free(output_img);
    return 0;
}