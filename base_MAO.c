#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "bmp_utils.h"

// Define block sizes to fit within L1/L2 cache (Tiling)
// Adjust these based on specific CPU cache architecture
#define BLOCK_Y 32
#define BLOCK_X 32

void convolve_optimized(const unsigned char* input, unsigned char* output, int width, int height, float kernel[3][3]) {
    const int channels = 3;
    const int row_stride = width * channels;

    // --- TECHNIQUE: Register Blocking ---
    // Load kernel values into local scalar variables (registers). 
    // This removes the overhead of 2D array indexing and memory lookups 
    // inside the hot loops (Hager & Wellein, Chapter 3).
    float k00 = kernel[0][0], k01 = kernel[0][1], k02 = kernel[0][2];
    float k10 = kernel[1][0], k11 = kernel[1][1], k12 = kernel[1][2];
    float k20 = kernel[2][0], k21 = kernel[2][1], k22 = kernel[2][2];

    // =========================================================================
    // PART 1: THE CORE (Optimized via Tiling and Unroll-and-Jam)
    // =========================================================================
    // We only process the inner core (1 to height-2, 1 to width-2) here.
    // This allows us to remove all "if" checks for boundaries (Zero-Padding).
    
    // --- TECHNIQUE: Cache Tiling (Blocking) ---
    // Iterating in blocks improves temporal locality by keeping the 'input' 
    // pixels in cache for multiple stencil operations (Hager & Wellein, Chapter 7).
    for (int yb = 1; yb < height - 1; yb += BLOCK_Y) {
        for (int xb = 1; xb < width - 1; xb += BLOCK_X) {
            
            int y_limit = (yb + BLOCK_Y < height - 1) ? yb + BLOCK_Y : height - 1;
            int x_limit = (xb + BLOCK_X < width - 1) ? xb + BLOCK_X : width - 1;

            for (int y = yb; y < y_limit; y++) {
                int x = xb;
                
                // --- TECHNIQUE: Unroll-and-Jam ---
                // We unroll the inner 'x' loop by factor 2 and "jam" the pixel 
                // computations together. This increases ILP and allows the CPU
                // to pipeline memory loads for the next pixel while calculating the current one.
                for (; x <= x_limit - 2; x += 2) {
                    int idx = (y * width + x) * channels;
                    
                    // Strength Reduction: Use pointers for the three relevant rows
                    const unsigned char* r0 = &input[idx - row_stride];
                    const unsigned char* r1 = &input[idx];
                    const unsigned char* r2 = &input[idx + row_stride];

                    // --- Pixel 1 (RGB) ---
                    for (int c = 0; c < 3; c++) {
                        float sum = r0[c-3]*k00 + r0[c]*k01 + r0[c+3]*k02 +
                                    r1[c-3]*k10 + r1[c]*k11 + r1[c+3]*k12 +
                                    r2[c-3]*k20 + r2[c]*k21 + r2[c+3]*k22;
                        output[idx + c] = clamp(sum); // [cite: 40, 41]
                    }

                    // --- Pixel 2 (RGB) ---
                    // Reuses row pointers; indices are offset by 'channels' (3)
                    int idx2 = idx + channels;
                    for (int c = 0; c < 3; c++) {
                        float sum = r0[c]*k00 + r0[c+3]*k01 + r0[c+6]*k02 +
                                    r1[c]*k10 + r1[c+3]*k11 + r1[c+6]*k12 +
                                    r2[c]*k20 + r2[c+3]*k21 + r2[c+6]*k22;
                        output[idx2 + c] = clamp(sum);
                    }
                }

                // Cleanup loop for the remainder of the block
                for (; x < x_limit; x++) {
                    int idx = (y * width + x) * channels;
                    const unsigned char* r0 = &input[idx - row_stride];
                    const unsigned char* r1 = &input[idx];
                    const unsigned char* r2 = &input[idx + row_stride];
                    for (int c = 0; c < 3; c++) {
                        float sum = r0[c-3]*k00 + r0[c]*k01 + r0[c+3]*k02 +
                                    r1[c-3]*k10 + r1[c]*k11 + r1[c+3]*k12 +
                                    r2[c-3]*k20 + r2[c]*k21 + r2[c+3]*k22;
                        output[idx + c] = clamp(sum);
                    }
                }
            }
        }
    }

    // =========================================================================
    // PART 2: THE BOUNDARY (Loop Peeling for Zero-Padding)
    // =========================================================================
    // --- TECHNIQUE: Loop Peeling ---
    // Zero-padding treats out-of-bounds neighbors as 0.
    // We separate the 1-pixel border into its own loop to keep the Core Loop 
    // branch-free and fast.
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Jump over the Core region we already processed
            if (y >= 1 && y < height - 1 && x >= 1 && x < width - 1) {
                x = width - 2; 
                continue;
            }

            float sum_r = 0, sum_g = 0, sum_b = 0;
            // Standard $3\times3$ stencil [cite: 13, 15]
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;

                    // Zero-padding check: Only accumulate if neighbor is valid
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        int p_idx = (ny * width + nx) * channels;
                        float kv = kernel[ky + 1][kx + 1];
                        sum_r += input[p_idx]     * kv; // [cite: 18, 19]
                        sum_g += input[p_idx + 1] * kv;
                        sum_b += input[p_idx + 2] * kv;
                    }
                }
            }
            int out_idx = (y * width + x) * channels;
            output[out_idx]     = clamp(sum_r);
            output[out_idx + 1] = clamp(sum_g);
            output[out_idx + 2] = clamp(sum_b);
        }
    }
}

int main(int argc, char* argv[]) {
    // Default to "output1.bmp" if no argument is provided
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
        convolve_optimized(input_img, output_img, info.width, info.height, kernel);
    }
    clock_t end = clock();
    
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    double total_bytes = 2.0 * img_size * iterations;

    double bandwidth_gbs = (total_bytes / 1000000000.0) / time_taken;
    
    printf("Processed '%s' (%dx%d)\n", input_filename, info.width, info.height);
    printf("Time Taken: %f seconds (over %d iterations)\n", time_taken, iterations);
    printf("Effective Bandwidth: %.3f GB/s\n", bandwidth_gbs);

    FILE *f_out = fopen("out_memory.bmp", "wb");
    fwrite(&header, sizeof(BMPHeader), 1, f_out);
    fwrite(&info, sizeof(BMPInfoHeader), 1, f_out);
    fseek(f_out, header.offset, SEEK_SET);
    fwrite(output_img, 1, img_size, f_out);
    
    fclose(f_out); free(input_img); free(output_img);
    return 0;
}