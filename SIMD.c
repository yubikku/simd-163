#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <x86intrin.h>
#include "bmp_utils.h"

// Macro to load 4 bytes (one pixel's worth of channels) and convert to 4 floats
// This is pure C and replaces the previous C++ lambda error.
#define LOAD_RGB_TO_FLOAT(ptr, offset) \
    _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si128((__m128i*)((ptr) + (offset)))))

void convolve_simd(const unsigned char* input, unsigned char* output, int width, int height, float kernel[3][3]) {
    const int channels = 3;
    const int row_stride = width * channels;

    // --- TECHNIQUE: Register Blocking ---
    // Broadcast kernel weights to SIMD registers. This is a key optimization 
    // from Hager & Wellein to reduce memory traffic by keeping the stencil
    // coefficients in CPU registers throughout the entire execution.
    __m128 k00 = _mm_set1_ps(kernel[0][0]); __m128 k01 = _mm_set1_ps(kernel[0][1]); __m128 k02 = _mm_set1_ps(kernel[0][2]);
    __m128 k10 = _mm_set1_ps(kernel[1][0]); __m128 k11 = _mm_set1_ps(kernel[1][1]); __m128 k12 = _mm_set1_ps(kernel[1][2]);
    __m128 k20 = _mm_set1_ps(kernel[2][0]); __m128 k21 = _mm_set1_ps(kernel[2][1]); __m128 k22 = _mm_set1_ps(kernel[2][2]);

    // =========================================================================
    // PART 1: THE CORE (Vectorized, Linear Access)
    // =========================================================================
    // By processing y from 1 to height-1, we avoid boundary checks in the hot loop.
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = (y * width + x) * channels;

            // Row pointers for the 3x3 stencil
            const unsigned char* r0 = &input[idx - row_stride];
            const unsigned char* r1 = &input[idx];
            const unsigned char* r2 = &input[idx + row_stride];

            // --- TECHNIQUE: SIMD Vectorization ---
            // We process R, G, and B simultaneously. 
            // The 4th slot in the SIMD register is used for the R-channel 
            // of the next pixel but is discarded during storage.
            __m128 sum = _mm_setzero_ps();

            // Row 0
            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_RGB_TO_FLOAT(r0, -3), k00));
            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_RGB_TO_FLOAT(r0,  0), k01));
            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_RGB_TO_FLOAT(r0,  3), k02));

            // Row 1
            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_RGB_TO_FLOAT(r1, -3), k10));
            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_RGB_TO_FLOAT(r1,  0), k11));
            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_RGB_TO_FLOAT(r1,  3), k12));

            // Row 2
            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_RGB_TO_FLOAT(r2, -3), k20));
            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_RGB_TO_FLOAT(r2,  0), k21));
            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_RGB_TO_FLOAT(r2,  3), k22));

            // --- TECHNIQUE: SIMD Clamping ---
            // Replaces conditional branches (if statements) with hardware max/min.
            sum = _mm_max_ps(sum, _mm_setzero_ps());
            sum = _mm_min_ps(sum, _mm_set1_ps(255.0f));

            // Convert back to integers and store results
            __m128i res_ints = _mm_cvtps_epi32(sum);
            unsigned int res[4];
            _mm_storeu_si128((__m128i*)res, res_ints);

            output[idx]     = (unsigned char)res[0];
            output[idx + 1] = (unsigned char)res[1];
            output[idx + 2] = (unsigned char)res[2];
        }
    }

    // =========================================================================
    // PART 2: THE BOUNDARY (Loop Peeling for Zero Padding)
    // =========================================================================
    // Zero-padding treats pixels outside the image as 0. 
    // We peel the 1-pixel border to a scalar loop so the SIMD core remains fast.
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // If we are inside the core, skip to the end of the line.
            if (y >= 1 && y < height - 1 && x >= 1 && x < width - 1) {
                x = width - 2;
                continue;
            }

            float s_r = 0, s_g = 0, s_b = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;

                    // Zero-padding logic: Check if neighbor is valid
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        int p_idx = (ny * width + nx) * channels;
                        float kv = kernel[ky + 1][kx + 1];
                        s_r += input[p_idx]     * kv;
                        s_g += input[p_idx + 1] * kv;
                        s_b += input[p_idx + 2] * kv;
                    }
                }
            }
            int out_idx = (y * width + x) * channels;
            output[out_idx]     = clamp(s_r);
            output[out_idx + 1] = clamp(s_g);
            output[out_idx + 2] = clamp(s_b);
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
        convolve_simd(input_img, output_img, info.width, info.height, kernel);
    }
    clock_t end = clock();
    
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    double total_bytes = 2.0 * img_size * iterations;

    double bandwidth_gbs = (total_bytes / 1000000000.0) / time_taken;
    
    printf("Processed '%s' (%dx%d)\n", input_filename, info.width, info.height);
    printf("Time Taken: %f seconds (over %d iterations)\n", time_taken, iterations);
    printf("Effective Bandwidth: %.3f GB/s\n", bandwidth_gbs);

    FILE *f_out = fopen("out_simd.bmp", "wb");
    fwrite(&header, sizeof(BMPHeader), 1, f_out);
    fwrite(&info, sizeof(BMPInfoHeader), 1, f_out);
    fseek(f_out, header.offset, SEEK_SET);
    fwrite(output_img, 1, img_size, f_out);
    
    fclose(f_out); free(input_img); free(output_img);
    return 0;
}