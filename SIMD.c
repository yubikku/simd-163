#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <x86intrin.h>
#include "bmp_utils.h"

// void convolve_simd(const unsigned char* input, unsigned char* output, int width, int height, float kernel[3][3]) {
//     int channels = 3;
//     memcpy(output, input, width * height * channels);

//     for (int y = 1; y < height - 1; y++) {
//         int x_start = 1;
//         int x_end = width - 1;
//         int flat_start = (y * width + x_start) * channels;
//         int flat_end = (y * width + x_end) * channels;

//         int i;
//         for (i = flat_start; i <= flat_end - 4; i += 4) {
//             __m128 sum_vec = _mm_setzero_ps();

//             for (int ky = -1; ky <= 1; ky++) {
//                 for (int kx = -1; kx <= 1; kx++) {
//                     int offset = (ky * width + kx) * channels;
                    
//                     int packed_chars = *(int*)&input[i + offset];
//                     __m128i in_chars = _mm_cvtsi32_si128(packed_chars); 
//                     __m128i in_ints = _mm_cvtepu8_epi32(in_chars);
//                     __m128 in_floats = _mm_cvtepi32_ps(in_ints);

//                     __m128 k_vec = _mm_set1_ps(kernel[ky+1][kx+1]);
//                     sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(in_floats, k_vec));
//                 }
//             }

//             __m128i out_ints = _mm_cvtps_epi32(sum_vec);
//             __m128i packed16 = _mm_packs_epi32(out_ints, out_ints);
//             __m128i packed8 = _mm_packus_epi16(packed16, packed16);
//             *(int*)&output[i] = _mm_cvtsi128_si32(packed8);
//         }

//         for (; i < flat_end; i++) {
//             float sum = 0;
//             for (int ky = -1; ky <= 1; ky++) {
//                 for (int kx = -1; kx <= 1; kx++) {
//                     int offset = (ky * width + kx) * channels;
//                     sum += input[i + offset] * kernel[ky+1][kx+1];
//                 }
//             }
//             output[i] = clamp(sum);
//         }
//     }
// }

// Macro to handle the load and conversion from 8-bit unsigned char to 32-bit float
// This avoids function call overhead and is standard-C compatible.
#define LOAD_AND_CONVERT(ptr, offset) \
    _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si128((__m128i*)((ptr) + (offset)))))

void convolve_simd(const unsigned char* input, unsigned char* output, int width, int height, float kernel[3][3]) {
    const int channels = 3;
    const int row_stride = width * channels;

    // --- TECHNIQUE: Register Blocking (SIMD) ---
    // Broadcast kernel coefficients into 128-bit registers[cite: 57].
    __m128 k00 = _mm_set1_ps(kernel[0][0]); __m128 k01 = _mm_set1_ps(kernel[0][1]); __m128 k02 = _mm_set1_ps(kernel[0][2]);
    __m128 k10 = _mm_set1_ps(kernel[1][0]); __m128 k11 = _mm_set1_ps(kernel[1][1]); __m128 k12 = _mm_set1_ps(kernel[1][2]);
    __m128 k20 = _mm_set1_ps(kernel[2][0]); __m128 k21 = _mm_set1_ps(kernel[2][1]); __m128 k22 = _mm_set1_ps(kernel[2][2]);

    // =========================================================================
    // PART 1: THE VECTORIZED CORE (Safe Inner Region)
    // =========================================================================
    // We process from y=1 to height-2 and x=1 to width-2 to avoid border checks.
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) { 
            int idx = (y * width + x) * channels;

            const unsigned char* in_r0 = &input[idx - row_stride];
            const unsigned char* in_r1 = &input[idx];
            const unsigned char* in_r2 = &input[idx + row_stride];

            __m128 sum = _mm_setzero_ps();
            
            // Apply 3x3 kernel using SIMD[cite: 58].
            // Each LOAD_AND_CONVERT handles R, G, B, and a 4th dummy/next-R channel.
            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_AND_CONVERT(in_r0, -3), k00));
            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_AND_CONVERT(in_r0,  0), k01));
            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_AND_CONVERT(in_r0,  3), k02));

            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_AND_CONVERT(in_r1, -3), k10));
            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_AND_CONVERT(in_r1,  0), k11));
            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_AND_CONVERT(in_r1,  3), k12));

            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_AND_CONVERT(in_r2, -3), k20));
            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_AND_CONVERT(in_r2,  0), k21));
            sum = _mm_add_ps(sum, _mm_mul_ps(LOAD_AND_CONVERT(in_r2,  3), k22));

            // --- TECHNIQUE: SIMD Clamping ---
            // Clamps all 4 channels in the register to [0.0, 255.0][cite: 41].
            sum = _mm_max_ps(sum, _mm_setzero_ps());
            sum = _mm_min_ps(sum, _mm_set1_ps(255.0f));

            // Convert floats back to integers and extract the first 3 (RGB)
            __m128i final_ints = _mm_cvtps_epi32(sum);
            unsigned int res[4];
            _mm_storeu_si128((__m128i*)res, final_ints);

            output[idx]     = (unsigned char)res[0];
            output[idx + 1] = (unsigned char)res[1];
            output[idx + 2] = (unsigned char)res[2];
        }
    }

    // =========================================================================
    // PART 2: THE BOUNDARY (Loop Peeling for Zero-Padding)
    // =========================================================================
    // This handles the edges separately to satisfy the zero-padding requirement[cite: 24, 63].
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (y >= 1 && y < height - 1 && x >= 1 && x < width - 1) {
                x = width - 2; // Skip the inner core
                continue;
            }

            float s_r = 0, s_g = 0, s_b = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;

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