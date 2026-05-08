#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <x86intrin.h>
#include "bmp_utils.h"

void convolve_simd(const unsigned char* input, unsigned char* output, int width, int height, float kernel[3][3]) {
    int channels = 3;
    memcpy(output, input, width * height * channels);

    for (int y = 1; y < height - 1; y++) {
        int x_start = 1;
        int x_end = width - 1;
        int flat_start = (y * width + x_start) * channels;
        int flat_end = (y * width + x_end) * channels;

        int i;
        for (i = flat_start; i <= flat_end - 4; i += 4) {
            __m128 sum_vec = _mm_setzero_ps();

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int offset = (ky * width + kx) * channels;
                    
                    int packed_chars = *(int*)&input[i + offset];
                    __m128i in_chars = _mm_cvtsi32_si128(packed_chars); 
                    __m128i in_ints = _mm_cvtepu8_epi32(in_chars);
                    __m128 in_floats = _mm_cvtepi32_ps(in_ints);

                    __m128 k_vec = _mm_set1_ps(kernel[ky+1][kx+1]);
                    sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(in_floats, k_vec));
                }
            }

            __m128i out_ints = _mm_cvtps_epi32(sum_vec);
            __m128i packed16 = _mm_packs_epi32(out_ints, out_ints);
            __m128i packed8 = _mm_packus_epi16(packed16, packed16);
            *(int*)&output[i] = _mm_cvtsi128_si32(packed8);
        }

        for (; i < flat_end; i++) {
            float sum = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int offset = (ky * width + kx) * channels;
                    sum += input[i + offset] * kernel[ky+1][kx+1];
                }
            }
            output[i] = clamp(sum);
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
    
    printf("SIMD Optimized Time: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    FILE *f_out = fopen("out_simd.bmp", "wb");
    fwrite(&header, sizeof(BMPHeader), 1, f_out);
    fwrite(&info, sizeof(BMPInfoHeader), 1, f_out);
    fseek(f_out, header.offset, SEEK_SET);
    fwrite(output_img, 1, img_size, f_out);
    
    fclose(f_out); free(input_img); free(output_img);
    return 0;
}