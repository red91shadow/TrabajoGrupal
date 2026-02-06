#include "scan_blelloch.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <immintrin.h>
#include <cstring>


void scan_bl_serial(const int* input, int* output, int size) {
    if (size == 0) return;

  
    int n = size;
    int power2 = 1;
    while (power2 < n) power2 *= 2;
    
    std::vector<int> v(power2, 0);
    std::memcpy(v.data(), input, n * sizeof(int));

    
    for (int stride = 2; stride <= power2; stride *= 2) {
        for (int i = 0; i < power2; i += stride) {
            v[i + stride - 1] += v[i + (stride / 2) - 1];
        }
    }

    
    
    int total_sum = v[power2 - 1];
    v[power2 - 1] = 0; 

    for (int stride = power2; stride >= 2; stride /= 2) {
        for (int i = 0; i < power2; i += stride) {
            int left = i + (stride / 2) - 1;
            int right = i + stride - 1;
            
            int temp = v[left];
            v[left] = v[right];         
            v[right] += temp;           
        }
    }

    
    for (int i = 0; i < size; ++i) {
        output[i] = v[i] + input[i];
    }
}


void scan_bl_simd(const int* input, int* output, int size) {
    if (size == 0) return;

    
    int n = size;
    int power2 = 1;
    while (power2 < n) power2 *= 2;
    std::vector<int> v(power2, 0);
    std::memcpy(v.data(), input, n * sizeof(int));

    for (int stride = 2; stride <= power2; stride *= 2) {
       
        for (int i = 0; i < power2; i += stride) {
            v[i + stride - 1] += v[i + (stride / 2) - 1];
        }
    }

    v[power2 - 1] = 0;
    for (int stride = power2; stride >= 2; stride /= 2) {
        for (int i = 0; i < power2; i += stride) {
            int left = i + (stride / 2) - 1;
            int right = i + stride - 1;
            int temp = v[left];
            v[left] = v[right];
            v[right] += temp;
        }
    }

    
    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256i vec_v = _mm256_loadu_si256((const __m256i*)&v[i]);
        __m256i vec_in = _mm256_loadu_si256((const __m256i*)&input[i]);
        __m256i vec_sum = _mm256_add_epi32(vec_v, vec_in);
        _mm256_storeu_si256((__m256i*)&output[i], vec_sum);
    }
    for (; i < size; ++i) output[i] = v[i] + input[i];
}

void scan_bl_omp(const int* input, int* output, int size) {
    if (size == 0) return;

    int power2 = 1;
    while (power2 < size) power2 *= 2;
    
   
    std::vector<int> v(power2, 0);
    
    
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) v[i] = input[i];

   
    for (int stride = 2; stride <= power2; stride *= 2) {
        #pragma omp parallel for
        for (int i = 0; i < power2; i += stride) {
            v[i + stride - 1] += v[i + (stride / 2) - 1];
        }
    }

    v[power2 - 1] = 0; 

    for (int stride = power2; stride >= 2; stride /= 2) {
        #pragma omp parallel for
        for (int i = 0; i < power2; i += stride) {
            int left = i + (stride / 2) - 1;
            int right = i + stride - 1;
            
            int temp = v[left];
            v[left] = v[right];
            v[right] += temp;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        output[i] = v[i] + input[i];
    }
}


void scan_bl_omp_simd(const int* input, int* output, int size) {
    if (size == 0) return;

    int power2 = 1;
    while (power2 < size) power2 *= 2;
    std::vector<int> v(power2, 0);

    #pragma omp parallel for simd
    for (int i = 0; i < size; ++i) v[i] = input[i];

    
    for (int stride = 2; stride <= power2; stride *= 2) {
        
        #pragma omp parallel for
        for (int i = 0; i < power2; i += stride) {
            v[i + stride - 1] += v[i + (stride / 2) - 1];
        }
    }

    
    v[power2 - 1] = 0;
    for (int stride = power2; stride >= 2; stride /= 2) {
        #pragma omp parallel for
        for (int i = 0; i < power2; i += stride) {
            int left = i + (stride / 2) - 1;
            int right = i + stride - 1;
            int temp = v[left];
            v[left] = v[right];
            v[right] += temp;
        }
    }

    
    #pragma omp parallel for simd
    for (int i = 0; i < size; ++i) {
        output[i] = v[i] + input[i];
    }
}