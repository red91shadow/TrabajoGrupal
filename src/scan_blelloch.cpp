#include "scan_blelloch.h"
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream> 


void scan_bl_serial(const int* input, int* output, int size) {
    if (size == 0) return;
    int accum = 0;
    for (int i = 0; i < size; ++i) {
        accum += input[i];
        output[i] = accum;
    }
}


void scan_bl_simd(const int* input, int* output, int size) {
    if (size == 0) return;

    
    const int BLOCK_SIZE = 1024; 
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<int> block_sums(num_blocks, 0);

  
    for (int b = 0; b < num_blocks; ++b) {
        int start = b * BLOCK_SIZE;
        int end = std::min(start + BLOCK_SIZE, size);
        
        int accum = 0;
        
        
        #pragma omp simd reduction(inscan, +:accum)
        for (int i = start; i < end; ++i) {
            accum += input[i];
            #pragma omp scan inclusive(accum)
            output[i] = accum;
        }
        
        
        if (end > start) {
            block_sums[b] = output[end - 1]; 
        }
    }

    
    for (int b = 1; b < num_blocks; ++b) {
        block_sums[b] += block_sums[b - 1];
    }

    for (int b = 1; b < num_blocks; ++b) {
        int start = b * BLOCK_SIZE;
        int end = std::min(start + BLOCK_SIZE, size);
        int offset = block_sums[b - 1];
        
        
        #pragma omp simd
        for (int i = start; i < end; ++i) {
            output[i] += offset;
        }
    }
}


void scan_bl_omp(const int* input, int* output, int size) {
    if (size == 0) return;
    int num_threads = omp_get_max_threads();
    std::vector<int> thread_sums(num_threads + 1, 0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        
        int chunk = (size + nthreads - 1) / nthreads;
        int start = tid * chunk;
        int end = std::min(start + chunk, size);
        
       
        int accum = 0;
        if (start < end) {
            for (int i = start; i < end; ++i) {
                accum += input[i];
                output[i] = accum;
            }
            thread_sums[tid + 1] = accum;
        }
        
        #pragma omp barrier

       
        #pragma omp single
        {
            for (int i = 1; i <= nthreads; ++i) {
                thread_sums[i] += thread_sums[i - 1];
            }
        } 

        // Sumar Offset Global
        int offset = thread_sums[tid];
        if (offset > 0 && start < end) {
            for (int i = start; i < end; ++i) {
                output[i] += offset;
            }
        }
    }
}


void scan_bl_omp_simd(const int* input, int* output, int size) {
    if (size == 0) return;
    int num_threads = omp_get_max_threads();
    std::vector<int> thread_sums(num_threads + 1, 0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int chunk = (size + nthreads - 1) / nthreads;
        int start = tid * chunk;
        int end = std::min(start + chunk, size);
        
        // 1. Scan Local VECTORIZADO
        int accum = 0;
        if (start < end) {
            // Usamos OMP SIMD para vectorizar el scan local
            #pragma omp simd reduction(inscan, +:accum)
            for (int i = start; i < end; ++i) {
                accum += input[i];
                #pragma omp scan inclusive(accum)
                output[i] = accum;
            }
            thread_sums[tid + 1] = accum;
        }

        #pragma omp barrier

        // 2. Scan de Offsets
        #pragma omp single
        {
            for (int i = 1; i <= nthreads; ++i) {
                thread_sums[i] += thread_sums[i - 1];
            }
        }

        // 3. Sumar Offset Global VECTORIZADO
        int offset = thread_sums[tid];
        if (offset > 0 && start < end) {
            #pragma omp simd
            for (int i = start; i < end; ++i) {
                output[i] += offset;
            }
        }
    }
}