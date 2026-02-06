#include "histogram.h"
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>

inline int get_bin_index(int value, int min_val, double bin_width, int num_bins)
{
    int idx = (int)((value - min_val) / bin_width);
    if (idx >= num_bins)
        idx = num_bins - 1;
    if (idx < 0)
        idx = 0;
    return idx;
}

void histogram_serial(const int *input, int size, int *bins, int num_bins, int min_val, int max_val)
{
    std::fill(bins, bins + num_bins, 0);

    int range = max_val - min_val + 1;
    double bin_width = (double)range / num_bins;

    for (int i = 0; i < size; ++i)
    {
        int idx = get_bin_index(input[i], min_val, bin_width, num_bins);
        bins[idx]++;
    }
}

void histogram_simd(const int *input, int size, int *bins, int num_bins, int min_val, int max_val)
{
    std::fill(bins, bins + num_bins, 0);

    int range = max_val - min_val + 1;
    double bin_width = (double)range / num_bins;

    int i = 0;

    __m256i v_min = _mm256_set1_epi32(min_val);

    for (; i <= size - 8; i += 8)
    {
        __m256i v_data = _mm256_loadu_si256((const __m256i *)&input[i]);

        __m256i v_sub = _mm256_sub_epi32(v_data, v_min);

        int temp[8];
        _mm256_storeu_si256((__m256i *)temp, v_sub);

        for (int k = 0; k < 8; ++k)
        {
            int idx = (int)(temp[k] / bin_width);
            if (idx >= num_bins)
                idx = num_bins - 1;
            else if (idx < 0)
                idx = 0;

            bins[idx]++;
        }
    }

    for (; i < size; ++i)
    {
        int idx = get_bin_index(input[i], min_val, bin_width, num_bins);
        bins[idx]++;
    }
}


void histogram_omp(const int *input, int size, int *bins, int num_bins, int min_val, int max_val)
{
    std::fill(bins, bins + num_bins, 0);

    int range = max_val - min_val + 1;
    double bin_width = (double)range / num_bins;

    const int CACHE_LINE = 64;
    const int PADDED_BINS = ((num_bins * sizeof(int) + CACHE_LINE - 1) / CACHE_LINE) * CACHE_LINE / sizeof(int);

    int max_threads = omp_get_max_threads();
    std::vector<int> local_bins_flat(max_threads * PADDED_BINS, 0);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int *my_bins = &local_bins_flat[tid * PADDED_BINS];

#pragma omp for schedule(static)
        for (int i = 0; i < size; ++i)
        {
            int idx = get_bin_index(input[i], min_val, bin_width, num_bins);
            my_bins[idx]++;
        }
    }

    for (int t = 0; t < max_threads; ++t)
    {
        int *thread_bins = &local_bins_flat[t * PADDED_BINS];
        for (int b = 0; b < num_bins; ++b)
        {
            bins[b] += thread_bins[b];
        }
    }
}


void histogram_omp_simd(const int *input, int size, int *bins, int num_bins, int min_val, int max_val)
{
    std::fill(bins, bins + num_bins, 0);

    int range = max_val - min_val + 1;
    double bin_width = (double)range / num_bins;

    const int CACHE_LINE = 64;
    const int PADDED_BINS = ((num_bins * sizeof(int) + CACHE_LINE - 1) / CACHE_LINE) * CACHE_LINE / sizeof(int);

    int max_threads = omp_get_max_threads();
    std::vector<int> local_bins_flat(max_threads * PADDED_BINS, 0);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int *my_bins = &local_bins_flat[tid * PADDED_BINS];

        
        int chunk_size = (size + nthreads - 1) / nthreads;
        

        int start = tid * chunk_size;
        int end = std::min(start + chunk_size, size);

       
        __m256i v_min = _mm256_set1_epi32(min_val);

        if (start < end)
        {
            int i = start;
            for (; i <= end - 8; i += 8)
            {
                __m256i v_data = _mm256_loadu_si256((const __m256i *)&input[i]);
                __m256i v_sub = _mm256_sub_epi32(v_data, v_min);

                int temp[8];
                _mm256_storeu_si256((__m256i *)temp, v_sub);

                for (int k = 0; k < 8; ++k)
                {
                    int idx = (int)(temp[k] / bin_width);
                    if (idx >= num_bins)
                        idx = num_bins - 1;
                    else if (idx < 0)
                        idx = 0;
                    my_bins[idx]++;
                }
            }

            for (; i < end; ++i)
            {
                int idx = get_bin_index(input[i], min_val, bin_width, num_bins);
                my_bins[idx]++;
            }
        }
    }

    for (int t = 0; t < max_threads; ++t)
    {
        int *thread_bins = &local_bins_flat[t * PADDED_BINS];
        for (int b = 0; b < num_bins; ++b)
        {
            bins[b] += thread_bins[b];
        }
    }
}