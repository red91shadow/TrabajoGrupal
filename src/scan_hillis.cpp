#include "scan_hillis.h"
#include <omp.h>
#include <immintrin.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>


void scan_hs_serial(const int *input, int *output, int size)
{
    if (size == 0)
        return;
    output[0] = input[0];
    for (int i = 1; i < size; ++i)
    {
        output[i] = output[i - 1] + input[i];
    }
}


void scan_hs_simd(const int *input, int *output, int size)
{
    if (size == 0)
        return;

    // Asignación única
    std::vector<int> buffer(size);
    int *in_buf = output;
    int *out_buf = buffer.data();

   
    std::memcpy(in_buf, input, size * sizeof(int));

    int steps = static_cast<int>(std::ceil(std::log2(size)));

    for (int step = 0; step < steps; ++step)
    {
        int offset = 1 << step;

      
        if (offset < size)
        {
            std::memcpy(out_buf, in_buf, offset * sizeof(int));
        }

        int i = offset;

        for (; i <= size - 8; i += 8)
        {
            __m256i v_curr = _mm256_loadu_si256((const __m256i *)&in_buf[i]);
            __m256i v_prev = _mm256_loadu_si256((const __m256i *)&in_buf[i - offset]);
            __m256i v_sum = _mm256_add_epi32(v_curr, v_prev);
            _mm256_storeu_si256((__m256i *)&out_buf[i], v_sum);
        }

        // Peeling
        for (; i < size; ++i)
        {
            out_buf[i] = in_buf[i] + in_buf[i - offset];
        }

        std::swap(in_buf, out_buf);
    }

    if (in_buf != output)
    {
        std::memcpy(output, in_buf, size * sizeof(int));
    }
}


void scan_hs_omp(const int *input, int *output, int size)
{
    if (size == 0)
        return;

    std::vector<int> buffer(size);
    int *in_buf = output;
    int *out_buf = buffer.data();
    int steps = static_cast<int>(std::ceil(std::log2(size)));

#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < size; ++i)
            output[i] = input[i];

        for (int step = 0; step < steps; ++step)
        {
            int offset = 1 << step;

#pragma omp for schedule(static)
            for (int i = 0; i < size; ++i)
            {
                if (i >= offset)
                {
                    out_buf[i] = in_buf[i] + in_buf[i - offset];
                }
                else
                {
                    out_buf[i] = in_buf[i];
                }
            }


#pragma omp single
            {
                std::swap(in_buf, out_buf);
            }
        }
        if (in_buf != output)
        {
#pragma omp for
            for (int i = 0; i < size; ++i)
                output[i] = in_buf[i];
        }
    }
}

// ---------------------------------------------------------
void scan_hs_omp_simd(const int *input, int *output, int size)
{
    if (size == 0)
        return;

    std::vector<int> buffer(size);
    int *in_buf = output;
    int *out_buf = buffer.data();
    int steps = static_cast<int>(std::ceil(std::log2(size)));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        int chunk = (size + nthreads - 1) / nthreads;
        int start = tid * chunk;
        int end = std::min(start + chunk, size);

        int i = start;
        for (; i <= end - 8; i += 8)
        {
            __m256i v = _mm256_loadu_si256((const __m256i *)&input[i]);
            _mm256_storeu_si256((__m256i *)&output[i], v);
        }
        for (; i < end; ++i)
            output[i] = input[i];

        for (int step = 0; step < steps; ++step)
        {
            int offset = 1 << step;

#pragma omp barrier

            int work_start = std::max(start, offset);

            if (start < offset)
            {
                int copy_limit = std::min(end, offset);
                for (int j = start; j < copy_limit; ++j)
                {
                    out_buf[j] = in_buf[j];
                }
            }

            if (work_start < end)
            {
                int j = work_start;

                // Procesamos de 8 en 8 usando Intrinsics
                for (; j <= end - 8; j += 8)
                {
                    __m256i v_curr = _mm256_loadu_si256((const __m256i *)&in_buf[j]);
                    __m256i v_prev = _mm256_loadu_si256((const __m256i *)&in_buf[j - offset]);
                    __m256i v_sum = _mm256_add_epi32(v_curr, v_prev);
                    _mm256_storeu_si256((__m256i *)&out_buf[j], v_sum);
                }

                for (; j < end; ++j)
                {
                    out_buf[j] = in_buf[j] + in_buf[j - offset];
                }
            }

#pragma omp barrier
#pragma omp single
            {
                std::swap(in_buf, out_buf);
            }
        }
        if (in_buf != output)
        {
#pragma omp barrier
            int i = start;
            for (; i <= end - 8; i += 8)
            {
                __m256i v = _mm256_loadu_si256((const __m256i *)&in_buf[i]);
                _mm256_storeu_si256((__m256i *)&output[i], v);
            }
            for (; i < end; ++i)
                output[i] = in_buf[i];
        }
    }
}