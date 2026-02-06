#include "histogram.h"
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <vector>

void histogram_serial(const int *input, int size, int *bins, int num_bins)
{
    for (int i = 0; i < size; ++i)
    {
        int valor = input[i];
        if (valor >= 0 && valor < num_bins)
        {
            bins[valor]++;
        }
    }
}

void histogram_simd(const int *input, int size, int *bins, int num_bins)
{
    // PROCESAMIENTO POR BLOQUES DE 8 (AVX2)
    int i = 0;
    
    // El límite es size - 8 para asegurar que siempre podemos cargar 8 elementos
    for (; i <= size - 8; i += 8)
    {
        // 1. Cargar 8 enteros desde memoria a registro AVX
        __m256i data = _mm256_loadu_si256((const __m256i *)&input[i]);
        
        // 2. Bajarlos a un arreglo temporal (AVX2 no tiene Scatter eficiente)
        int temp[8];
        _mm256_storeu_si256((__m256i *)temp, data);
        
        // 3. Sumar directamente al histograma global
        // Nota: Como 'num_bins' es pequeño (256), esto se mantiene en Caché L1.
        // Usamos validación de rango por seguridad.
        if (temp[0] < num_bins) bins[temp[0]]++;
        if (temp[1] < num_bins) bins[temp[1]]++;
        if (temp[2] < num_bins) bins[temp[2]]++;
        if (temp[3] < num_bins) bins[temp[3]]++;
        if (temp[4] < num_bins) bins[temp[4]]++;
        if (temp[5] < num_bins) bins[temp[5]]++;
        if (temp[6] < num_bins) bins[temp[6]]++;
        if (temp[7] < num_bins) bins[temp[7]]++;
    }
    
    // PEELING (Procesar los elementos restantes que no completaron un grupo de 8)
    for (; i < size; ++i)
    {
        int valor = input[i];
        if (valor < num_bins) {
            bins[valor]++;
        }
    }
}

void histogram_omp(const int *input, int size, int *bins, int num_bins)
{
    // Alinear los bins locales para evitar false sharing
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
            int valor = input[i];
            if (valor >= 0 && valor < num_bins)
            {
                my_bins[valor]++;
            }
        }
    }

    // Reducción
    for (int t = 0; t < max_threads; ++t)
    {
        int *thread_bins = &local_bins_flat[t * PADDED_BINS];
        for (int b = 0; b < num_bins; ++b)
        {
            bins[b] += thread_bins[b];
        }
    }
}

void histogram_omp_simd(const int *input, int size, int *bins, int num_bins)
{
    const int CACHE_LINE = 64;
    const int PADDED_BINS = ((num_bins * sizeof(int) + CACHE_LINE - 1) / CACHE_LINE) * CACHE_LINE / sizeof(int);
    
    int max_threads = omp_get_max_threads();
    std::vector<int> local_bins_flat(max_threads * PADDED_BINS, 0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int *my_bins = &local_bins_flat[tid * PADDED_BINS];

        #pragma omp for schedule(static) nowait
        for (int i = 0; i <= size - 8; i += 8)
        {
            __m256i data = _mm256_loadu_si256((__m256i *)&input[i]);
            
            int temp[8];
            _mm256_storeu_si256((__m256i *)temp, data);
            
            // Desenrollar completamente
            if (temp[0] >= 0 && temp[0] < num_bins) my_bins[temp[0]]++;
            if (temp[1] >= 0 && temp[1] < num_bins) my_bins[temp[1]]++;
            if (temp[2] >= 0 && temp[2] < num_bins) my_bins[temp[2]]++;
            if (temp[3] >= 0 && temp[3] < num_bins) my_bins[temp[3]]++;
            if (temp[4] >= 0 && temp[4] < num_bins) my_bins[temp[4]]++;
            if (temp[5] >= 0 && temp[5] < num_bins) my_bins[temp[5]]++;
            if (temp[6] >= 0 && temp[6] < num_bins) my_bins[temp[6]]++;
            if (temp[7] >= 0 && temp[7] < num_bins) my_bins[temp[7]]++;
        }
        
        // Un solo thread procesa el resto
        #pragma omp single
        {
            int resto_start = (size / 8) * 8;
            for (int i = resto_start; i < size; ++i)
            {
                int valor = input[i];
                if (valor >= 0 && valor < num_bins)
                {
                    my_bins[valor]++;
                }
            }
        }
    }

    // Reducción
    for (int t = 0; t < max_threads; ++t)
    {
        int *thread_bins = &local_bins_flat[t * PADDED_BINS];
        for (int b = 0; b < num_bins; ++b)
        {
            bins[b] += thread_bins[b];
        }
    }
}