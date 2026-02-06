#include "scan_hillis.h"
#include <omp.h>
#include <immintrin.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

// ---------------------------------------------------------
// 1. SERIAL - Prefix Sum Secuencial (Baseline)
// ---------------------------------------------------------
void scan_hs_serial(const int* input, int* output, int size) {
    if (size == 0) return;
    
    output[0] = input[0];
    for (int i = 1; i < size; ++i) {
        output[i] = output[i - 1] + input[i];
    }
}

// ---------------------------------------------------------
// 2. SIMD - Hillis Steele con Vectorización
// ---------------------------------------------------------
void scan_hs_simd(const int* input, int* output, int size) {
    if (size == 0) return;

    // Buffers para ping-pong
    std::vector<int> buffer(size);
    int* in_buf = output;
    int* out_buf = buffer.data();
    
    // Copiar input a output
    std::memcpy(in_buf, input, size * sizeof(int));

    int steps = static_cast<int>(std::ceil(std::log2(size)));

    for (int step = 0; step < steps; ++step) {
        int offset = 1 << step; // 1, 2, 4, 8, 16...

        // Elementos sin vecino a la izquierda (i < offset)
        for (int i = 0; i < std::min(offset, size); ++i) {
            out_buf[i] = in_buf[i];
        }

        // Procesar con SIMD los elementos >= offset
        int i = offset;
        
        // Vectorización: procesar de 8 en 8
        for (; i <= size - 8; i += 8) {
            __m256i vec_current = _mm256_loadu_si256((const __m256i*)&in_buf[i]);
            __m256i vec_prev = _mm256_loadu_si256((const __m256i*)&in_buf[i - offset]);
            __m256i vec_sum = _mm256_add_epi32(vec_current, vec_prev);
            _mm256_storeu_si256((__m256i*)&out_buf[i], vec_sum);
        }

        // Peeling: elementos restantes
        for (; i < size; ++i) {
            out_buf[i] = in_buf[i] + in_buf[i - offset];
        }

        // Intercambiar buffers
        std::swap(in_buf, out_buf);
    }

    // Si el resultado quedó en el buffer auxiliar, copiarlo
    if (in_buf != output) {
        std::memcpy(output, in_buf, size * sizeof(int));
    }
}

// ---------------------------------------------------------
// 3. OPENMP - Hillis Steele Paralelo
// ---------------------------------------------------------
void scan_hs_omp(const int* input, int* output, int size) {
    if (size == 0) return;

    // Copiar input a output en paralelo
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        output[i] = input[i];
    }

    std::vector<int> buffer(size);
    int* in_buf = output;
    int* out_buf = buffer.data();
    
    int steps = static_cast<int>(std::ceil(std::log2(size)));

    for (int step = 0; step < steps; ++step) {
        int offset = 1 << step;

        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            if (i >= offset) {
                out_buf[i] = in_buf[i] + in_buf[i - offset];
            } else {
                out_buf[i] = in_buf[i];
            }
        }
        
        std::swap(in_buf, out_buf);
    }

    // Copiar resultado final si es necesario
    if (in_buf != output) {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            output[i] = in_buf[i];
        }
    }
}

// ---------------------------------------------------------
// 4. OPENMP + SIMD - Versión Híbrida Optimizada
// ---------------------------------------------------------
void scan_hs_omp_simd(const int* input, int* output, int size) {
    if (size == 0) return;

    // Inicialización paralela
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        output[i] = input[i];
    }

    std::vector<int> buffer(size);
    int* in_buf = output;
    int* out_buf = buffer.data();
    
    int steps = static_cast<int>(std::ceil(std::log2(size)));

    for (int step = 0; step < steps; ++step) {
        int offset = 1 << step;

        #pragma omp parallel
        {
            // Copiar elementos sin vecino (i < offset)
            #pragma omp for nowait
            for (int i = 0; i < std::min(offset, size); ++i) {
                out_buf[i] = in_buf[i];
            }

            // Procesar elementos con SIMD (i >= offset)
            #pragma omp for schedule(static)
            for (int i = offset; i < size; ++i) {
                // Verificar si podemos usar SIMD (bloques de 8)
                if (i <= size - 8 && (i - offset) % 8 == 0) {
                    // Cada thread procesa su propio bloque de 8
                    __m256i vec_current = _mm256_loadu_si256((const __m256i*)&in_buf[i]);
                    __m256i vec_prev = _mm256_loadu_si256((const __m256i*)&in_buf[i - offset]);
                    __m256i vec_sum = _mm256_add_epi32(vec_current, vec_prev);
                    _mm256_storeu_si256((__m256i*)&out_buf[i], vec_sum);
                    
                    // Saltar los siguientes 7 elementos (ya procesados)
                    i += 7;
                } else {
                    // Procesamiento escalar para elementos que no alinean
                    out_buf[i] = in_buf[i] + in_buf[i - offset];
                }
            }
        }
        
        std::swap(in_buf, out_buf);
    }

    // Copiar resultado final si quedó en el buffer auxiliar
    if (in_buf != output) {
        #pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            output[i] = in_buf[i];
        }
    }
}