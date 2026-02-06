#include "scan_hillis.h"
#include <omp.h>
#include <immintrin.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

// ---------------------------------------------------------
// 1. SERIAL (Sin cambios, ya es óptimo)
// ---------------------------------------------------------
void scan_hs_serial(const int* input, int* output, int size) {
    if (size == 0) return;
    output[0] = input[0];
    for (int i = 1; i < size; ++i) {
        output[i] = output[i - 1] + input[i];
    }
}

// ---------------------------------------------------------
// 2. SIMD (Hillis Steele Vectorizado)
// ---------------------------------------------------------
void scan_hs_simd(const int* input, int* output, int size) {
    if (size == 0) return;

    // Asignación única
    std::vector<int> buffer(size);
    int* in_buf = output;
    int* out_buf = buffer.data();
    
    // Copia inicial usando memcpy (muy rápido)
    std::memcpy(in_buf, input, size * sizeof(int));

    int steps = static_cast<int>(std::ceil(std::log2(size)));

    for (int step = 0; step < steps; ++step) {
        int offset = 1 << step;
        
        // Optimización: Solo procesamos desde 'offset'
        // Copiamos la parte inicial que no cambia (serialmente es más rápido que vectorizar memcpy pequeños)
        if (offset < size) {
            std::memcpy(out_buf, in_buf, offset * sizeof(int));
        }

        int i = offset;
        
        // Bucle Principal SIMD
        // Usamos unrolling manual para exprimir AVX2
        for (; i <= size - 8; i += 8) {
            __m256i v_curr = _mm256_loadu_si256((const __m256i*)&in_buf[i]);
            __m256i v_prev = _mm256_loadu_si256((const __m256i*)&in_buf[i - offset]);
            __m256i v_sum  = _mm256_add_epi32(v_curr, v_prev);
            _mm256_storeu_si256((__m256i*)&out_buf[i], v_sum);
        }

        // Peeling
        for (; i < size; ++i) {
            out_buf[i] = in_buf[i] + in_buf[i - offset];
        }

        std::swap(in_buf, out_buf);
    }

    if (in_buf != output) {
        std::memcpy(output, in_buf, size * sizeof(int));
    }
}

// ---------------------------------------------------------
// 3. OPENMP (CORREGIDO: Región paralela externa)
// ---------------------------------------------------------
void scan_hs_omp(const int* input, int* output, int size) {
    if (size == 0) return;

    // 1. Asignación de memoria (costoso pero inevitable aquí)
    std::vector<int> buffer(size);
    int* in_buf = output;
    int* out_buf = buffer.data();
    int steps = static_cast<int>(std::ceil(std::log2(size)));

    // REGIÓN PARALELA ÚNICA (Evita overhead de crear hilos 20 veces)
    #pragma omp parallel
    {
        // Copia inicial paralela
        #pragma omp for
        for (int i = 0; i < size; ++i) output[i] = input[i];

        for (int step = 0; step < steps; ++step) {
            int offset = 1 << step;

            // Bucle de trabajo
            #pragma omp for schedule(static)
            for (int i = 0; i < size; ++i) {
                if (i >= offset) {
                    out_buf[i] = in_buf[i] + in_buf[i - offset];
                } else {
                    out_buf[i] = in_buf[i];
                }
            }
            
            // Sincronización obligatoria antes de cambiar buffers
            // Solo el hilo maestro hace el swap de punteros
            #pragma omp single
            {
                std::swap(in_buf, out_buf);
            }
            // Barrier implícito al final de single, o explícito para seguridad
        } // Fin del for steps

        // Copia final si es necesario
        if (in_buf != output) {
            #pragma omp for
            for (int i = 0; i < size; ++i) output[i] = in_buf[i];
        }
    } 
}


void scan_hs_omp_simd(const int* input, int* output, int size) {
    if (size == 0) return;

    std::vector<int> buffer(size);
    int* in_buf = output;
    int* out_buf = buffer.data();
    int steps = static_cast<int>(std::ceil(std::log2(size)));

    #pragma omp parallel
    {
        // Inicialización vectorizada
        #pragma omp for simd
        for (int i = 0; i < size; ++i) output[i] = input[i];

        for (int step = 0; step < steps; ++step) {
            int offset = 1 << step;

            // División estratégica:
            // 1. Parte izquierda (copia simple)
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < std::min(offset, size); ++i) {
                out_buf[i] = in_buf[i];
            }

            // 2. Parte derecha (Cálculo Pesado con SIMD)
            // Usamos 'simd' explícito aquí
            #pragma omp for simd schedule(static)
            for (int i = offset; i < size; ++i) {
                out_buf[i] = in_buf[i] + in_buf[i - offset];
            }
            
            // Esperar a todos antes de cambiar punteros
            #pragma omp barrier

            #pragma omp single
            {
                std::swap(in_buf, out_buf);
            }
        }

        if (in_buf != output) {
            #pragma omp for simd
            for (int i = 0; i < size; ++i) output[i] = in_buf[i];
        }
    }
}