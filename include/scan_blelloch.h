#ifndef SCAN_BLELLOCH_H
#define SCAN_BLELLOCH_H

void scan_bl_serial(const int* input, int* output, int size);
void scan_bl_simd(const int* input, int* output, int size);
void scan_bl_omp(const int* input, int* output, int size);
void scan_bl_omp_simd(const int* input, int* output, int size);

#endif