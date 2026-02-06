#ifndef SCAN_HILLIS_H
#define SCAN_HILLIS_H

void scan_hs_serial(const int* input, int* output, int size);
void scan_hs_simd(const int* input, int* output, int size);
void scan_hs_omp(const int* input, int* output, int size);
void scan_hs_omp_simd(const int* input, int* output, int size);

#endif