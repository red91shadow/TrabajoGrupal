#ifndef HISTOGRAM_H
#define HISTOGRAM_H

void histogram_serial(const int* input, int size, int* bins, int num_bins, int min_val, int max_val);
void histogram_simd(const int* input, int size, int* bins, int num_bins, int min_val, int max_val);
void histogram_omp(const int* input, int size, int* bins, int num_bins, int min_val, int max_val);
void histogram_omp_simd(const int* input, int size, int* bins, int num_bins, int min_val, int max_val);

#endif