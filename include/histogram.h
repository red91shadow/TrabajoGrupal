#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>

void histogram_serial(const int *input, int size, int *bins, int num_bins);
void histogram_simd(const int *input, int size, int *bins, int num_bins);     
void histogram_omp(const int *input, int size, int *bins, int num_bins);      
void histogram_omp_simd(const int *input, int size, int *bins, int num_bins); 

#endif