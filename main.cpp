#include <iostream>
#include <vector>

#include <omp.h>

#include <immintrin.h> //avx

#include <fmt/core.h>
#include <fmt/ranges.h>

int escalar_simd(const std::vector<float>& x, const std::vector<float> y) {
    int num_elementos = x.size();
    int suma = 0;
    
    int tope = (num_elementos/8) * 8;

    float sum_tmp[8];

    for(int i=0;i<tope;i+=8) {
        __m256 mx = _mm256_loadu_ps(&x[i]);
        __m256 my = _mm256_loadu_ps(&y[i]);

        __m256 mz = _mm256_mul_ps(mx,my);

        _mm256_storeu_ps(sum_tmp, mz);

        for (int j = 0; j < 8; j++) {
            suma += sum_tmp[j];
        }
    }

    //iterar sobre los elementos faltantes
    for(int i=tope;i<num_elementos;i++) {
        suma += x[i]*y[i];
    }

    return suma;
}

int escalar_simd2(const std::vector<float>& x, const std::vector<float> y) {
    int num_elementos = x.size();
    int suma = 0;
    
    int tope = (num_elementos/8) * 8;

    float sum_tmp[8];

    /**
     * _mm_hadd_ps
     * 
     * a = [ a0 | a1 | a2 | a3 ]
     * b = [ b0 | b1 | b2 | b3 ]
     * 
     * res = [ a0 + a1 | a2 + a3 | b0 + b1 | b2 + b3 ]
     */

    for(int i=0;i<tope;i+=8) {
        __m256 mx = _mm256_loadu_ps(&x[i]);
        __m256 my = _mm256_loadu_ps(&y[i]);

        __m256 mz = _mm256_mul_ps(mx,my);

        __m128 low  = _mm256_castps256_ps128(mz);   //[a0,a1,a2,a3,a4,a5,a6,a7] -> [a0,a1,a2,a3]
        __m128 high = _mm256_extractf128_ps(mz, 1); //[a0,a1,a2,a3,a4,a5,a6,a7] -> [a4,a5,a6,a7]

        __m128 sum_128 = _mm_add_ps(low, high);     //sum_128 = [a0+a4, a1+a5, a2+a6, a3+a7]

        sum_128 = _mm_hadd_ps(sum_128, sum_128);    //sum_128 = [a0+a4+a1+a5, a2+a6+a3+a7, ... , ...]
        sum_128 = _mm_hadd_ps(sum_128, sum_128);

        suma += _mm_cvtss_f32(sum_128);             //extraer el primer elemento de sum_128
    }

    //iterar sobre los elementos faltantes
    for(int i=tope;i<num_elementos;i++) {
        suma += x[i]*y[i];
    }

    return suma;
}

int escalar_secciones_paralelas(const std::vector<float>& x, const std::vector<float> y) {
    int num_elementos = x.size();
    int suma = 0;

    #pragma omp parallel shared(x,y,num_elementos,suma)
    {
        int thread_id = omp_get_thread_num();
        int thread_count = omp_get_num_threads();

        int block_size = std::ceil(1.0 * num_elementos / thread_count);

        int start = thread_id * block_size;
        int end = (thread_id+1)*block_size;

        if(end>num_elementos) {
            end = num_elementos;
        }

        //fmt::println("thread_{}: [{}..{}]", thread_id, start, end);

        int local_sum = 0;
        for(int i=start;i<end;i++) {
            local_sum += x[i]*y[i];
        }

        #pragma omp critical
        suma = suma + local_sum;
    }

    return suma;
}

int main() {

    std::vector<float> x = {1,1,1,1,1,1,1,1,1,1};
    std::vector<float> y = {2,2,2,2,2,2,2,2,2,2};

    auto suma1 = escalar_simd(x,y);
    fmt::println("SIMD_1: {}", suma1);

    auto suma2 = escalar_simd2(x,y);
    fmt::println("SIMD_2: {}", suma2);

    auto suma3 = escalar_secciones_paralelas(x,y);
    fmt::println("OpenMP: {}", suma3);
    return 0;
}

