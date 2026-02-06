#include <iostream>
#include <vector>
#include <omp.h>
#include "histogram.h"
#include "utils.h" 
#include "scan_hillis.h" 

void run_histogram_tests() {
    // Tamaños de prueba (puedes ajustar estos números)
    std::vector<int> test_sizes = {1000, 50000000, 100000000}; 
    const int NUM_BINS = 256; 
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "          TESTS DE HISTOGRAMA           " << std::endl;
    std::cout << "========================================" << std::endl;
    
    for (int N : test_sizes) {
        std::cout << "\n[ DATA SIZE: " << N << " elementos ]" << std::endl;
        
        // 1. Preparar datos (Generación paralela para rapidez)
        std::vector<int> datos(N);
        #pragma omp parallel for
        for(int i=0; i<N; ++i) datos[i] = i % NUM_BINS;

        // Vectores de resultados (limpios para cada test)
        std::vector<int> bins_ref(NUM_BINS, 0);       // Serial (Referencia)
        std::vector<int> bins_simd(NUM_BINS, 0);      // SIMD
        std::vector<int> bins_omp(NUM_BINS, 0);       // OpenMP
        std::vector<int> bins_omp_simd(NUM_BINS, 0);  // OpenMP + SIMD

        // ---------------------------------------------------------
        // 1. SERIAL
        // ---------------------------------------------------------
        double t_serial = measure_time([&](){ 
            histogram_serial(datos.data(), N, bins_ref.data(), NUM_BINS); 
        });
        std::cout << "  Serial:       " << t_serial << " ms" << std::endl;
        save_to_csv("resultados.csv", "Histograma", "Serial", N, t_serial, true);

        // ---------------------------------------------------------
        // 2. SIMD 
        // ---------------------------------------------------------
        double t_simd = measure_time([&](){ 
            histogram_simd(datos.data(), N, bins_simd.data(), NUM_BINS); 
        });
        bool ok_simd = (bins_ref == bins_simd);
        std::cout << "  SIMD:         " << t_simd << " ms " << (ok_simd ? "(OK)" : "(FAIL)") << std::endl;
        save_to_csv("resultados.csv", "Histograma", "SIMD", N, t_simd, ok_simd);

        // ---------------------------------------------------------
        // 3. OPENMP
        // ---------------------------------------------------------
        double t_omp = measure_time([&](){ 
            histogram_omp(datos.data(), N, bins_omp.data(), NUM_BINS); 
        });
        bool ok_omp = (bins_ref == bins_omp);
        std::cout << "  OpenMP:       " << t_omp << " ms " << (ok_omp ? "(OK)" : "(FAIL)") << std::endl;
        save_to_csv("resultados.csv", "Histograma", "OpenMP", N, t_omp, ok_omp);

        // ---------------------------------------------------------
        // 4. OPENMP + SIMD
        // ---------------------------------------------------------
        double t_omp_simd = measure_time([&](){ 
            histogram_omp_simd(datos.data(), N, bins_omp_simd.data(), NUM_BINS); 
        });
        bool ok_omp_simd = (bins_ref == bins_omp_simd);
        std::cout << "  OpenMP+SIMD:  " << t_omp_simd << " ms " << (ok_omp_simd ? "(OK)" : "(FAIL)") << std::endl;
        save_to_csv("resultados.csv", "Histograma", "OpenMP+SIMD", N, t_omp_simd, ok_omp_simd);
    }
}

void run_scan_hillis_tests() {
    // Para Scan Hillis, OJO: Usa mucha memoria (doble buffer).
    // Empieza con tamaños más "humildes" que el histograma.
    // Hillis hace O(N log N) operaciones, es más pesado que el histograma.
    std::vector<int> test_sizes = {1000000, 10000000, 50000000}; 
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "      TESTS DE SCAN (HILLIS-STEELE)     " << std::endl;
    std::cout << "========================================" << std::endl;
    
    for (int N : test_sizes) {
        std::cout << "\n[ DATA SIZE: " << N << " elementos ]" << std::endl;
        
        // 1. Datos (puros 1s para verificar fácil: resultado debe ser 1,2,3...N)
        std::vector<int> datos(N, 1);
        
        std::vector<int> out_ref(N);
        std::vector<int> out_simd(N);
        std::vector<int> out_omp(N);
        std::vector<int> out_omp_simd(N);

        // --- SERIAL ---
        double t_serial = measure_time([&](){ 
            scan_hs_serial(datos.data(), out_ref.data(), N); 
        });
        std::cout << "  Serial:       " << t_serial << " ms" << std::endl;
        save_to_csv("resultados.csv", "ScanHillis", "Serial", N, t_serial, true);

        // --- SIMD ---
        double t_simd = measure_time([&](){ 
            scan_hs_simd(datos.data(), out_simd.data(), N); 
        });
        bool ok_simd = (out_ref == out_simd); // Compara vector completo
        std::cout << "  SIMD:         " << t_simd << " ms " << (ok_simd ? "(OK)" : "(FAIL)") << std::endl;
        save_to_csv("resultados.csv", "ScanHillis", "SIMD", N, t_simd, ok_simd);

        // --- OPENMP ---
        double t_omp = measure_time([&](){ 
            scan_hs_omp(datos.data(), out_omp.data(), N); 
        });
        bool ok_omp = (out_ref == out_omp);
        std::cout << "  OpenMP:       " << t_omp << " ms " << (ok_omp ? "(OK)" : "(FAIL)") << std::endl;
        save_to_csv("resultados.csv", "ScanHillis", "OpenMP", N, t_omp, ok_omp);

        // --- OPENMP + SIMD ---
        double t_omp_simd = measure_time([&](){ 
            scan_hs_omp_simd(datos.data(), out_omp_simd.data(), N); 
        });
        bool ok_omp_simd = (out_ref == out_omp_simd);
        std::cout << "  OpenMP+SIMD:  " << t_omp_simd << " ms " << (ok_omp_simd ? "(OK)" : "(FAIL)") << std::endl;
        save_to_csv("resultados.csv", "ScanHillis", "OpenMP+SIMD", N, t_omp_simd, ok_omp_simd);
    }
}

int main() {
    int opcion = 0;
    do {
        // Limpiar pantalla simple
        std::cout << "\n=================================\n";
        std::cout << "      PROYECTO HPC - MENU        \n";
        std::cout << "=================================\n";
        std::cout << "1. Correr pruebas de Histograma\n";
        std::cout << "2. Correr pruebas de Scan (Hillis)\n";
        std::cout << "3. Correr pruebas de Scan (Blelloch)\n";
        std::cout << "0. Salir\n";
        std::cout << "Seleccione: ";
        std::cin >> opcion;

        switch (opcion) {
            case 1: run_histogram_tests(); break;
            case 2: run_scan_hillis_tests(); break;
            case 3: std::cout << "En construccion...\n"; break;
            case 0: std::cout << "Adios!\n"; break;
            default: std::cout << "Opcion invalida.\n";
        }
        
        if (opcion != 0) wait_for_enter();

    } while (opcion != 0);

    return 0;
}