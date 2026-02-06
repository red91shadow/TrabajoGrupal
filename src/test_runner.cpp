#include "test_runner.h"
#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip> 
#include "histogram.h"
#include "scan_hillis.h"
#include "scan_blelloch.h"
#include "utils.h"


void clear_screen() {
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
}

void TestRunner::run() {
    int opcion = 0;
    do {
        clear_screen();
        std::cout << "\n=================================\n"
                  << "      PROYECTO HPC - MENU        \n"
                  << "=================================\n"
                  << "1. Histograma\n"
                  << "2. Scan (Hillis-Steele)\n"
                  << "3. Scan (Blelloch)\n"
                  << "0. Salir\n"
                  << "Seleccione: ";
        std::cin >> opcion;

        switch (opcion) {
            case 1: run_histogram_tests(); break;
            case 2: run_scan_hillis_tests(); break;
            case 3: run_scan_blelloch_tests(); break;
            case 0: std::cout << "Saliendo...\n"; break;
            default: std::cout << "Opcion invalida.\n";
        }
        
        if (opcion != 0) wait_for_enter();

    } while (opcion != 0);
}


void TestRunner::run_histogram_tests() {
    std::vector<int> sizes = {10000000, 50000000, 100000000}; 
    const int BINS = 256; 

    std::cout << "\n=== TESTS DE HISTOGRAMA ===\n";

    for (int N : sizes) {
        std::cout << "\n[ N=" << N << " ]\n";
        
        // Generar datos
        std::vector<int> data(N);
        #pragma omp parallel for
        for(int i=0; i<N; ++i) data[i] = i % BINS;

        std::vector<int> ref(BINS), v_simd(BINS), v_omp(BINS), v_ompsimd(BINS);

        
        double t_ref = measure_time([&](){ histogram_serial(data.data(), N, ref.data(), BINS, 0, BINS-1); });
        std::cout << "  Serial:       " << t_ref << " ms" << std::endl;
        save_to_csv("resultados.csv", "Histograma", "Serial", N, t_ref, true);

      
        auto run_test = [&](const char* name, std::vector<int>& out, auto func) {
            double t = measure_time(func);
            bool ok = (out == ref);
            std::cout << "  " << std::left << std::setw(14) << name << ": " << t << " ms " << (ok ? "(OK)" : "(FAIL)") << std::endl;
            save_to_csv("resultados.csv", "Histograma", name, N, t, ok);
        };

        run_test("SIMD",        v_simd,    [&](){ histogram_simd(data.data(), N, v_simd.data(), BINS, 0, BINS-1); });
        run_test("OpenMP",      v_omp,     [&](){ histogram_omp(data.data(), N, v_omp.data(), BINS, 0, BINS-1); });
        run_test("OpenMP+SIMD", v_ompsimd, [&](){ histogram_omp_simd(data.data(), N, v_ompsimd.data(), BINS, 0, BINS-1); });
    }
}


void TestRunner::run_scan_hillis_tests() {
    std::vector<int> sizes = {1000000, 10000000, 50000000}; 
    std::cout << "\n=== TESTS SCAN HILLIS-STEELE ===\n";
    
    for (int N : sizes) {
        std::cout << "\n[ N=" << N << " ]\n";
        std::vector<int> data(N, 1), ref(N), v_simd(N), v_omp(N), v_ompsimd(N);

        double t_ref = measure_time([&](){ scan_hs_serial(data.data(), ref.data(), N); });
        std::cout << "  Serial:       " << t_ref << " ms" << std::endl;
        save_to_csv("resultados.csv", "ScanHillis", "Serial", N, t_ref, true);

        auto run_test = [&](const char* name, std::vector<int>& out, auto func) {
            double t = measure_time(func);
            bool ok = (out == ref);
            std::cout << "  " << std::left << std::setw(14) << name << ": " << t << " ms " << (ok ? "(OK)" : "(FAIL)") << std::endl;
            save_to_csv("resultados.csv", "ScanHillis", name, N, t, ok);
        };

        run_test("SIMD",        v_simd,    [&](){ scan_hs_simd(data.data(), v_simd.data(), N); });
        run_test("OpenMP",      v_omp,     [&](){ scan_hs_omp(data.data(), v_omp.data(), N); });
        run_test("OpenMP+SIMD", v_ompsimd, [&](){ scan_hs_omp_simd(data.data(), v_ompsimd.data(), N); });
    }
}


void TestRunner::run_scan_blelloch_tests() {
    std::vector<int> sizes = {1000000, 10000000, 50000000}; 
    std::cout << "\n=== TESTS SCAN BLELLOCH ===\n";

    for (int N : sizes) {
        std::cout << "\n[ N=" << N << " ]\n";
        std::vector<int> data(N, 1), ref(N), v_simd(N), v_omp(N), v_ompsimd(N);

        double t_ref = measure_time([&](){ scan_bl_serial(data.data(), ref.data(), N); });
        std::cout << "  Serial:       " << t_ref << " ms" << std::endl;
        save_to_csv("resultados.csv", "ScanBlelloch", "Serial", N, t_ref, true);

        auto run_test = [&](const char* name, std::vector<int>& out, auto func) {
            double t = measure_time(func);
            bool ok = (out == ref);
            std::cout << "  " << std::left << std::setw(14) << name << ": " << t << " ms " << (ok ? "(OK)" : "(FAIL)") << std::endl;
            save_to_csv("resultados.csv", "ScanBlelloch", name, N, t, ok);
        };

        run_test("SIMD",        v_simd,    [&](){ scan_bl_simd(data.data(), v_simd.data(), N); });
        run_test("OpenMP",      v_omp,     [&](){ scan_bl_omp(data.data(), v_omp.data(), N); });
        run_test("OpenMP+SIMD", v_ompsimd, [&](){ scan_bl_omp_simd(data.data(), v_ompsimd.data(), N); });
    }
}