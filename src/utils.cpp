#include "utils.h"
#include <fstream>
#include <iostream>

void save_to_csv(const std::string& filename, 
                 const std::string& alg_name, 
                 const std::string& impl_name, 
                 int data_size, 
                 double time_ms, 
                 bool passed) {
    
    bool file_exists = false;
    {
        std::ifstream f(filename);
        file_exists = f.good();
    }

    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) return;

    if (!file_exists) {
        file << "Algoritmo,Implementacion,N,Tiempo_ms,Correcto\n";
    }

    file << alg_name << "," 
         << impl_name << "," 
         << data_size << "," 
         << time_ms << "," 
         << (passed ? "SI" : "NO") << "\n";
}

void wait_for_enter() {
    std::cout << "\nPresiona ENTER para continuar...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
}