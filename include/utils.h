#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <functional> // Para poder recibir funciones lambda

void save_to_csv(const std::string& filename, 
                 const std::string& alg_name, 
                 const std::string& impl_name, 
                 int data_size, 
                 double time_ms, 
                 bool passed);

void wait_for_enter();

template<typename Func>
double measure_time(Func funcion_a_medir) {
    auto start = std::chrono::high_resolution_clock::now();
    
    funcion_a_medir();
    
    auto end = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}

#endif