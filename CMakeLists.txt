cmake_minimum_required(VERSION 3.16)

project(Examen01)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -mavx2")

find_package(fmt CONFIG REQUIRED)

add_executable(correccion_examen01 main.cpp)

target_link_libraries(correccion_examen01 PRIVATE 
    fmt::fmt
)