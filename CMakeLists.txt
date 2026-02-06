cmake_minimum_required(VERSION 3.16)

project(Examen01)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenMP REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2") 


find_package(fmt CONFIG REQUIRED)


add_executable(grupal
    main.cpp
    src/histogram.cpp
    include/histogram.h
    src/utils.cpp
    include/utils.h
    src/scan_hillis.cpp
    include/scan_hillis.h
    src/scan_blelloch.cpp
    include/scan_blelloch.h

)


target_include_directories(grupal PRIVATE "${PROJECT_SOURCE_DIR}/include")


target_link_libraries(grupal PRIVATE
    fmt::fmt
    OpenMP::OpenMP_CXX  
)