cmake_minimum_required(VERSION 3.20)
project(stsupport_ccgcky LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_ccgcky MODULE
    cpp/category.cpp
    cpp/grammar.cpp
    cpp/parser.cpp
    cpp/bindings.cpp
)
target_include_directories(_ccgcky PRIVATE cpp)

if(MSVC)
    target_compile_options(_ccgcky PRIVATE /utf-8 /permissive- /W3 /EHsc)
else()
    target_compile_options(_ccgcky PRIVATE -Wall -Wextra)
endif()

install(TARGETS _ccgcky DESTINATION stsupport)
