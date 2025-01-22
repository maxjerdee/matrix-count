cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Add the sample_core Python module
pybind11_add_module(sample_core MODULE src/sample_core.cpp src/preprocessing.cpp src/helpers.cpp src/estimates.cpp)
install(TARGETS sample_core DESTINATION ${SKBUILD_PROJECT_NAME})