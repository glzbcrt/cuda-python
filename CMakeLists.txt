cmake_minimum_required(VERSION 3.0.0)
project(cuda-python VERSION 0.1.0 LANGUAGES C CXX CUDA)
add_library(cuda-python SHARED kernel.cu)