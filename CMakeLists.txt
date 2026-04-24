cmake_minimum_required(VERSION 3.18)
project(neural_field LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# Frontera rtx nodes carry NVIDIA Quadro RTX 5000 (Turing, sm_75).
# Override with -DCMAKE_CUDA_ARCHITECTURES=xx if targeting a different GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(neural_field
  src/main.cu
  src/io.cpp
)

target_include_directories(neural_field PRIVATE src)
target_link_libraries(neural_field PRIVATE cufft)

# warnings on both the host and device compilers
target_compile_options(neural_field PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

install(TARGETS neural_field DESTINATION bin)
