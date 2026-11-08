cmake_minimum_required(VERSION 3.15)
project(rag_core CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(rag_core INTERFACE)

target_include_directories(rag_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_executable(test_tokenizer test/main.cpp)

target_link_libraries(test_tokenizer PRIVATE rag_core)
