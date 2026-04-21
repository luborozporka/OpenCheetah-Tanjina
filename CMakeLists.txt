cmake_minimum_required(VERSION 3.11)
project(Cheetah)
include(CMakeFindDependencyMacro)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_compile_options(-fPIC)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

## Dependencies
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Eigen3 3.3.3 REQUIRED)
find_package(SEAL REQUIRED)
find_package(emp-tool REQUIRED)
find_package(emp-ot REQUIRED)
# Boost.Asio: Phase 1 multi-client server
# Set BOOST_ROOT if Boost is installed outside the default system prefix
find_package(Boost 1.66 REQUIRED COMPONENTS system)
include_directories(${EIGEN3_INCLUDE_DIR}
                    ${EMP-OT_INCLUDE_DIRS} 
                    ${EMP-TOOL_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS})

## cheetah-core
add_library(gemini SHARED)
include_directories(include)
add_subdirectory(include)
target_link_libraries(gemini SEAL::seal Eigen3::Eigen Threads::Threads)

if (USE_APPROX_RESHARE)
  target_compile_definitions(gemini PUBLIC USE_APPROX_RESHARE=1)
else()
  target_compile_definitions(gemini PUBLIC USE_APPROX_RESHARE=0)
endif()

## SCI
add_subdirectory(SCI)
include_directories(SCI/src/)

## demo
macro (add_network_cheetah _name)
  add_executable(${_name}-cheetah "networks/main_${_name}.cpp")
  target_link_libraries(${_name}-cheetah gemini SCI-Cheetah)
  target_compile_definitions(${_name}-cheetah PUBLIC SCI_OT=1 USE_CHEETAH=1)
endmacro()

add_network_cheetah(resnet50)
add_network_cheetah(sqnet)
add_network_cheetah(densenet121)

add_executable(cheetah-server
  server/cheetah-server.cpp
  networks/main_sqnet.cpp
  networks/main_resnet50.cpp)
target_compile_definitions(cheetah-server
  PUBLIC SCI_OT=1 USE_CHEETAH=1 CHEETAH_EMBEDDED_BUILD=1)
target_link_libraries(cheetah-server gemini SCI-Cheetah Boost::system Threads::Threads)

add_executable(cheetah-client
  client/cheetah-client.cpp
  networks/main_sqnet.cpp
  networks/main_resnet50.cpp)
target_compile_definitions(cheetah-client
  PUBLIC SCI_OT=1 USE_CHEETAH=1 CHEETAH_EMBEDDED_BUILD=1)
target_link_libraries(cheetah-client gemini SCI-Cheetah Boost::system Threads::Threads)
add_network_cheetah(short1)
add_network_cheetah(short2)
add_network_cheetah(relu1_28_28_1)
add_network_cheetah(relu784)
add_network_cheetah(relu12_23_34_45)
add_network_cheetah(relu422280)
add_network_cheetah(relu1)
add_network_cheetah(mp1)
add_network_cheetah(mp2)
add_network_cheetah(mp3)
add_network_cheetah(mp4)
