cmake_minimum_required(VERSION 3.20)

project(gmes_mt_forward_mfem LANGUAGES CXX)

option(GMES_MT_BUILD_PYTHON "Build the pybind11 Python module" ON)
option(GMES_MT_ENABLE_OPENMP "Enable OpenMP when available" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

find_package(MPI REQUIRED)
find_package(MFEM CONFIG QUIET)

if(TARGET MFEM::mfem)
  set(GMES_MFEM_TARGET MFEM::mfem)
elseif(TARGET mfem)
  set(GMES_MFEM_TARGET mfem)
elseif(DEFINED MFEM_LIBRARIES)
  set(GMES_MFEM_TARGET ${MFEM_LIBRARIES})
else()
  find_path(MFEM_INCLUDE_DIR
    NAMES mfem.hpp
    HINTS ${CMAKE_PREFIX_PATH} $ENV{CONDA_PREFIX}
    PATH_SUFFIXES include include/mfem
  )
  find_library(MFEM_LIBRARY
    NAMES mfem
    HINTS ${CMAKE_PREFIX_PATH} $ENV{CONDA_PREFIX}
    PATH_SUFFIXES lib
  )

  if(NOT MFEM_INCLUDE_DIR OR NOT MFEM_LIBRARY)
    message(FATAL_ERROR
      "MFEM was not found. Set MFEM_DIR to a directory containing "
      "MFEMConfig.cmake, or set CMAKE_PREFIX_PATH/CONDA_PREFIX to a prefix "
      "with include/mfem.hpp and lib/libmfem.so.")
  endif()

  add_library(MFEM::mfem UNKNOWN IMPORTED)
  set_target_properties(MFEM::mfem PROPERTIES
    IMPORTED_LOCATION "${MFEM_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MFEM_INCLUDE_DIR}"
  )
  set(GMES_MFEM_TARGET MFEM::mfem)
  set(MFEM_INCLUDE_DIRS "${MFEM_INCLUDE_DIR}")
endif()

add_library(gmes_mt_core
  src/mt_background_1d.cpp
  src/mt_boundary_conditions.cpp
  src/mt_driver.cpp
  src/mt_observations.cpp
  src/mt_solver.cpp
  src/mt_tensor_mesh.cpp
  src/mt_tensor_model.cpp
)
set_target_properties(gmes_mt_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(gmes_mt_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MFEM_INCLUDE_DIRS}
)

target_link_libraries(gmes_mt_core
  PUBLIC
    MPI::MPI_CXX
    ${GMES_MFEM_TARGET}
)

target_compile_definitions(gmes_mt_core PUBLIC GMES_MT_USE_MFEM=1)

if(GMES_MT_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(gmes_mt_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(mt_forward_cli apps/mt_forward_cli.cpp)
target_link_libraries(mt_forward_cli PRIVATE gmes_mt_core)

if(GMES_MT_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(mt_forward_mfem MODULE bindings/pybind_module.cpp)
  target_link_libraries(mt_forward_mfem PRIVATE gmes_mt_core)
  set_target_properties(mt_forward_mfem PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()

enable_testing()
add_test(
  NAME mt_forward_cli_smoke
  COMMAND mt_forward_cli --freq 1.0 --nx 4 --ny 4 --nz 4 --rho 100.0
)
