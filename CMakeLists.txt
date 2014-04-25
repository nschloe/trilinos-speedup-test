CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

PROJECT(trilinos-speedup-test CXX)

FIND_PACKAGE(Trilinos REQUIRED)

# Make sure the compilers match.
IF(NOT ${Trilinos_CXX_COMPILER} STREQUAL ${CMAKE_CXX_COMPILER})
  MESSAGE(FATAL_ERROR "C++ compilers don't match (Trilinos: ${Trilinos_CXX_COMPILER}, ${PROJECT_NAME}: ${CMAKE_CXX_COMPILER}).")
ENDIF()

# Add compiler flags (e.g., -openmp).
LIST(APPEND CMAKE_CXX_FLAGS ${Trilinos_CXX_COMPILER_FLAGS})

IF(NOT DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build,
options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Profile
Release RelWithDebInfo MinSizeRel.")
ENDIF()

#IF(NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
#    # Add support for timing for non-debug builds.
#    ADD_DEFINITIONS(-DNOSH_TEUCHOS_TIME_MONITOR)
#ENDIF()

IF(CMAKE_COMPILER_IS_GNUCXX)
    SET(CMAKE_CXX_FLAGS_DEBUG "-Og -g -ggdb -Wall -pedantic -fbounds-check -Wextra -Wstrict-null-sentinel -Wshadow -Woverloaded-virtual -Weffc++ -Wsign-compare -ansi -std=c++11" )
    SET(CMAKE_CXX_FLAGS_PROFILE "-pg -O2" )
ENDIF()

ADD_SUBDIRECTORY(epetra)
ADD_SUBDIRECTORY(tpetra)
