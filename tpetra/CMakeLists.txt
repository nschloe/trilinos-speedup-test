INCLUDE_DIRECTORIES(SYSTEM
                    ${Trilinos_INCLUDE_DIRS}
                    ${Trilinos_TPL_INCLUDE_DIRS})

SET(MY_EXECUTABLE "tpetra-poisson1d")
ADD_EXECUTABLE(${MY_EXECUTABLE} "poisson1d.cpp")
TARGET_LINK_LIBRARIES(${MY_EXECUTABLE}
                      ${Trilinos_LIBRARIES})
