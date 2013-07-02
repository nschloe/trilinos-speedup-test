INCLUDE_DIRECTORIES(SYSTEM
                    ${Trilinos_INCLUDE_DIRS}
                    ${Trilinos_TPL_INCLUDE_DIRS})

# ------------------------------------------------------------------------------
SET(MY_EXECUTABLE "poisson1d-epetra")
ADD_EXECUTABLE(${MY_EXECUTABLE} "poisson1d-epetra.cpp")
TARGET_LINK_LIBRARIES(${MY_EXECUTABLE}
                      ${Trilinos_LIBRARIES})
# ------------------------------------------------------------------------------
SET(MY_EXECUTABLE "poisson1d-tpetra")
ADD_EXECUTABLE(${MY_EXECUTABLE} "poisson1d-tpetra.cpp")
TARGET_LINK_LIBRARIES(${MY_EXECUTABLE}
                      ${Trilinos_LIBRARIES})
# ------------------------------------------------------------------------------
