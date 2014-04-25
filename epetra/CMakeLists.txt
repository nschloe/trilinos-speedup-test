INCLUDE_DIRECTORIES(SYSTEM
                    ${Trilinos_INCLUDE_DIRS}
                    ${Trilinos_TPL_INCLUDE_DIRS})

SET(MY_EXECUTABLE "epetra-poisson1d")
ADD_EXECUTABLE(${MY_EXECUTABLE} "poisson1d.cpp")
TARGET_LINK_LIBRARIES(${MY_EXECUTABLE}
                      ${Trilinos_LIBRARIES})

SET(MY_EXECUTABLE "epetra-dot")
ADD_EXECUTABLE(${MY_EXECUTABLE} "strong-scaling-dotproduct.cpp")
TARGET_LINK_LIBRARIES(${MY_EXECUTABLE}
                      ${Trilinos_LIBRARIES})

SET(MY_EXECUTABLE "epetra-medley")
ADD_EXECUTABLE(${MY_EXECUTABLE} "strong-scaling-medley.cpp")
TARGET_LINK_LIBRARIES(${MY_EXECUTABLE}
                      ${Trilinos_LIBRARIES})
