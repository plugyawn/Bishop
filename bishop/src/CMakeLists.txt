add_library(bishop bishop.cpp)
target_include_directories(bishop PUBLIC ${EIGEN3_INCLUDE_DIRS})

add_executable(bishop_example main.cpp)
target_link_libraries(bishop_example bishop)
