add_library(bishop bishop.cpp)

add_executable(main main.cpp)
target_link_libraries(main bishop)
# Link the main of bishop to the main in main.cpp (static linkage?)