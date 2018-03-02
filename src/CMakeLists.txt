add_subdirectory(test)

add_executable(uff_to_plan uff_to_plan.cpp)
target_link_libraries(uff_to_plan nvinfer nvparsers)
