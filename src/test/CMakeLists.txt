cuda_add_executable(test_trt test_trt.cu)
target_link_libraries(test_trt ${OpenCV_LIBS} nvinfer nvparsers)
