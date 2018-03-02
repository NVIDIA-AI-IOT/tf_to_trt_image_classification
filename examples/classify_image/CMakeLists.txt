cuda_add_executable(classify_image classify_image.cu)
target_link_libraries(classify_image ${OpenCV_LIBS} nvinfer nvparsers)
