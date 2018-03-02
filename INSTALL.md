Installation
===

1. Flash the Jetson TX2 using JetPack 3.2.  Be sure to install
  * CUDA 9.0
  * OpenCV4Tegra
  * cuDNN
  * TensorRT 3.0

2. Install TensorFlow on Jetson TX2.
  1. ...
  2. ...

3. Install uff converter on Jetson TX2.
  1. Download TensorRT 3.0.4 for Ubuntu 16.04 and CUDA 9.0 tar package from https://developer.nvidia.com/nvidia-tensorrt-download.
  2. Extract archive 

            tar -xzf TensorRT-3.0.4.Ubuntu-16.04.3.x86_64.cuda-9.0.cudnn7.0.tar.gz

  3. Install uff python package using pip 

            sudo pip install TensorRT-3.0.4/uff/uff-0.2.0-py2.py3-none-any.whl

4. Clone and build this project

    ```
    git clone --recursive https://gitlab-master.nvidia.com/jwelsh/trt_image_classification.git
    cd trt_image_classification
    mkdir build
    cd build
    cmake ..
    make 
    cd ..
    ```
