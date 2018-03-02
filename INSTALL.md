Installation
===

1. Flash the Jetson TX2 using JetPack 3.2.  Be sure to install
   * CUDA 9.0
   * OpenCV4Tegra
   * cuDNN
   * TensorRT 3.0

2. Install TensorFlow on Jetson TX2.
   1. Download TensorFlow 1.5.0 / CUDA 9 / cuDNN 7 pip wheel from [here](https://drive.google.com/open?id=1BNOaSdfd6YyitTa4DLD7j4L45-Duo2LR).
   2. Install TensorFlow using pip
  
            sudo pip install tensorflow-1.5.0rc0-cp27-cp27mu-linux_aarch64.whl

3. Install uff converter on Jetson TX2.
   1. Download TensorRT 3.0.4 for Ubuntu 16.04 and CUDA 9.0 tar package from https://developer.nvidia.com/nvidia-tensorrt-download.
   2. Extract archive 

            tar -xzf TensorRT-3.0.4.Ubuntu-16.04.3.x86_64.cuda-9.0.cudnn7.0.tar.gz

   3. Install uff python package using pip 

            sudo pip install TensorRT-3.0.4/uff/uff-0.2.0-py2.py3-none-any.whl

4. Clone and build this project

    ```
    git clone --recursive https://github.com/NVIDIA-Jetson/tf_to_trt_image_classification.git
    cd tf_to_trt_image_classification
    mkdir build
    cd build
    cmake ..
    make 
    cd ..
    ```
