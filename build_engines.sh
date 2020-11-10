#!/bin/bash

# VGG16
python build_engine.py data/frozen_graphs/vgg_16.pb data/engines/vgg_16_fp32_bs1.engine
python build_engine.py data/frozen_graphs/vgg_16.pb data/engines/vgg_16_fp16_bs1.engine --fp16_mode
python build_engine.py data/frozen_graphs/vgg_16.pb data/engines/vgg_16_int8_bs1.engine --int8_mode

# INCEPTION V2
python build_engine.py data/frozen_graphs/inception_v2.pb data/engines/inception_v2_fp32_bs1.engine
python build_engine.py data/frozen_graphs/inception_v2.pb data/engines/inception_v2_fp16_bs1.engine --fp16_mode
python build_engine.py data/frozen_graphs/inception_v2.pb data/engines/inception_v2_int8_bs1.engine --int8_mode

# RESNET 50
python build_engine.py data/frozen_graphs/resnet_v1_50.pb data/engines/resnet_v1_50_fp32_bs1.engine
python build_engine.py data/frozen_graphs/resnet_v1_50.pb data/engines/resnet_v1_50_fp16_bs1.engine --fp16_mode
python build_engine.py data/frozen_graphs/resnet_v1_50.pb data/engines/resnet_v1_50_int8_bs1.engine --int8_mode

# MOBILENET V1
python build_engine.py data/frozen_graphs/mobilenet_v1_1p0_224.pb data/engines/mobilenet_v1_1p0_224_fp32_bs1.engine
python build_engine.py data/frozen_graphs/mobilenet_v1_1p0_224.pb data/engines/mobilenet_v1_1p0_224_fp16_bs1.engine --fp16_mode
python build_engine.py data/frozen_graphs/mobilenet_v1_1p0_224.pb data/engines/mobilenet_v1_1p0_224_int8_bs1.engine --int8_mode