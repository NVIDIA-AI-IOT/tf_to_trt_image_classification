# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

model_urls=(
  http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz 
  http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz 
  http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz 
  http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz 
  http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz 
  http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz 
  http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz 
  http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz 
  http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
  http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
  http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz
  http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
  http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz
  http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz
  http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz
  http://download.tensorflow.org/models/mobilenet_v1_0.50_160_2017_06_14.tar.gz
  http://download.tensorflow.org/models/mobilenet_v1_0.25_128_2017_06_14.tar.gz
)

mkdir -p data/checkpoints
cd data/checkpoints

for url in ${model_urls[@]}
do
  tarname=$(basename $url)
  echo $tarname
  wget $url
  tar -xzf $(basename $url)
  rm $tarname
done

cd ../..
