# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

image_urls=(
  http://farm3.static.flickr.com/2017/2496831224_221cd963a2.jpg
  http://farm3.static.flickr.com/2582/4106642219_190bf0f817.jpg
  http://farm4.static.flickr.com/3226/2719028129_9aa2e27675.jpg
)

image_names=(
  gordon_setter.jpg
  lifeboat.jpg
  golden_retriever.jpg
)

image_folder=data/images

mkdir -p $image_folder

for i in ${!image_urls[@]}
do
  echo ${image_urls[$i]}
  echo ${image_names[$i]}
  wget -O $image_folder/${image_names[$i]} ${image_urls[$i]}
done
