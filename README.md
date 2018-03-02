TensorFlow->TensorRT Image Classification
===

This contains examples, scripts and code related to image classification using TensorFlow models
(from [here](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained))
converted to TensorRT.  Converting TensorFlow models to TensorRT offers significant performance
gains on the Jetson TX2 as seen [below](#models).

<a name="toc"></a>
## Table of Contents

* [Model overview](#models)

* [Download models and create frozen graphs](#download)

* [Convert frozen graph to TensorRT engine](#convert)

* [Execute TensorRT engine](#execute)

* [Benchmark all models](#benchmark)

<a name="models"></a>
## Models

The table below shows various details related to pretrained models ported from the TensorFlow 
slim model zoo.  

| <sub>Model</sub> | <sub>Input Size</sub> | <sub>TensorRT (TX2 / Half)</sub> | <sub>TensorRT (TX2 / Float)</sub> | <sub>TensorFlow (TX2 / Float)</sub> | <sub>Input Name</sub> | <sub>Output Name</sub> | <sub>Preprocessing Fn.</sub> |
|--- |:---:|:---:|:---:|:---:|---|---|---|
| <sub>inception_v1</sub> | <sub>224x224</sub> | <sub>7.98ms</sub> | <sub>12.8ms</sub> | <sub>27.6ms</sub> | <sub>input</sub> | <sub>InceptionV1/Logits/SpatialSqueeze</sub> | <sub>inception</sub> |
| <sub>inception_v3</sub> | <sub>299x299</sub> | <sub>26.3ms</sub> | <sub>46.1ms</sub> | <sub>98.4ms</sub> | <sub>input</sub> | <sub>InceptionV3/Logits/SpatialSqueeze</sub> | <sub>inception</sub> |
| <sub>inception_v4</sub> | <sub>299x299</sub> | <sub>52.1ms</sub> | <sub>88.2ms</sub> | <sub>176ms</sub> | <sub>input</sub> | <sub>InceptionV4/Logits/Logits/BiasAdd</sub> | <sub>inception</sub> |
| <sub>inception_resnet_v2</sub> | <sub>299x299</sub> | <sub>53.0ms</sub> | <sub>98.7ms</sub> | <sub>168ms</sub> | <sub>input</sub> | <sub>InceptionResnetV2/Logits/Logits/BiasAdd</sub> | <sub>inception</sub> |
| <sub>resnet_v1_50</sub> | <sub>224x224</sub> | <sub>15.7ms</sub> | <sub>27.1ms</sub> | <sub>63.9ms</sub> | <sub>input</sub> | <sub>resnet_v1_50/SpatialSqueeze</sub> | <sub>vgg</sub> |
| <sub>resnet_v1_101</sub> | <sub>224x224</sub> | <sub>29.9ms</sub> | <sub>51.8ms</sub> | <sub>107ms</sub> | <sub>input</sub> | <sub>resnet_v1_101/SpatialSqueeze</sub> | <sub>vgg</sub> |
| <sub>resnet_v1_152</sub> | <sub>224x224</sub> | <sub>42.6ms</sub> | <sub>78.2ms</sub> | <sub>157ms</sub> | <sub>input</sub> | <sub>resnet_v1_152/SpatialSqueeze</sub> | <sub>vgg</sub> |
| <sub>resnet_v2_50</sub> | <sub>299x299</sub> | <sub>27.5ms</sub> | <sub>44.4ms</sub> | <sub>92.2ms</sub> | <sub>input</sub> | <sub>resnet_v2_50/SpatialSqueeze</sub> | <sub>inception</sub> |
| <sub>resnet_v2_101</sub> | <sub>299x299</sub> | <sub>49.2ms</sub> | <sub>83.1ms</sub> | <sub>160ms</sub> | <sub>input</sub> | <sub>resnet_v2_101/SpatialSqueeze</sub> | <sub>inception</sub> |
| <sub>resnet_v2_152</sub> | <sub>299x299</sub> | <sub>74.6ms</sub> | <sub>124ms</sub> | <sub>230ms</sub> | <sub>input</sub> | <sub>resnet_v2_152/SpatialSqueeze</sub> | <sub>inception</sub> |
| <sub>mobilenet_v1_0p25_128</sub> | <sub>128x128</sub> | <sub>2.67ms</sub> | <sub>2.65ms</sub> | <sub>15.7ms</sub> | <sub>input</sub> | <sub>MobilenetV1/Logits/SpatialSqueeze</sub> | <sub>inception</sub> |
| <sub>mobilenet_v1_0p5_160</sub> | <sub>160x160</sub> | <sub>3.95ms</sub> | <sub>4.00ms</sub> | <sub>16.9ms</sub> | <sub>input</sub> | <sub>MobilenetV1/Logits/SpatialSqueeze</sub> | <sub>inception</sub> |
| <sub>mobilenet_v1_1p0_224</sub> | <sub>224x224</sub> | <sub>12.9ms</sub> | <sub>12.9ms</sub> | <sub>24.4ms</sub> | <sub>input</sub> | <sub>MobilenetV1/Logits/SpatialSqueeze</sub> | <sub>inception</sub> |
| <sub>vgg_16</sub> | <sub>224x224</sub> | <sub>38.2ms</sub> | <sub>79.2ms</sub> | <sub>171ms</sub> | <sub>input</sub> | <sub>vgg_16/fc8/BiasAdd</sub> | <sub>vgg</sub> |

<!--| inception_v2 | 224x224 | 10.3ms | 16.9ms | 38.3ms | input | InceptionV2/Logits/SpatialSqueeze | inception |-->
<!--| vgg_19 | 224x224 | 97.3ms | OOM | input | vgg_19/fc8/BiasAdd | vgg |-->


The times recorded include data transfer to GPU, network execution, and
data transfer back from GPU.  Time does not include preprocessing. 
See **scripts/test_tf.py**, **scripts/test_trt.py**, and **src/test/test_trt.cu** 
for implementation details.  To reproduce the timings run

```
python scripts/test_tf.py
python scripts/test_trt.py
```

The timing results will be located in **data/test_output_tf.txt** and **data/test_output_trt.txt**.  Note
that you must download and convert the models (as in the quick start) prior to running the benchmark scripts.

<a name="download"></a>
## Download models and create frozen graphs

Run the following bash script to download all of the pretrained models. 

```
source scripts/download_models.sh
``` 

If there are any models you don't want to use, simply remove their URL and name from the model lists in [scripts/download_models.sh](scripts/download_models.sh).  

Next, because the TensorFlow models are provided in checkpoint format, we must convert them to frozen graphs for optimization with TensorRT.  Run the [scripts/models_to_frozen_graphs.py](scripts/models_to_frozen_graphs.py) script.  

```
python scripts/models_to_frozen_graphs.py
```

If you removed any models in the previous step, you must add ``'exclude': true`` to the corresponding item in the ``NETS`` dictionary located in [scripts/model_meta.py](scripts/model_meta.py). 

<a name="convert"></a>
## Convert frozen graph to TensorRT engine

<a name="execute"></a>
## Execute TensorRT engine

```
./build/examples/classify_image/classify_image data/images/gordon_setter.jpg data/plans/inception_v1.plan data/imagenet_labels_1001.txt input InceptionV1/Logits/SpatialSqueeze inception
```
    
<a name="benchmark"></a>
## Benchmark all models

```
python scripts/frozen_graphs_to_plans.py
```
