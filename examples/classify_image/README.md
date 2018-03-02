# Classify Single Image 

This example demonstrates how to classify a single image using one of the TensorFlow pretrained
models converted to TensorRT.


## Running the Example

**If you haven't already, follow the installation instructions [here](../../INSTALL.md).**

### Convert Frozen Model to PLAN

Assuming you have a trained and frozen image classification model, convert it to a plan
using the following convert_plan script.

```
python scripts/convert_plan.py data/frozen_graphs/inception_v1.pb data/plans/inception_v1.plan input 224 224 InceptionV1/Logits/SpatialSqueeze 1 1048576 float
```

For reference, the inputs to the convert_plan.py script are

1. frozen graph path
2. output plan path
3. input node name
4. input height
5. input width
6. output node name
7. max batch size
8. max workspace size
9. data type (float or half)

### Run the example program

Once the plan file is generated, run the example Cpp/CUDA program to classify the image.

```
./build/examples/classify_image/classify_image data/images/gordon_setter.jpg data/plans/inception_v1.plan data/imagenet_labels_1001.txt input InceptionV1/Logits/SpatialSqueeze inception
```

You should see that the most probable index is 215, which using our label file corresponds to
a "Gordon setter".  For reference, the inputs to the classify_image example are

1. input image path
2. plan file path
3. labels file (one label per line, line number corresponds to index in output)
4. input node name
5. output node name
6. preprocessing function (either vgg or inception. see: [this](../../data/DEFAULT_NETS.md))

To use other networks, supply the classify_image executable with arguments corresponding to the
default networks table ([link](../../data/DEFAULT_NETS.md)).  You will need to generate the 
corresponding PLAN file as well.

