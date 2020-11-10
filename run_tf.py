import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit

# get initial meomry before importing other libraries
gpu_free, gpu_total = cuda.mem_get_info()
gpu_used_0 = (gpu_total - gpu_free)

import argparse
import numpy as np
import tensorflow as tf
import time

parser = argparse.ArgumentParser()
parser.add_argument('frozen_graph', type=str)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_runs', type=int, default=10)
parser.add_argument('--allow_growth', action='store_true')
args = parser.parse_args()

# LOAD MODEL
with open(args.frozen_graph, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")
    
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = args.allow_growth  # disable upfront memory allocation
tf_config.allow_soft_placement = True

if 'vgg_16' in args.frozen_graph:
    output_name = 'vgg_16/fc8/BiasAdd'
elif 'vgg_19' in args.frozen_graph:
    output_name = 'vgg_19/fc8/BiasAdd'
elif 'inception_v1' in args.frozen_graph:
    output_name = 'InceptionV1/Logits/SpatialSqueeze'
elif 'inception_v2' in args.frozen_graph:
    output_name = 'InceptionV2/Logits/SpatialSqueeze'
elif 'resnet_v1_50' in args.frozen_graph:
    output_name = 'resnet_v1_50/SpatialSqueeze'
elif 'resnet_v1_101' in args.frozen_graph:
    output_name = 'resnet_v1_101/SpatialSqueeze'
elif 'resnet_v1_152' in args.frozen_graph:
    output_name = 'resnet_v1_152/SpatialSqueeze'
elif 'mobilenet_v1_1p0_224' in args.frozen_graph:
    output_name = 'MobilenetV1/Logits/SpatialSqueeze'
else:
    raise RuntimeError('Could not find output name for model.')


with tf.Session(config=tf_config, graph=graph) as tf_sess:
    tf_input = tf_sess.graph.get_tensor_by_name('input' + ':0')
    tf_output = tf_sess.graph.get_tensor_by_name(output_name + ':0')
            
            
    input = np.zeros((args.batch_size, 224, 224, 3)).astype(np.float32)

    for i in range(args.num_runs):

        output = tf_sess.run([tf_output], feed_dict={
                            tf_input: input
        })[0]

    gpu_free, gpu_total = cuda.mem_get_info()
    gpu_used_1 = (gpu_total - gpu_free)
    print('%dMB GPU MEMORY DELTA' % ((gpu_used_1 - gpu_used_0) // 1e6))

