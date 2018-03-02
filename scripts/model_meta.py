# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

import numpy as np
import sys
sys.path.append("third_party/models/research/")
sys.path.append("third_party/models")
sys.path.append("third_party/")
sys.path.append("third_party/models/research/slim/")
import tensorflow.contrib.slim as tf_slim
import slim.nets as nets
import slim.nets.vgg
import slim.nets.inception
import slim.nets.resnet_v1
import slim.nets.resnet_v2
import slim.nets.mobilenet_v1


def create_label_map(label_file='data/imagenet_labels_1001.txt'):
    label_map = {}
    with open(label_file, 'r') as f:
        labels = f.readlines()
        for i, label in enumerate(labels):
            label_map[i] = label
    return label_map
        

IMAGNET2012_LABEL_MAP = create_label_map()


def preprocess_vgg(image):
    return np.array(image, dtype=np.float32) - np.array([123.68, 116.78, 103.94])

def postprocess_vgg(output):
    output = output.flatten()
    predictions_top5 = np.argsort(output)[::-1][0:5]
    labels_top5 = [IMAGNET2012_LABEL_MAP[p + 1] for p in predictions_top5]
    return labels_top5

def preprocess_inception(image):
    return 2.0 * (np.array(image, dtype=np.float32) / 255.0 - 0.5)

def postprocess_inception(output):
    output = output.flatten()
    predictions_top5 = np.argsort(output)[::-1][0:5]
    labels_top5 = [IMAGNET2012_LABEL_MAP[p] for p in predictions_top5]
    return labels_top5
def mobilenet_v1_1p0_224(*args, **kwargs):
    kwargs['depth_multiplier'] = 1.0
    return nets.mobilenet_v1.mobilenet_v1(*args, **kwargs)

def mobilenet_v1_0p5_160(*args, **kwargs):
    kwargs['depth_multiplier'] = 0.5
    return nets.mobilenet_v1.mobilenet_v1(*args, **kwargs)

def mobilenet_v1_0p25_128(*args, **kwargs):
    kwargs['depth_multiplier'] = 0.25
    return nets.mobilenet_v1.mobilenet_v1(*args, **kwargs)   


CHECKPOINT_DIR = 'data/checkpoints/'
FROZEN_GRAPHS_DIR = 'data/frozen_graphs/'
# UFF_DIR = 'data/uff/'
PLAN_DIR = 'data/plans/'


NETS = {

    'vgg_16': {
        'model': nets.vgg.vgg_16,
        'arg_scope': nets.vgg.vgg_arg_scope,
        'num_classes': 1000,
        'input_name': 'input',
        'output_names': ['vgg_16/fc8/BiasAdd'],
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3, 
        'preprocess_fn': preprocess_vgg,
        'postprocess_fn': postprocess_vgg,
        'checkpoint_filename': CHECKPOINT_DIR + 'vgg_16.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'vgg_16.pb',
        'trt_convert_status': "works",
        'plan_filename': PLAN_DIR + 'vgg_16.plan'
    },

    'vgg_19': {
        'model': nets.vgg.vgg_19,
        'arg_scope': nets.vgg.vgg_arg_scope,
        'num_classes': 1000,
        'input_name': 'input',
        'output_names': ['vgg_19/fc8/BiasAdd'],
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3, 
        'preprocess_fn': preprocess_vgg,
        'postprocess_fn': postprocess_vgg,
        'checkpoint_filename': CHECKPOINT_DIR + 'vgg_19.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'vgg_19.pb',
        'trt_convert_status': "works",
        'plan_filename': PLAN_DIR + 'vgg_19.plan',
        'exclude': True
    },

    'inception_v1': {
        'model': nets.inception.inception_v1,
        'arg_scope': nets.inception.inception_v1_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['InceptionV1/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'inception_v1.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'inception_v1.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'trt_convert_status': "works",
        'plan_filename': PLAN_DIR + 'inception_v1.plan'
    },

    'inception_v2': {
        'model': nets.inception.inception_v2,
        'arg_scope': nets.inception.inception_v2_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['InceptionV2/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'inception_v2.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'inception_v2.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'trt_convert_status': "bad results",
        'plan_filename': PLAN_DIR + 'inception_v2.plan'
    },

    'inception_v3': {
        'model': nets.inception.inception_v3,
        'arg_scope': nets.inception.inception_v3_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['InceptionV3/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'inception_v3.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'inception_v3.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'trt_convert_status': "works",
        'plan_filename': PLAN_DIR + 'inception_v3.plan'
    },

    'inception_v4': {
        'model': nets.inception.inception_v4,
        'arg_scope': nets.inception.inception_v4_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['InceptionV4/Logits/Logits/BiasAdd'],
        'checkpoint_filename': CHECKPOINT_DIR + 'inception_v4.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'inception_v4.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'trt_convert_status': "works",
        'plan_filename': PLAN_DIR + 'inception_v4.plan'
    },
    
    'inception_resnet_v2': {
        'model': nets.inception.inception_resnet_v2,
        'arg_scope': nets.inception.inception_resnet_v2_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['InceptionResnetV2/Logits/Logits/BiasAdd'],
        'checkpoint_filename': CHECKPOINT_DIR + 'inception_resnet_v2_2016_08_30.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'inception_resnet_v2.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'trt_convert_status': "works",
        'plan_filename': PLAN_DIR + 'inception_resnet_v2.plan'
    },

    'resnet_v1_50': {
        'model': nets.resnet_v1.resnet_v1_50,
        'arg_scope': nets.resnet_v1.resnet_arg_scope,
        'num_classes': 1000,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['resnet_v1_50/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'resnet_v1_50.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v1_50.pb',
        'preprocess_fn': preprocess_vgg,
        'postprocess_fn': postprocess_vgg,
        'plan_filename': PLAN_DIR + 'resnet_v1_50.plan'
    },

    'resnet_v1_101': {
        'model': nets.resnet_v1.resnet_v1_101,
        'arg_scope': nets.resnet_v1.resnet_arg_scope,
        'num_classes': 1000,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['resnet_v1_101/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'resnet_v1_101.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v1_101.pb',
        'preprocess_fn': preprocess_vgg,
        'postprocess_fn': postprocess_vgg,
        'plan_filename': PLAN_DIR + 'resnet_v1_101.plan'
    },

    'resnet_v1_152': {
        'model': nets.resnet_v1.resnet_v1_152,
        'arg_scope': nets.resnet_v1.resnet_arg_scope,
        'num_classes': 1000,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['resnet_v1_152/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'resnet_v1_152.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v1_152.pb',
        'preprocess_fn': preprocess_vgg,
        'postprocess_fn': postprocess_vgg,
        'plan_filename': PLAN_DIR + 'resnet_v1_152.plan'
    },

    'resnet_v2_50': {
        'model': nets.resnet_v2.resnet_v2_50,
        'arg_scope': nets.resnet_v2.resnet_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['resnet_v2_50/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'resnet_v2_50.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v2_50.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'plan_filename': PLAN_DIR + 'resnet_v2_50.plan'
    },

    'resnet_v2_101': {
        'model': nets.resnet_v2.resnet_v2_101,
        'arg_scope': nets.resnet_v2.resnet_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['resnet_v2_101/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'resnet_v2_101.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v2_101.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'plan_filename': PLAN_DIR + 'resnet_v2_101.plan'
    },

    'resnet_v2_152': {
        'model': nets.resnet_v2.resnet_v2_152,
        'arg_scope': nets.resnet_v2.resnet_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['resnet_v2_152/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'resnet_v2_152.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v2_152.pb',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
        'plan_filename': PLAN_DIR + 'resnet_v2_152.plan'
    },

    #'resnet_v2_200': {

    #},

    'mobilenet_v1_1p0_224': {
        'model': mobilenet_v1_1p0_224,
        'arg_scope': nets.mobilenet_v1.mobilenet_v1_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['MobilenetV1/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 
            'mobilenet_v1_1.0_224.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'mobilenet_v1_1p0_224.pb',
        'plan_filename': PLAN_DIR + 'mobilenet_v1_1p0_224.plan',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
    },

    'mobilenet_v1_0p5_160': {
        'model': mobilenet_v1_0p5_160,
        'arg_scope': nets.mobilenet_v1.mobilenet_v1_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 160,
        'input_height': 160,
        'input_channels': 3,
        'output_names': ['MobilenetV1/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 
            'mobilenet_v1_0.50_160.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'mobilenet_v1_0p5_160.pb',
        'plan_filename': PLAN_DIR + 'mobilenet_v1_0p5_160.plan',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
    },

    'mobilenet_v1_0p25_128': {
        'model': mobilenet_v1_0p25_128,
        'arg_scope': nets.mobilenet_v1.mobilenet_v1_arg_scope,
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 128,
        'input_height': 128,
        'input_channels': 3,
        'output_names': ['MobilenetV1/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 
            'mobilenet_v1_0.25_128.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'mobilenet_v1_0p25_128.pb',
        'plan_filename': PLAN_DIR + 'mobilenet_v1_0p25_128.plan',
        'preprocess_fn': preprocess_inception,
        'postprocess_fn': postprocess_inception,
    },
}


