# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

CHECKPOINT_DIR = 'data/checkpoints/'
FROZEN_GRAPHS_DIR = 'data/frozen_graphs/'
UFF_DIR = 'data/uff/'

NETS = {

    'vgg_16': {
        'num_classes': 1000,
        'input_name': 'input',
        'output_names': ['vgg_16/fc8/BiasAdd'],
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3, 
        'preprocess_fn': 'preprocess_vgg',
        'checkpoint_filename': CHECKPOINT_DIR + 'vgg_16.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'vgg_16.pb',
        'trt_convert_status': "works",
        'uff_filename': UFF_DIR + 'vgg_16.uff'
    },

    'vgg_19': {
        'num_classes': 1000,
        'input_name': 'input',
        'output_names': ['vgg_19/fc8/BiasAdd'],
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3, 
        'preprocess_fn': 'preprocess_vgg',
        'checkpoint_filename': CHECKPOINT_DIR + 'vgg_19.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'vgg_19.pb',
        'trt_convert_status': "works",
        'uff_filename': UFF_DIR + 'vgg_19.uff'
    },

    'inception_v1': {
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['InceptionV1/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'inception_v1.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'inception_v1.pb',
        'preprocess_fn': 'preprocess_inception',
        'trt_convert_status': "works",
        'uff_filename': UFF_DIR + 'inception_v1.uff'
    },

    'inception_v2': {
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['InceptionV2/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'inception_v2.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'inception_v2.pb',
        'preprocess_fn': 'preprocess_inception',
        'trt_convert_status': "bad results",
        'uff_filename': UFF_DIR + 'inception_v2.uff'
    },

    'inception_v3': {
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['InceptionV3/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'inception_v3.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'inception_v3.pb',
        'preprocess_fn': 'preprocess_inception',
        'trt_convert_status': "works",
        'uff_filename': UFF_DIR + 'inception_v3.uff'
    },

    'inception_v4': {
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['InceptionV4/Logits/Logits/BiasAdd'],
        'checkpoint_filename': CHECKPOINT_DIR + 'inception_v4.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'inception_v4.pb',
        'preprocess_fn': 'preprocess_inception',
        'trt_convert_status': "works",
        'uff_filename': UFF_DIR + 'inception_v4.uff'
    },
    
    'inception_resnet_v2': {
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['InceptionResnetV2/Logits/Logits/BiasAdd'],
        'checkpoint_filename': CHECKPOINT_DIR + 'inception_resnet_v2_2016_08_30.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'inception_resnet_v2.pb',
        'preprocess_fn': 'preprocess_inception',
        'trt_convert_status': "works",
        'uff_filename': UFF_DIR + 'inception_resnet_v2.uff'
    },

    'resnet_v1_50': {
        'num_classes': 1000,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['resnet_v1_50/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'resnet_v1_50.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v1_50.pb',
        'preprocess_fn': 'preprocess_vgg',
        'uff_filename': UFF_DIR + 'resnet_v1_50.uff'
    },

    'resnet_v1_101': {
        'num_classes': 1000,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['resnet_v1_101/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'resnet_v1_101.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v1_101.pb',
        'preprocess_fn': 'preprocess_vgg',
        'uff_filename': UFF_DIR + 'resnet_v1_101.uff'
    },

    'resnet_v1_152': {
        'num_classes': 1000,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['resnet_v1_152/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'resnet_v1_152.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v1_152.pb',
        'preprocess_fn': 'preprocess_vgg',
        'uff_filename': UFF_DIR + 'resnet_v1_152.uff'
    },

    'resnet_v2_50': {
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['resnet_v2_50/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'resnet_v2_50.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v2_50.pb',
        'preprocess_fn': 'preprocess_inception',
        'uff_filename': UFF_DIR + 'resnet_v2_50.uff'
    },

    'resnet_v2_101': {
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['resnet_v2_101/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'resnet_v2_101.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v2_101.pb',
        'preprocess_fn': 'preprocess_inception',
        'uff_filename': UFF_DIR + 'resnet_v2_101.uff'
    },

    'resnet_v2_152': {
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 299,
        'input_height': 299,
        'input_channels': 3,
        'output_names': ['resnet_v2_152/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 'resnet_v2_152.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'resnet_v2_152.pb',
        'preprocess_fn': 'preprocess_inception',
        'uff_filename': UFF_DIR + 'resnet_v2_152.uff'
    },

    #'resnet_v2_200': {

    #},

    'mobilenet_v1_1p0_224': {
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 224,
        'input_height': 224,
        'input_channels': 3,
        'output_names': ['MobilenetV1/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 
            'mobilenet_v1_1.0_224.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'mobilenet_v1_1p0_224.pb',
        'uff_filename': UFF_DIR + 'mobilenet_v1_1p0_224.uff',
        'preprocess_fn': 'preprocess_inception',
    },

    'mobilenet_v1_0p5_160': {
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 160,
        'input_height': 160,
        'input_channels': 3,
        'output_names': ['MobilenetV1/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 
            'mobilenet_v1_0.50_160.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'mobilenet_v1_0p5_160.pb',
        'uff_filename': UFF_DIR + 'mobilenet_v1_0p5_160.uff',
        'preprocess_fn': 'preprocess_inception',
    },

    'mobilenet_v1_0p25_128': {
        'num_classes': 1001,
        'input_name': 'input',
        'input_width': 128,
        'input_height': 128,
        'input_channels': 3,
        'output_names': ['MobilenetV1/Logits/SpatialSqueeze'],
        'checkpoint_filename': CHECKPOINT_DIR + 
            'mobilenet_v1_0.25_128.ckpt',
        'frozen_graph_filename': FROZEN_GRAPHS_DIR + 'mobilenet_v1_0p25_128.pb',
        'uff_filename': UFF_DIR + 'mobilenet_v1_0p25_128.uff',
        'preprocess_fn': 'preprocess_inception',
    },
}


