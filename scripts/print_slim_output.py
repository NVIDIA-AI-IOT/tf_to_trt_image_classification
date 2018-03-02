# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

import tensorflow as tf
import sys
sys.path.append("third_party/models/research/")
sys.path.append("third_party/models")
sys.path.append("third_party/")
sys.path.append("third_party/models/research/slim/")
sys.path.append("scripts")
import tensorflow.contrib.slim as tf_slim
import slim.nets as nets
import slim.nets.vgg
from model_meta import NETS


if __name__ == '__main__':

    with open("data/output_names.txt", 'w') as f:
        for net_name, net_meta in NETS.items():

            tf.reset_default_graph()
            tf_sess = tf.Session()
            tf_input = tf.placeholder(
                tf.float32, 
                (
                    None, 
                    net_meta['input_height'], 
                    net_meta['input_width'], 
                    net_meta['input_channels']
                ),
                name=net_meta['input_name']
            )

            with tf_slim.arg_scope(net_meta['arg_scope']()):
                tf_net, tf_end_points = net_meta['model'](
                    tf_input, 
                    is_training=False,
                    num_classes=net_meta['num_classes']
                )
                print("Output name for %s is %s" % (net_name, tf_net.name))
            f.write("%s\t%s\n" % (net_name, tf_net.name))
        f.close()
