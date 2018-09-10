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
from model_meta import NETS, CHECKPOINT_DIR, FROZEN_GRAPHS_DIR
from convert_relu6 import convertRelu6
import os



if __name__ == '__main__':

    if not os.path.exists(CHECKPOINT_DIR):
        print("%s does not exist.  Exiting." % CHECKPOINT_DIR)
        exit()

    if not os.path.exists(FROZEN_GRAPHS_DIR):
        print("%s does not exist.  Creating it now." % FROZEN_GRAPHS_DIR)
        os.makedirs(FROZEN_GRAPHS_DIR)

    for net_name, net_meta in NETS.items():

        if 'exclude' in net_meta.keys() and net_meta['exclude'] is True:
            continue

        print("Converting %s" % net_name)
        print(net_meta)

        tf.reset_default_graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as tf_sess:
            tf_sess = tf.Session(config=tf_config)
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

            tf_saver = tf.train.Saver()
            tf_saver.restore(
                save_path=net_meta['checkpoint_filename'], 
                sess=tf_sess
            )
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                tf_sess,
                tf_sess.graph_def,
                output_node_names=net_meta['output_names']
            )

            frozen_graph = convertRelu6(frozen_graph)

            with open(net_meta['frozen_graph_filename'], 'wb') as f:
                f.write(frozen_graph.SerializeToString())
        
            f.close()
