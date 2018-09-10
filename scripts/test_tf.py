# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

import sys
sys.path.append('third_party/models/')
sys.path.append('third_party/models/research')
sys.path.append('third_party/models/research/slim')
#from PIL import Image
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from model_meta import NETS, FROZEN_GRAPHS_DIR, CHECKPOINT_DIR
import time
import cv2


TEST_IMAGE_PATH='data/images/gordon_setter.jpg'
TEST_OUTPUT_PATH='data/test_output_tf.txt'
NUM_RUNS=50

if __name__ == '__main__':

    with open(TEST_OUTPUT_PATH, 'w') as test_f:
        for net_name, net_meta in NETS.items():

            if 'exclude' in net_meta.keys() and net_meta['exclude'] is True:
                continue

            print("Testing %s" % net_name)

            with open(net_meta['frozen_graph_filename'], 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name="")
            
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            tf_config.allow_soft_placement = True

            with tf.Session(config=tf_config, graph=graph) as tf_sess:
                tf_input = tf_sess.graph.get_tensor_by_name(net_meta['input_name'] + ':0')
                tf_output = tf_sess.graph.get_tensor_by_name(net_meta['output_names'][0] + ':0')

                # load and preprocess image
                image = cv2.imread(TEST_IMAGE_PATH)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (net_meta['input_width'], net_meta['input_height']))
                image = net_meta['preprocess_fn'](image)


                # run network
                times = []
                for i in range(NUM_RUNS + 1):
                    t0 = time.time()
                    output = tf_sess.run([tf_output], feed_dict={
                        tf_input: image[None, ...]
                    })[0]
                    t1 = time.time()
                    times.append(1000 * (t1 - t0))
                avg_time = np.mean(times[1:]) # don't include first run

                # parse output
                top5 = net_meta['postprocess_fn'](output)
                print(top5)
                test_f.write("%s %s\n" % (net_name, avg_time))
