# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

import sys
sys.path.append('third_party/models/')
sys.path.append('third_party/models/research')
sys.path.append('third_party/models/research/slim')
import uff
from model_meta import NETS, FROZEN_GRAPHS_DIR, CHECKPOINT_DIR, PLAN_DIR
from convert_plan import frozenToPlan
import os



if __name__ == '__main__':

    if not os.path.exists('data/plans'):
        os.makedirs('data/plans')

    for net_name, net_meta in NETS.items():
        
        if 'exclude' in net_meta.keys() and net_meta['exclude'] is True:
            continue

        print("Convertings %s to PLAN" % net_name)
         
        frozenToPlan(net_meta['frozen_graph_filename'],
            net_meta['plan_filename'],
            net_meta['input_name'],
            net_meta['input_height'],
            net_meta['input_width'],
            net_meta['output_names'][0],
            1, # batch size
            1 << 20, # workspace size
            'half' # data type
        )
