# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

import sys
sys.path.append('third_party/models/')
sys.path.append('third_party/models/research')
sys.path.append('third_party/models/research/slim')
import uff
from model_meta import NETS, FROZEN_GRAPHS_DIR, CHECKPOINT_DIR, UFF_DIR


if __name__ == '__main__':

    for net_name, net_meta in NETS.items():
        
        if 'exclude' in net_meta.keys() and net_meta['exclude'] is True:
            continue

        print("Convertings %s to UFF" % net_name)
        
        uff_model = uff.from_tensorflow_frozen_model(
            frozen_file=net_meta['frozen_graph_filename'],
            output_nodes=net_meta['output_names'],
            output_filename=net_meta['uff_filename'],
            text=False
        )
