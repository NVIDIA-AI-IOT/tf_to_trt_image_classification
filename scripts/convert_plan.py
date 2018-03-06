# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

import os
import subprocess
import uff
import pdb
import sys

UFF_TO_PLAN_EXE_PATH = 'build/src/uff_to_plan'
TMP_UFF_FILENAME = 'data/tmp.uff'


def frozenToPlan(frozen_graph_filename, plan_filename, input_name, input_height, 
        input_width, output_name, max_batch_size, max_workspace_size, data_type):

    # generate uff from frozen graph
    uff_model = uff.from_tensorflow_frozen_model(
        frozen_file=frozen_graph_filename,
        output_nodes=[output_name],
        output_filename=TMP_UFF_FILENAME,
        text=False,
    )

    # convert frozen graph to engine (plan)
    args = [
        TMP_UFF_FILENAME,
        plan_filename,
        input_name,
        str(input_height),
        str(input_width),
        output_name,
        str(max_batch_size),
        str(max_workspace_size), 
        data_type # float / half
    ]
    subprocess.call([UFF_TO_PLAN_EXE_PATH] + args)

    # cleanup tmp file
    os.remove(TMP_UFF_FILENAME)


if __name__ == '__main__':

    if not os.path.exists('data/plans'):
        os.makedirs('data/plans')

    if len(sys.argv) is not 10:
        print("usage: python convert_plan.py <frozen_graph_path> <output_plan_path> <input_name> <input_height>"
              " <input_width> <output_name> <max_batch_size> <max_workspace_size> <data_type>")
        exit()

    frozen_graph_filename = sys.argv[1]
    plan_filename = sys.argv[2]
    input_name = sys.argv[3]
    input_height = sys.argv[4]
    input_width = sys.argv[5]
    output_name = sys.argv[6]
    max_batch_size = sys.argv[7]
    max_workspace_size = sys.argv[8]
    data_type = sys.argv[9]
    
    frozenToPlan(frozen_graph_filename,
        plan_filename,
        input_name,
        input_height,
        input_width,
        output_name,
        max_batch_size,
        max_workspace_size,
        data_type
    )
