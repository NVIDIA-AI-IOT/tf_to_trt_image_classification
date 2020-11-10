import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit

# get initial meomry before importing other libraries
gpu_free, gpu_total = cuda.mem_get_info()
gpu_used_0 = (gpu_total - gpu_free)

import argparse
import tensorrt as trt
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('engine_path', type=str)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_runs', type=int, default=10)
args = parser.parse_args()

logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)

with open(args.engine_path, 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())
    
context = engine.create_execution_context()


input_gpu = gpuarray.to_gpu(np.zeros((args.batch_size, 3, 224, 224)).astype(np.float32))
output_gpu = gpuarray.to_gpu(np.zeros((args.batch_size, 1000)).astype(np.float32))

for i in range(args.num_runs):
    
    context.execute(args.batch_size, [int(input_gpu.gpudata), int(output_gpu.gpudata)])
    
gpu_free, gpu_total = cuda.mem_get_info()
gpu_used_1 = (gpu_total - gpu_free)
print('%dMB GPU MEMORY DELTA' % ((gpu_used_1 - gpu_used_0) // 1e6))

