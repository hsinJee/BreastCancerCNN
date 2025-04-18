import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import tensorflow as tf
tf.config.experimental.get_memory_info('GPU:0')
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))