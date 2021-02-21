import tensorflow as tf
class Config:
    device = "gpu" if tf.test.is_gpu_available() else "cpu"
