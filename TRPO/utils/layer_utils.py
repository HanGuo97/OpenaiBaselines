import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.layers import core as layers_core



def dense(*args, **kargs):
    return layers_core.dense(*args, **kargs)


def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return constant_op.constant(out)
    return _initializer
