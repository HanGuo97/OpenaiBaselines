from tensorflow.python.ops import array_ops


def concatenate(arrs, axis=0):
    return array_ops.concat(axis=axis, values=arrs)
