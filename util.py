#-*- coding:utf-8 -*-
'''
@Author: li ziyu
@time:2018/9/26 0:34
'''
import tensorflow as tf
import numpy as np

def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)

def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


if __name__ == '__main__':
    x_np = np.arange(10).reshape(2, 5).astype(np.float32)
    x_tf = tf.constant(x_np)
    with tf.Session() as sess:
        print(sess.run(reduce_std(x_tf, keepdims=True)))
        print(sess.run(reduce_std(x_tf, axis=0, keepdims=True)))
        print(sess.run(reduce_std(x_tf, axis=1, keepdims=True)))
    print(np.std(x_np, keepdims=True))
    print(np.std(x_np, axis=0, keepdims=True))
    print(np.std(x_np, axis=1, keepdims=True))