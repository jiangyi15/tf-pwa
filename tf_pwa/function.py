import tensorflow as tf


def nll_funciton(f, data, phsp):
    """
    nagtive log likelihood for minimize

    """

    def g():
        ll = tf.math.log(f(data))
        int_mc = tf.reduce_mean(f(phsp))
        return -tf.reduce_sum(ll) + ll.shape[0] * tf.math.log(int_mc)

    return g
