import pytest


def test_additivity():
    import tensorflow as tf

    from program1 import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(12345)
