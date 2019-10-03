import pytest
import tensorflow as tf

import syft


def test_keras_layer_class():
    hook = syft.TensorFlowHook(tf)
    bob = syft.VirtualWorker(hook, id="bob")

    x_to_give = tf.constant([-2.0, 3.0, 5.0])
    layer_to_give = tf.keras.layers.Activation('linear')
    expected = layer_to_give(x_to_give)

    x_ptr = x_to_give.send(bob)
    layer_ptr = layer_to_give.send(bob)

    res_ptr = layer_ptr(x_ptr)
    actual = res_ptr.get()

    assert tf.math.equal(actual, expected).numpy().all()


def test_keras_sequential_class():
    hook = syft.TensorFlowHook(tf)
    bob = syft.VirtualWorker(hook, id="bob")

    x_to_give = tf.constant([-2.0, 3.0, 5.0])
    model_to_give = tf.keras.Sequential(
        tf.keras.layers.Activation('linear'))

    expected = model_to_give(x_to_give)

    x_ptr = x_to_give.send(bob)
    model_ptr = model_to_give.send(bob)

    res_ptr = model_ptr(x_ptr)
    actual = res_ptr.get()

    assert tf.math.equal(actual, expected).numpy().all()
