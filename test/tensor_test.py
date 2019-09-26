import pytest
import tensorflow as tf

import syft


def test_send_get_constant(remote):
    x_to_give = tf.constant(2.0)
    x_ptr = x_to_give.send(remote)
    x_gotten = x_ptr.get()
    assert tf.math.equal(x_to_give, x_gotten)

def test_add(remote):
  x = tf.constant([3.0, 3.0]).send(remote)
  y = tf.constant([2.0, 2.0]).send(remote)

  z = x + y
  assert z.get().equal(tf.constant([5.0, 5.0]))
