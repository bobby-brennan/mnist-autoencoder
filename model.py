import tensorflow as tf
import numpy as np

class Model(object):
  def __init__(self, model_dir):
    tf.reset_default_graph()
    self.session = tf.Session()
    saver = tf.train.import_meta_graph(model_dir + '.meta')
    saver.restore(self.session, model_dir)
    self.decoded = self.session.graph.get_tensor_by_name('Decoded:0')
    self.encoded = self.session.graph.get_tensor_by_name('Encoded:0')
    self.x = self.session.graph.get_tensor_by_name('x:0')

  def encode(self, data):
    val = self.session.run(self.encoded, feed_dict={self.x: data})
    return val

  def decode(self, encodings):
    val = self.session.run(self.decoded, feed_dict={self.encoded: encodings})
    return val

