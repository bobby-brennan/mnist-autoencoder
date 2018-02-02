import tensorflow as tf
import numpy as np

MODEL_DIR = "./model/model.ckpt"

class Model(object):
  def __init__(self):
    tf.reset_default_graph()
    self.session = tf.Session()
    saver = tf.train.import_meta_graph(MODEL_DIR + '.meta')
    saver.restore(self.session, MODEL_DIR)
    self.decoded = self.session.graph.get_tensor_by_name('Decoded:0')
    self.encoded = self.session.graph.get_tensor_by_name('Encoded:0')
    self.x = self.session.graph.get_tensor_by_name('x:0')

  def encode(self, image):
    image = np.reshape(image, [28*28])
    val = self.session.run(self.encoded, feed_dict={self.x: [image]})
    return val[0]

  def decode(self, encoding):
    val = self.session.run(self.decoded, feed_dict={self.encoded: [encoding]})
    return np.reshape(val[0], [28, 28])

