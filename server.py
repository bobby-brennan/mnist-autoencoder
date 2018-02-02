import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

BATCH_SIZE = 50
MODEL = "./model/model.ckpt"

mnist = input_data.read_data_sets('/tmp/MNIST_data')
first_batch = mnist.train.next_batch(BATCH_SIZE)

loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
  saver = tf.train.import_meta_graph('./model/model.ckpt.meta')
  saver.restore(sess, MODEL)
  decoded = loaded_graph.get_tensor_by_name('Decoded:0')
  encoded = loaded_graph.get_tensor_by_name('Encoded:0')
  x = loaded_graph.get_tensor_by_name('x:0')

  def encode(image):
    val = sess.run(encoded, feed_dict={x: [image]})
    return val[0]
  def decode(encoding):
    val = sess.run(decoded, feed_dict={encoded: [encoding]})
    return val[0]

  print decode([.5, .5]);
