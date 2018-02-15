#!/usr/bin/env python

"""TensorFlow MNIST AutoEncoder

This is my attempt to write the autoencoder for MNIST by Andrej Karpathy using 
ConvNetJS in TensorFlow. Mostly to get some more experience working in 
Tensorflow.

Sources:
    - http://cs.stanford.edu/people/karpathy/convnetjs/demo/autoencoder.html
    - https://www.tensorflow.org/get_started/mnist/pros

Author: Gertjan van den Burg
Date: Thu Oct 26 16:49:29 CEST 2017

"""

import numpy as np
import tensorflow as tf

from magenta.models.image_stylization.image_utils import form_image_grid
from tensorflow.examples.tutorials.mnist import input_data

MODEL_FILE = "./model/model.ckpt"
GAN_MODEL_FILE = "./gan_model/model.ckpt"

BATCH_SIZE = 50
GRID_ROWS = 5
GRID_COLS = 10
TRAINING_STEPS = 200000

ENCODING_SIZE = 2
IMAGE_SIZE = 28*28

def weight_variable(shape):
    # From the mnist tutorial
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def fc_layer(previous, input_size, output_size, name=None):
    W = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    return tf.add(tf.matmul(previous, W), b, name=name)

def encoder(x):
    # first fully connected layer with 50 neurons using tanh activation
    l1 = tf.nn.tanh(fc_layer(x, IMAGE_SIZE, 50))
    # second fully connected layer with 50 neurons using tanh activation
    l2 = tf.nn.tanh(fc_layer(l1, 50, 50))
    # third fully connected layer with 2 neurons
    l3 = tf.nn.tanh(fc_layer(l2, 50, ENCODING_SIZE), name="Encoded")
    return l3

def decoder(encoded):
    # fourth fully connected layer with 50 neurons and tanh activation
    l4 = tf.nn.tanh(fc_layer(encoded, ENCODING_SIZE, 50))
    # fifth fully connected layer with 50 neurons and tanh activation
    l5 = tf.nn.tanh(fc_layer(l4, 50, 50))
    # readout layer
    out = tf.nn.relu(fc_layer(l5, 50, IMAGE_SIZE), name="Decoded")
    return out

def discriminator(x):
    l1 = tf.nn.tanh(fc_layer(x, IMAGE_SIZE, 50))
    l2 = tf.nn.tanh(fc_layer(l1, 50, 50))
    l3 = tf.nn.tanh(fc_layer(l2, 50, 1, "Discriminated"))
    return l3

def autoencoder(x):
    encoded = encoder(x)
    decoded = decoder(encoded)
    # let's use an l2 loss on the output image
    loss = tf.reduce_mean(tf.squared_difference(x, decoded), name="Loss")
    return loss, decoded, encoded

def gancoder(x, fake_encoded):
    encoded = encoder(x)
    decoded = decoder(encoded)
    decoded_fake = decoder(fake_encoded)
    discriminated = discriminator(decoded)
    discriminated_fake = discriminator(decoded_fake)
    discriminator_loss_real = tf.reduce_mean(tf.abs(discriminated - 1))
    discriminator_loss_fake = tf.reduce_mean(tf.abs(discriminated_fake + 1))
    discriminator_loss = (discriminator_loss_real + discriminator_loss_fake) / 4
    generator_loss = tf.reduce_mean(tf.squared_difference(x, decoded), name="GenLoss")
    #discriminator_loss_real = tf.reduce_mean(tf.squared_difference(discriminated, tf.ones([BATCH_SIZE])))
    #discriminator_loss_fake = tf.reduce_mean(tf.squared_difference(discriminated_fake, tf.zeros([BATCH_SIZE])))
    #discriminator_loss = discriminator_loss_real + discriminator_loss_fake
    return generator_loss, discriminator_loss, decoded, encoded

def layer_grid_summary(name, var, image_dims):
    prod = np.prod(image_dims)
    grid = form_image_grid(tf.reshape(var, [BATCH_SIZE, prod]), [GRID_ROWS, GRID_COLS], image_dims, 1)
    return tf.summary.image(name, grid)

def create_summaries(loss, x, latent, output):
    writer = tf.summary.FileWriter("./logs")
    tf.summary.scalar("Loss", loss)
    layer_grid_summary("Input", x, [28, 28])
    layer_grid_summary("Encoder", latent, [ENCODING_SIZE, 1])
    layer_grid_summary("Output", output, [28, 28])
    return writer, tf.summary.merge_all()

def create_gan_summaries(g_loss, d_loss, x, latent, output):
    writer = tf.summary.FileWriter("./logs")
    tf.summary.scalar("GenLoss", g_loss)
    tf.summary.scalar("DisLoss", d_loss)
    layer_grid_summary("Input", x, [28, 28])
    layer_grid_summary("Encoder", latent, [ENCODING_SIZE, 1])
    layer_grid_summary("Output", output, [28, 28])
    return writer, tf.summary.merge_all()

def make_image(name, var, image_dims):
    prod = np.prod(image_dims)
    grid = form_image_grid(tf.reshape(var, [BATCH_SIZE, prod]), [GRID_ROWS, 
        GRID_COLS], image_dims, 1)
    s_grid = tf.squeeze(grid, axis=0)

    # This reproduces the code in: tensorflow/core/kernels/summary_image_op.cc
    im_min = tf.reduce_min(s_grid)
    im_max = tf.reduce_max(s_grid)

    kZeroThreshold = tf.constant(1e-6)
    max_val = tf.maximum(tf.abs(im_min), tf.abs(im_max))

    offset = tf.cond(
            im_min < tf.constant(0.0),
            lambda: tf.constant(128.0),
            lambda: tf.constant(0.0)
            )
    scale = tf.cond(
            im_min < tf.constant(0.0),
            lambda: tf.cond(
                max_val < kZeroThreshold,
                lambda: tf.constant(0.0),
                lambda: tf.div(127.0, max_val)
                ),
            lambda: tf.cond(
                im_max < kZeroThreshold,
                lambda: tf.constant(0.0),
                lambda: tf.div(255.0, im_max)
                )
            )
    s_grid = tf.cast(tf.add(tf.multiply(s_grid, scale), offset), tf.uint8)
    enc = tf.image.encode_jpeg(s_grid)

    fwrite = tf.write_file(name, enc)
    return fwrite


def run(gan=False):
    # initialize the data
    mnist = input_data.read_data_sets('/tmp/MNIST_data')

    # placeholders for the images
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    rand = tf.placeholder(tf.float32, shape=[None, ENCODING_SIZE], name="rand")

    if gan:
      g_loss, d_loss, output, latent = gancoder(x, rand)
      g_train_step = tf.train.AdamOptimizer(1e-4).minimize(g_loss - 10 * tf.minimum(d_loss, .5))
      d_train_step = tf.train.AdamOptimizer(1e-4).minimize(d_loss)
      writer, summary_op = create_gan_summaries(g_loss, d_loss, x, latent, output)
    else:
      loss, output, latent = autoencoder(x)
      train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
      writer, summary_op = create_summaries(loss, x, latent, output)

    first_batch = mnist.train.next_batch(BATCH_SIZE)
    saver = tf.train.Saver()

    # Run the training loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(make_image("images/input.jpg", x, [28, 28]), feed_dict={x : first_batch[0]})
        for i in range(int(TRAINING_STEPS + 1)):
            batch = mnist.train.next_batch(BATCH_SIZE)
            rands = np.random.uniform(size=[BATCH_SIZE, ENCODING_SIZE], low=-1.0, high=1.0)
            feed = {x : batch[0], rand : rands}
            if i % 500 == 0:
                if gan:
                  summary, g_loss_cur, d_loss_cur = sess.run([summary_op, g_loss, d_loss], feed_dict=feed)
                  print("step %d, g loss: %g, d loss: %g" % (i, g_loss_cur, d_loss_cur))
                else:
                  summary, train_loss = sess.run([summary_op, loss], feed_dict=feed)
                  print("step %d, training loss: %g" % (i, train_loss))

                writer.add_summary(summary, i)
                writer.flush()

            if i % 1000 == 0:
                sess.run(make_image("images/output_%06i.jpg" % i, output, [28, 28]), feed_dict={x : first_batch[0]})
                saver.save(sess, GAN_MODEL_FILE if gan else MODEL_FILE)

            if gan:
              g_train_step.run(feed_dict=feed)
              d_train_step.run(feed_dict=feed)
            else:
              train_step.run(feed_dict=feed)

        # Save latent space
        pred = sess.run(latent, feed_dict={x : mnist.test._images})
        pred = np.asarray(pred)
        pred = np.reshape(pred, (mnist.test._num_examples, ENCODING_SIZE))
        labels = np.reshape(mnist.test._labels, (mnist.test._num_examples, 1))
        pred = np.hstack((pred, labels))
        np.savetxt("latent_relu.csv", pred)
        saver.save(sess, GAN_MODEL_FILE if gan else MODEL_FILE)

def main():
  run(True)

if __name__ == '__main__':
    main()
