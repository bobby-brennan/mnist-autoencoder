#!/usr/bin/env python
import sys, getopt
import numpy as np
import tensorflow as tf
from autoencoder import *

from magenta.models.image_stylization.image_utils import form_image_grid
from tensorflow.examples.tutorials.mnist import input_data

MODELS_DIR = "./models/"

BATCH_SIZE = 50
GRID_ROWS = 5
GRID_COLS = 10
TRAINING_STEPS = 2000000

ENCODING_SIZE = 2
IMAGE_SIZE = 28*28

GENERATOR_LOSS_WEIGHT = .99
DISCRIMINATOR_LOSS_WEIGHT = 1 - GENERATOR_LOSS_WEIGHT
MAX_GAN_STEPS = 5

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


def run(strategy='autoencoder', loadFromSaved=False):
    # initialize the data
    mnist = input_data.read_data_sets('/tmp/MNIST_data')

    # placeholders for the images
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")

    if strategy == 'gan':
      g_loss, d_loss, output, latent, discriminated = Autoencoder.gancoder(x)
      composite_loss = GENERATOR_LOSS_WEIGHT * g_loss - DISCRIMINATOR_LOSS_WEIGHT * d_loss
      all_vars = tf.trainable_variables()
      enc_vars = [var for var in all_vars if 'encoder_' in var.name]
      dec_vars = [var for var in all_vars if 'decoder_' in var.name]
      dis_vars = [var for var in all_vars if 'discriminator_' in var.name]
      assert len(enc_vars) + len(dec_vars) + len(dis_vars) == len(all_vars)
      gen_train_step = tf.train.AdamOptimizer(1e-4).minimize(composite_loss, var_list=(enc_vars + dec_vars))
      dis_train_step = tf.train.AdamOptimizer(1e-4).minimize(d_loss, var_list=dis_vars)
      writer, summary_op = create_gan_summaries(g_loss, d_loss, x, latent, output)
    elif strategy == 'autoencoder':
      loss, output, latent = Autoencoder.autoencoder(x)
      train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
      writer, summary_op = create_summaries(loss, x, latent, output)
    else:
        raise Exception("Unknown strategy: " + strategy)

    first_batch = mnist.train.next_batch(BATCH_SIZE)
    saver = tf.train.Saver()
    checkpoint_file = MODELS_DIR + "/" + strategy + "/model.ckpt"

    # Run the training loop
    with tf.Session() as sess:
        if loadFromSaved:
            saver.restore(sess, checkpoint_file)
        sess.run(tf.global_variables_initializer())
        sess.run(make_image("images/input.jpg", x, [28, 28]), feed_dict={x : first_batch[0]})
        for i in range(int(TRAINING_STEPS + 1)):
            batch = mnist.train.next_batch(BATCH_SIZE)
            feed = {x : batch[0]}
            if i % 500 == 0:
                if strategy == 'gan':
                  summary, g_loss_cur, d_loss_cur, discrim = sess.run([summary_op, g_loss, d_loss, discriminated], feed_dict=feed)
                  print("step %d, g loss: %g, d loss: %g" % (i, g_loss_cur, d_loss_cur))
                elif strategy == 'autoencoder':
                  summary, train_loss = sess.run([summary_op, loss], feed_dict=feed)
                  print("step %d, training loss: %g" % (i, train_loss))

                writer.add_summary(summary, i)
                writer.flush()

            if i % 1000 == 0:
                sess.run(make_image("images/output_%06i.jpg" % i, output, [28, 28]), feed_dict={x : first_batch[0]})
                saver.save(sess, checkpoint_file)

            if strategy == 'gan':
                loss_ratio = g_loss_cur / d_loss_cur
                if loss_ratio < 1:
                    d_steps = int(1 / loss_ratio)
                    g_steps = 1
                else:
                    d_steps = 1
                    g_steps = int(loss_ratio)
                for j in range(min(g_steps, MAX_GAN_STEPS)):
                    gen_train_step.run(feed_dict=feed)
                for j in range(min(d_steps, MAX_GAN_STEPS)):
                    dis_train_step.run(feed_dict=feed)
            elif strategy == 'autoencoder':
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
    opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["strategy=","reload="])
    strategy = 'autoencoder'
    loadFromSaved = False
    for opt, arg in opts:
        if opt.endswith('strategy'):
            strategy = arg
        elif opt.endswith('reload'):
            if arg.lower() == "true":
                loadFromSaved = True
    print("Running with strategy", strategy)
    if loadFromSaved:
        print("Will load from saved checkpoint")
    run(strategy, loadFromSaved)

if __name__ == '__main__':
    main()
