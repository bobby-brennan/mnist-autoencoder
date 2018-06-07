#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import random
from autoencoder import *
from wiki import *

MODEL_FILE = "./models/rnn/model.ckpt"

BATCH_SIZE = 500
TRAINING_STEPS = 2000000
HIDDEN_SIZE = 50

STR_SIZE = 10
SEQ_MAX_LENGTH = 30
N_CLASSES = 2

def structured_string(length):
    text = ""
    for i in range(length):
        if random.random() < .5:
            text = text + "abc"
        else:
            text = text + "eee"
    return text

def random_string(length):
    text = ""
    for i in range(length):
      rand = random.random()
      if rand > .5:
          text = text + "e"
      elif rand < .1667:
          text = text + "a"
      elif rand < 2 * .1667:
          text = text + "b"
      else:
          text = text + "c"
    return text

def get_data(batch_size, str_size):
    data = []
    labels = []
    lens = []
    for i in range(batch_size):
        lens.append(str_size)
        if random.random() < .5:
            data.append(encode_text(structured_string(str_size)))
            labels.append([0, 1])
        else:
            data.append(encode_text(random_string(str_size * 3)))
            labels.append([1, 0])
    return data, labels, lens

def dynamic_rnn(x, seqlen):
    x = tf.unstack(x, SEQ_MAX_LENGTH, 1)
    w = tf.Variable(tf.random_normal([HIDDEN_SIZE, N_CLASSES]))
    b = tf.Variable(tf.random_normal([N_CLASSES]))
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x,
                                                dtype=tf.float32,
                                                sequence_length=seqlen)
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])
    batch_size = tf.shape(outputs)[0]
    index = tf.range(0, batch_size) * SEQ_MAX_LENGTH + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, HIDDEN_SIZE]), index)
    return tf.add(tf.matmul(outputs, w), b)


def run():
    x = tf.placeholder(tf.float32, shape=[None, SEQ_MAX_LENGTH, INPUT_SIZE], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 2], name="y")
    seqlen = tf.placeholder(tf.int32, [None])
    pred = dynamic_rnn(x, seqlen, w, b)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(int(TRAINING_STEPS + 1)):
            data, labels, lens = get_data(BATCH_SIZE, STR_SIZE)
            feed = {x: data, y: labels, seqlen: lens}
            if i % 2 == 0:
                train_loss, acc = sess.run([loss, accuracy], feed_dict=feed)
                print("step %d, training loss: %g, %g" % (i, train_loss, acc))
                saver.save(sess, MODEL_FILE)

            train_step.run(feed_dict=feed)

        saver.save(sess, MODEL_FILE)

def main():
    run()

if __name__ == '__main__':
    main()
