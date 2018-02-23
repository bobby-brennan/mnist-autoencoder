#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import random
import wikipedia as wiki
from autoencoder import *

MODEL_FILE = "./models/wiki/model.ckpt"

BATCH_SIZE = 500
TRAINING_STEPS = 2000000

ENCODING_SIZE = 3

START_ORD = 32
END_ORD = 126 # non-inclusive
INPUT_SIZE = END_ORD - START_ORD + 1

START_PAGE = "New York City"

def encode_text(text):
    chars = []
    for c in text:
        one_hot = np.zeros(INPUT_SIZE)
        num = ord(c) - START_ORD
        if num < 0 or num >= len(one_hot):
            num = len(one_hot) - 1
        one_hot[num] = 1
        chars.append(one_hot)
    return chars

def decode_text(encoded):
    text = ""
    for c in encoded:
        idx = np.argmax(c)
        text = text + chr(idx + START_ORD)
    return text

def run():
    x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE], name="x")
    loss, output, latent = Autoencoder.autoencoder(x, ENCODING_SIZE)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    page = wiki.page(START_PAGE)
    pages_seen = [START_PAGE]
    page_batch_index = 0
    saver = tf.train.Saver()

    def get_new_page():
        random.shuffle(pages_seen)
        source_page = wiki.page(pages_seen[0])
        if len(source_page.links) == 0:
            return get_new_page()
        random.shuffle(source_page.links)
        try:
            page = wiki.page(source_page.links[0])
        except:
            return get_new_page()
        pages_seen.append(source_page.links[0])
        return page

    # Run the training loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(int(TRAINING_STEPS + 1)):
            start = page_batch_index * BATCH_SIZE
            batch_text = page.content[start : start + BATCH_SIZE]
            page_batch_index = page_batch_index + 1
            if len(batch_text) == 0:
                page = get_new_page()
                print("Changed to page " + page.title)
                batch_text = page.content[0:BATCH_SIZE]
                page_batch_index = 0
            feed = {x : encode_text(batch_text)}
            if i % 500 == 0:
                train_loss = sess.run(loss, feed_dict=feed)
                print("step %d, training loss: %g" % (i, train_loss))
                saver.save(sess, MODEL_FILE)

            train_step.run(feed_dict=feed)

        saver.save(sess, MODEL_FILE)

def main():
    run()

if __name__ == '__main__':
    main()
