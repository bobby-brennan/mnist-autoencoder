import tensorflow as tf

HIDDEN_SIZE = 40

class Autoencoder:
    @staticmethod
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    @staticmethod
    def fc_layer(previous, input_size, output_size, name):
        W = Autoencoder.weight_variable([input_size, output_size], name + '_W')
        b = Autoencoder.bias_variable([output_size], name + '_b')
        return tf.add(tf.matmul(previous, W), b, name=name)

    @staticmethod
    def encoder(x, encoding_size):
        l1 = tf.nn.tanh(Autoencoder.fc_layer(x, x.get_shape().as_list()[1], HIDDEN_SIZE, 'encoder_1'))
        l2 = tf.nn.tanh(Autoencoder.fc_layer(l1, HIDDEN_SIZE, HIDDEN_SIZE, 'encoder_2'))
        l3 = tf.nn.tanh(Autoencoder.fc_layer(l2, HIDDEN_SIZE, encoding_size, 'encoder_3'), name="Encoded")
        return l3

    @staticmethod
    def decoder(encoded, out_size, relu=True):
        l4 = tf.nn.tanh(Autoencoder.fc_layer(encoded, encoded.get_shape().as_list()[1], HIDDEN_SIZE, 'decoder_1'))
        l5 = tf.nn.tanh(Autoencoder.fc_layer(l4, HIDDEN_SIZE, HIDDEN_SIZE, 'decoder_2'))
        if relu:
            out = tf.nn.relu(Autoencoder.fc_layer(l5, HIDDEN_SIZE, out_size, 'decoder_3'), name='Decoded')
        else:
            out = Autoencoder.fc_layer(l5, HIDDEN_SIZE, out_size, 'Decoded')
        return out

    @staticmethod
    def discriminator(x):
        l1 = tf.nn.tanh(Autoencoder.fc_layer(x, x.get_shape().as_list()[1], HIDDEN_SIZE, 'discriminator_1'))
        l2 = tf.nn.tanh(Autoencoder.fc_layer(l1, HIDDEN_SIZE, HIDDEN_SIZE, 'discriminator_2',))
        l3 = tf.nn.tanh(Autoencoder.fc_layer(l2, HIDDEN_SIZE, 1, 'discriminator_3'), name="Discriminated")
        return l3

    @staticmethod
    def autoencoder(x, encoding_size=2, cross_entropy=False):
        encoded = Autoencoder.encoder(x, encoding_size)
        decoded = Autoencoder.decoder(encoded, x.get_shape().as_list()[1], not cross_entropy)
        if cross_entropy:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=x, logits=decoded), name="Loss")
        else:
            loss = tf.reduce_mean(tf.squared_difference(x, decoded), name="Loss")
        return loss, decoded, encoded

    @staticmethod
    def add_noise(x, mean, stddev):
        noise = tf.random_normal(shape=tf.shape(x), mean=mean, stddev=stddev)
        return x + noise

    @staticmethod
    def gancoder(x, encoding_size=2):
        encoded = Autoencoder.encoder(x, encoding_size)
        decoded = Autoencoder.decoder(encoded, x.get_shape().as_list()[1])

        to_discriminate = tf.concat([x, decoded], 0)
        discriminated = Autoencoder.discriminator(to_discriminate)
        num_inputs = tf.shape(x)[0]
        num_decodes = tf.shape(decoded)[0]
        discrimination_labels = tf.concat([tf.ones([num_inputs, 1]), 0 - tf.ones([num_decodes, 1])], 0)

        discriminator_loss = tf.reduce_mean(tf.squared_difference(discrimination_labels, discriminated))
        generator_loss = tf.reduce_mean(tf.squared_difference(x, decoded), name="GenLoss")
        return generator_loss, discriminator_loss, decoded, encoded, discriminated

    @staticmethod
    def rnn(x, seqlen, maxlen, output_size, name):
        x = tf.unstack(x, maxlen, axis=1)
        w = tf.Variable(tf.random_normal([HIDDEN_SIZE, output_size]))
        b = tf.Variable(tf.random_normal([output_size]))
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x,
                                                    dtype=tf.float32,
                                                    sequence_length=seqlen)
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        batch_size = tf.shape(outputs)[0]
        index = tf.range(0, batch_size) * maxlen + (seqlen - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, HIDDEN_SIZE]), index)
        return tf.add(tf.matmul(outputs, w), b)

    @staticmethod
    def rnncoder(x, seqlen, maxlen, encoding_size=2):
        print(x.get_shape())
        encoded = Autoencoder.rnn(x, seqlen, maxlen, encoding_size, "Encoded")
        print(encoded.get_shape())
        x_shape = x.get_shape().as_list()
        decoded = Autoencoder.decoder(encoded, x_shape[1] * x_shape[2], False)
        decoded = tf.reshape(decoded, [-1, x_shape[1], x_shape[2]])
        print(decoded.get_shape())
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=x, logits=decoded), name="Loss")
        return loss, decoded, encoded

