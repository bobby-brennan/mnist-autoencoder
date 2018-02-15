import tensorflow as tf

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
        print("in %d out %d" % (input_size, output_size))
        W = Autoencoder.weight_variable([input_size, output_size], name + '_W')
        b = Autoencoder.bias_variable([output_size], name + '_b')
        return tf.add(tf.matmul(previous, W), b, name=name)

    @staticmethod
    def encoder(x, encoding_size):
        l1 = tf.nn.tanh(Autoencoder.fc_layer(x, x.get_shape().as_list()[1], 50, 'encoder_1'))
        l2 = tf.nn.tanh(Autoencoder.fc_layer(l1, 50, 50, 'encoder_2'))
        l3 = tf.nn.tanh(Autoencoder.fc_layer(l2, 50, encoding_size, 'encoder_3'), name="Encoded")
        return l3

    @staticmethod
    def decoder(encoded, out_size):
        l4 = tf.nn.tanh(Autoencoder.fc_layer(encoded, encoded.get_shape().as_list()[1], 50, 'decoder_1'))
        l5 = tf.nn.tanh(Autoencoder.fc_layer(l4, 50, 50, 'decoder_2'))
        out = tf.nn.relu(Autoencoder.fc_layer(l5, 50, out_size, 'decoder_3'), name="Decoded")
        return out

    @staticmethod
    def discriminator(x):
        l1 = tf.nn.tanh(Autoencoder.fc_layer(x, x.get_shape().as_list()[1], 50, 'discriminator_1'))
        l2 = tf.nn.tanh(Autoencoder.fc_layer(l1, 50, 50, 'discriminator_2',))
        l3 = tf.nn.tanh(Autoencoder.fc_layer(l2, 50, 1, 'discriminator_3'), name="Discriminated")
        return l3

    @staticmethod
    def autoencoder(x):
        encoded = Autoencoder.encoder(x, 2)
        decoded = Autoencoder.decoder(encoded, x.get_shape().as_list()[1])
        # let's use an l2 loss on the output image
        loss = tf.reduce_mean(tf.squared_difference(x, decoded), name="Loss")
        return loss, decoded, encoded

    @staticmethod
    def add_noise(x):
        noise = tf.random_normal(shape=tf.shape(x), mean=0.5, stddev=5)
        return x + noise

    @staticmethod
    def gancoder(x, fake_encoded):
        encoded = Autoencoder.encoder(x, 2)
        decoded = Autoencoder.decoder(encoded, x.get_shape().as_list()[1])
        decoded_fake = Autoencoder.decoder(fake_encoded, x.get_shape().as_list()[1])
        discriminated = Autoencoder.discriminator(Autoencoder.add_noise(x))
        discriminated_fake = Autoencoder.discriminator(Autoencoder.add_noise(decoded_fake))
        discriminator_loss_real = tf.reduce_mean(tf.abs(discriminated - 1))
        discriminator_loss_fake = tf.reduce_mean(tf.abs(discriminated_fake + 1))
        discriminator_loss = (discriminator_loss_real + discriminator_loss_fake) / 4
        generator_loss = tf.reduce_mean(tf.squared_difference(x, decoded), name="GenLoss")
        return generator_loss, discriminator_loss, decoded, encoded

