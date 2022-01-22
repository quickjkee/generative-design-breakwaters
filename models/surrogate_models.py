import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def deep_conv_net():
    class Linear(layers.Layer):
        def __init__(self, name):
            self.output_dim = 1
            self.point = [[[98, 103], [176, 181]], [[160, 165], [123, 128]]]
            super(Linear, self).__init__(name=name)

        def build(self, input_shape):
            self.w = self.add_weight(shape=(self.output_dim, self.output_dim),
                                     trainable=False,
                                     initializer=tf.keras.initializers.RandomUniform(minval=-1., maxval=1.),
                                     name='w')
            self.b = self.add_weight(shape=(self.output_dim,),
                                     trainable=False,
                                     initializer=tf.keras.initializers.RandomUniform(minval=-1., maxval=1.),
                                     name='b')

        def call(self, inputs):
            pix_input1 = tf.keras.backend.mean(
                inputs[:, self.point[0][0][0]:self.point[0][0][1], self.point[0][1][0]:self.point[0][1][1]],
                axis=[1, 2, 3])
            pix_input2 = tf.keras.backend.mean(
                inputs[:, self.point[1][0][0]:self.point[1][0][1], self.point[1][1][0]:self.point[1][1][1]],
                axis=[1, 2, 3])
            pix_input = (pix_input1 + pix_input2) / 2

            return pix_input * (-1.30648437) + 1.24899516

    input_img = layers.Input(shape=(224, 224, 1))

    skip_connection = []
    skip_connection.append(input_img)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)
    skip_connection.append(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)
    skip_connection.append(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    skip_connection.append(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.concatenate([x, skip_connection[1]])
    skip_connection.append(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    decoded_first = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(decoded_first)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.concatenate([x, skip_connection[4]])
    skip_connection.append(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.concatenate([x, skip_connection[3]])
    skip_connection.append(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.concatenate([x, skip_connection[2]])
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoded')(x)

    wh_output = Linear(name='linear')(decoded)

    model = Model(inputs=input_img, outputs=[decoded, wh_output])

    return model


def assistant_net():
    input_img = layers.Input(shape=(224, 224, 1))

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    classifier = Model(input_img, x)

    return classifier

