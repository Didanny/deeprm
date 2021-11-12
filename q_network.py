import os

# from tensorflow.python.keras.initializers.initializers_v2 import Zeros
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
# from keras import initializations

CHECKPOINT_DIR = os.path.abspath('./ckpt')
LOG_DIR = os.path.abspath('./logs')

class QNetwork():
    # Constructor
    def __init__(self, params) -> None:
        super().__init__()

        self.input_height = params.network_input_height
        self.input_width = params.network_input_width
        self.output_height = params.network_output_dim
        # print("QNET:{0}".format(params.network_output_dim))

        self.model = self.get_compiled_model()

    # Build and compile the network
    def get_compiled_model(self):
        import tensorflow as tf
        from tensorflow import keras
        from keras import layers
        from keras.layers import Dense
        from keras import initializers
        l_in = keras.Input(
            shape=(self.input_height*self.input_width)
            )
        l_hid = Dense(
            units=20,
            input_shape=(self.input_height, self.input_width),
            kernel_initializer=initializers.RandomNormal(.01),
            bias_initializer=initializers.Zeros(),
            activation='relu'
        )
        l_out = layers.Dense(
            units=self.output_height,
            kernel_initializer=initializers.RandomNormal(0.01),
            bias_initializer=initializers.Zeros(),
            activation='linear'
        )

        model = tf.keras.Sequential([l_in, l_hid, l_out])
        model.compile('rmsprop', 'mse')
        return model

    # Make a prediction
    def get_q_values(self, x, batch_size=None):
        return self.model.predict(x, batch_size=batch_size)

    def act(self, state):
        state = np.expand_dims(state, axis=0)
        return np.argmax(self.model.predict(state), axis=-1)[0]

    def training_step(self, x, y):
        loss = self.model.train_on_batch(x=x, y=y)
        return loss

    def save_model(self, filename):
        self.model.save(filename)