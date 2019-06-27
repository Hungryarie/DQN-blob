from keras.callbacks import TensorBoard
import tensorflow as tf

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam

from constants import *


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class model_1:
    def __init__(self, input_shape, output_num):
        self.model = self.create_model(input_shape, output_num)
        self.target_model = self.create_model(input_shape, output_num)

    def create_model(self, input_shape, output_num):
        if LOAD_MODEL is not None:
            print(f"loading model: {LOAD_MODEL}")
            model = load_model(LOAD_MODEL)
            print("model loaded")
        else:
            model = Sequential()

            model.add(Conv2D(256, (3, 3), input_shape=input_shape))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(256, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(Dense(64))

            model.add(Dense(output_num, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
            model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model