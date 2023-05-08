from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, concatenate, Dropout, Conv2DTranspose, RandomFlip, \
    BatchNormalization
import tensorflow as tf
from keras import regularizers


class Base(tf.keras.Model):
    def __init__(self):
        super(Base, self).__init__()
        self.dense1 = Dense(10, activation='relu')
        self.bnLayer1 = BatchNormalization()
        self.dropoutLayer = Dropout(0.2)
        self.dense2 = Dense(8, activation='relu')
        self.bnLayer2 = BatchNormalization()
        self.dense3 = Dense(5, activation='relu')
        self.outLayer = Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bnLayer1(x)
        x = self.dropoutLayer(x)
        x = self.dense2(x)
        x = self.bnLayer2(x)
        x = self.dense3(x)
        out = self.outLayer(x)
        return out


class ConvBase(tf.keras.Model):
    def __init__(self):
        super(ConvBase, self).__init__()
        self.convLayer1 = Conv2D(
            filters=8, kernel_size=(3, 3), strides=1, activation='relu',
            padding='same', kernel_initializer='HeNormal'
        )
        self.convLayer2 = Conv2D(
            filters=6, kernel_size=(3, 3), strides=1, activation='relu',
            padding='same', kernel_initializer='HeNormal'
        )
        self.convLayer3 = Conv2D(
            filters=4, kernel_size=(3, 3), strides=1, activation='relu',
            padding='same', kernel_initializer='HeNormal'
        )
        self.convLayer4 = Conv2D(
            filters=2, kernel_size=(3, 3), strides=1, activation='relu',
            padding='valid', kernel_initializer='HeNormal'
        )
        self.flat = Flatten()
        self.dense1 = Dense(10, activation='relu')
        self.bnLayer1 = BatchNormalization()
        self.dropoutLayer = Dropout(0.2)
        self.dense2 = Dense(8, activation='relu')
        self.bnLayer2 = BatchNormalization()
        self.dense3 = Dense(5, activation='relu')
        self.outLayer = Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        x = self.convLayer1(inputs[0])
        x = self.convLayer2(x)
        x = self.convLayer3(x)
        x = self.convLayer4(x)
        x = self.flat(x)
        x = concatenate([x, inputs[1]])
        x = self.dense1(x)
        x = self.bnLayer1(x)
        x = self.dropoutLayer(x)
        x = self.dense2(x)
        x = self.bnLayer2(x)
        x = self.dense3(x)
        out = self.outLayer(x)
        return out


class Conv(tf.keras.Model):
    def __init__(self):
        super(Conv, self).__init__()
        self.convLayer1 = Conv2D(
            filters=8, kernel_size=(3, 3), strides=1, activation='relu',
            padding='same',
            kernel_initializer='HeNormal'
        )
        self.convLayer2 = Conv2D(
            filters=6, kernel_size=(3, 3), strides=1, activation='relu',
            padding='same',
            kernel_initializer='HeNormal'
        )
        self.convLayer3 = Conv2D(
            filters=4, kernel_size=(3, 3), strides=1, activation='relu',
            padding='same',
            kernel_initializer='HeNormal'
        )
        self.convLayer4 = Conv2D(
            filters=2, kernel_size=(3, 3), strides=1, activation='relu',
            padding='valid', kernel_initializer='HeNormal'
        )
        self.flat = Flatten()
        self.dense1 = Dense(10, activation='relu')
        self.bnLayer1 = BatchNormalization()
        self.dropoutLayer = Dropout(0.2)
        self.dense2 = Dense(8, activation='relu')
        self.bnLayer2 = BatchNormalization()
        self.dense3 = Dense(5, activation='relu')
        self.outLayer = Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        x = self.convLayer1(inputs)
        x = self.convLayer2(x)
        x = self.convLayer3(x)
        x = self.convLayer4(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.bnLayer1(x)
        x = self.dropoutLayer(x)
        x = self.dense2(x)
        x = self.bnLayer2(x)
        x = self.dense3(x)
        out = self.outLayer(x)
        return out


class FlattenBase(tf.keras.Model):
    def __init__(self):
        super(FlattenBase, self).__init__()
        self.flat = Flatten()
        self.dense1 = Dense(10, activation='relu',
                            kernel_regularizer=regularizers.L1(1e-3))
        self.bnLayer1 = BatchNormalization()
        self.dropoutLayer = Dropout(0.2)
        self.dense2 = Dense(8, activation='relu',
                            kernel_regularizer=regularizers.L1(1e-3))
        self.bnLayer2 = BatchNormalization()
        self.dense3 = Dense(5, activation='relu',
                            kernel_regularizer=regularizers.L1(1e-3))
        self.outLayer = Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        x = self.flat(inputs[0])
        x = concatenate([x, inputs[1]])
        x = self.dense1(x)
        x = self.bnLayer1(x)
        x = self.dropoutLayer(x)
        x = self.dense2(x)
        x = self.bnLayer2(x)
        x = self.dense3(x)
        out = self.outLayer(x)
        return out


class FlattenModel(tf.keras.Model):
    def __init__(self):
        super(FlattenModel, self).__init__()
        self.flat = Flatten()
        self.dense1 = Dense(10, activation='relu', kernel_regularizer=regularizers.L1(1e-3))
        self.bnLayer1 = BatchNormalization()
        self.dropoutLayer = Dropout(0.2)
        self.dense2 = Dense(8, activation='relu', kernel_regularizer=regularizers.L1(1e-3))
        self.bnLayer2 = BatchNormalization()
        self.dense3 = Dense(5, activation='relu', kernel_regularizer=regularizers.L1(1e-3))
        self.outLayer = Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        x = self.flat(inputs)
        x = self.dense1(x)
        x = self.bnLayer1(x)
        x = self.dropoutLayer(x)
        x = self.dense2(x)
        x = self.bnLayer2(x)
        x = self.dense3(x)
        out = self.outLayer(x)
        return out
