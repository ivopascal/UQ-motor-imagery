from keras_uncertainty.layers import RBFClassifier, FlipoutDense
#from keras_uncertainty.layers import add_l2_regularization
from keras.constraints import max_norm
import keras
import keras.layers

from keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.backend import log, square, clip

from keras.regularizers import l2


def add_l2_regularization(model, l2_strength=1e-4):
    for layer in model.layers:
        if layer.trainable_weights:
            # Wrap the regularization inside a lambda to ensure it's callable
            layer_loss = lambda: l2(l2_strength)(layer.trainable_weights[0])
            model.add_loss(layer_loss)


class BaseConvModel:
    def __init__(self, hp=None, C=22, T=1125, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4):
        self.C = C  # Numbear of electrodes
        self.T = T  # Time samples of network input
        self.f = f  # Number of convolutional kernels
        self.k1 = k1  # Kernel size
        self.k2 = (self.C, 1)  # Kernel size
        self.fp = fp  # Pooling size
        self.sp = sp  # Pool stride
        self.Nc = Nc  # Number of classes
        self.input_shape = (self.C, self.T, 1)
        self.hp = hp

    def add_conv_filters(self, model):
        model.add(Conv2D(filters=self.f, kernel_size=self.k1,
                         padding='SAME',
                         activation="linear",
                         input_shape=self.input_shape,
                         kernel_constraint=max_norm(2, axis=(0, 1, 2))))
        model.add(Conv2D(filters=self.f, kernel_size=self.k2,
                         padding='SAME',
                         activation="linear",
                         kernel_constraint=max_norm(2, axis=(0, 1, 2))))
        return model

    def add_batch_norm(self, model):
        model.add(BatchNormalization(momentum=0.9, epsilon=1e-05))
        model.add(Activation(lambda x: square(x)))
        return model

    def add_pooling(self, model):
        model.add(AveragePooling2D(pool_size=self.fp,
                                   strides=self.sp))
        model.add(Activation(lambda x: log(clip(x, min_value=1e-7, max_value=10000))))
        return model

    def flatten(self, model):
        model.add(Flatten())
        return model

    def add_dense(self, model):
        model.add(Dense(self.Nc, activation='softmax',
                        kernel_constraint=max_norm(0.5)))
        return model

    def compile_model(self, model, lr=1e-4, loss='categorical_crossentropy', metrics=['accuracy']):
        optimizer = Adam(learning_rate=lr)
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)
        return model

    def init_model(self):
        return Sequential()

    def build(self, hp):
        self.hp = hp
        model = self.init_model()
        self.add_conv_filters(model)
        self.add_batch_norm(model)
        self.add_pooling(model)
        self.flatten(model)
        self.add_dense(model)
        self.compile_model(model)
        return model

    def get_model(self):
        return self.build(None)


class SCN_DUQ(BaseConvModel):
    def __init__(self, hp=None, C=22, T=1001, f=40,
                 k1=(1, 25), fp=(1, 75), sp=(1, 15),
                 Nc=4):
        super().__init__(hp, C, T, f, k1, fp, sp, Nc)

    def add_dense(self, model):
        model.add(keras.layers.Dense(100, activation='relu', kernel_constraint=max_norm(0.5)))
        return model

    def add_rbf_layer(self, model):
        centr_dims = 2
        length_scale = 0.1
        train_centroids = False
        model.add(RBFClassifier(self.Nc, length_scale, centroid_dims=centr_dims, trainable_centroids=train_centroids))
        return model

    def build(self, hp):
        self.hp = hp
        model = keras.models.Sequential()
        model = self.add_conv_filters(model)
        model = self.add_batch_norm(model)
        model = self.add_pooling(model)
        model = self.flatten(model)
        model = self.add_dense(model)
        #model.add(Activation('softmax'))
        model = self.add_rbf_layer(model)
        model = self.compile_model(model, loss='binary_crossentropy', metrics=["categorical_accuracy"])
        add_l2_regularization(model)
        return model

