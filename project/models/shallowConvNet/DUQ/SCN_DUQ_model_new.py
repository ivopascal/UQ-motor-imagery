from keras import layers, optimizers, regularizers, constraints, backend
from keras_uncertainty.layers import RBFClassifier
import keras
from tensorflow.python.keras import backend as K


def add_l2_regularization(model, l2_strength=1e-4):
    for layer in model.layers:
        for w in layer.trainable_weights:
            model.add_loss(regularizers.l2(l2_strength)(w))

def ShallowConvNet(nb_classes, Chans=22, Samples=1001, dropoutRate=0.5):
    model = keras.models.Sequential(name="DUQ-SCN")
    model.add(layers.Conv2D(40, (1, 25),
                            input_shape=(Chans, Samples, 1),
                            kernel_constraint=constraints.max_norm(2.)))
    model.add(layers.Conv2D(40, (Chans, 1), use_bias=False,
                            kernel_constraint=constraints.max_norm(2.)))
    model.add(layers.BatchNormalization(epsilon=1e-5, momentum=0.9))
    model.add(layers.Activation(K.square))
    model.add(layers.AveragePooling2D(pool_size=(1, 75), strides=(1, 15)))
    model.add(layers.Activation(lambda x: K.log(K.clip(x, 1e-7, 1e4))))
    model.add(layers.Dropout(dropoutRate))
    model.add(layers.Flatten())
    model.add(layers.Dense(nb_classes,
                           kernel_constraint=constraints.max_norm(0.5)))
    model.add(RBFClassifier(nb_classes, length_scale=0.2))

    add_l2_regularization(model)                       # BEFORE compile

    model.compile(optimizer=optimizers.Adam(1e-2),
                  loss="categorical_crossentropy",
                  metrics=["categorical_accuracy"])
    return model
