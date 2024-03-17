#https://github.com/NeuroTechX/moabb/blob/develop/moabb/pipelines/deep_learning.py#L65-L144

"""Deep learning integrated in MOABB Implementation using the tensorflow, keras
and scikeras framework."""

# Authors: Igor Carrara <igor.carrara@inria.fr>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#          Sylvain Chevallier <sylvain.chevallier@universite-paris-saclay.fr>

# License: BSD (3-clause)

from typing import Any, Dict

import tensorflow as tf
from keras import backend as K
from keras.constraints import max_norm
from keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    AvgPool2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    Flatten,
    Input,
    Lambda,
    LayerNormalization,
    MaxPooling2D,
    Permute,
)
from keras.models import Model, Sequential
from scikeras.wrappers import KerasClassifier

from moabb.pipelines.utils_deep_model import EEGNet, EEGNet_TC, TCN_block


# =====================================================================================
# ShallowConvNet
# =====================================================================================
def square(x):
    """Function to square the input tensor element-wise.

    Element-wise square.
    """
    return K.square(x)


def log(x):
    """Function to take the log of the input tensor element-wise. We use a clip
    to avoid taking the log of 0. min_value=1e-7, max_value=10000.

    Parameters
    ----------
    x: tensor

    Returns
    -------
    tensor
    """
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


class KerasShallowConvNet(KerasClassifier):
    """Keras implementation of the Shallow Convolutional Network as described
    in [1]_.

    This implementation is taken from code by the Army Research Laboratory (ARL)
    at https://github.com/vlawhern/arl-eegmodels

    We use the original parameter implemented on the paper.

    Note that this implementation has not been verified by the original
    authors.

    References
    ----------
    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger,
           K., Tangermann, M., ... & Ball, T. (2017). Deep learning with convolutional neural networks
           for EEG decoding and visualization. Human brain mapping, 38(11), 5391-5420.
           https://doi.org/10.1002/hbm.23730

    Notes
    -----
    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        loss,
        optimizer="Adam",
        epochs=1000,
        batch_size=64,
        verbose=0,
        random_state=None,
        validation_split=0.2,
        history_plot=False,
        path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_classes_ = None
        self.loss = loss
        if optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.validation_split = validation_split
        self.history_plot = history_plot
        self.path = path

    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
        input_main = Input(shape=(self.X_shape_[1], self.X_shape_[2], 1))
        block1 = Conv2D(
            40,
            (1, 25),
            input_shape=(self.X_shape_[1], self.X_shape_[2], 1),
            kernel_constraint=max_norm(2.0, axis=(0, 1, 2)),
        )(input_main)
        block1 = Conv2D(
            40,
            (self.X_shape_[1], 1),
            use_bias=False,
            kernel_constraint=max_norm(2.0, axis=(0, 1, 2)),
        )(block1)
        block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
        block1 = Activation(square)(block1)
        block1 = AveragePooling2D(pool_size=(1, 75), strides=(1, 15))(block1)
        block1 = Activation(log)(block1)
        block1 = Dropout(0.5)(block1)
        flatten = Flatten()(block1)
        dense = Dense(self.n_classes_, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation("softmax")(dense)

        model = Model(inputs=input_main, outputs=softmax)

        model.compile(loss=compile_kwargs["loss"], optimizer=compile_kwargs["optimizer"])

        return model

