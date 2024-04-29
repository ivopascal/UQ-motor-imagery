from keras.models import Model
from keras.layers import Dense, Activation, Permute, Dropout
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import SeparableConv2D, DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten
from keras.constraints import max_norm
from keras import backend as K

def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))


def ShallowConvNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.

    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25

    Note that this implementation has not been verified by the original
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations.
    """

    # start the model
    input_main = Input((Chans, Samples, 1))
    block1 = Conv2D(40, (1, 25),        #since sampling rate is 250 i use 25 instead of 13
                    input_shape=(Chans, Samples, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 75), strides=(1, 15))(block1)        #bigger pool size and strides because higher sampling rate
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)