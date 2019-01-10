import keras
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
#from keras.engine.topology import merge
from keras.layers import concatenate, Conv2D, Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.pooling import AveragePooling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.models import Model
from keras.regularizers import *


def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + "squeeze1x1")(x)
    x = Activation('relu', name=s_id + "relu_" + "squeeze1x1")(x)

    left = Conv2D(expand, (1, 1), padding='valid', name=s_id + "expand1x1")(x)
    left = Activation('relu', name=s_id + "relu_" + "expand1x1")(left)

    right = Conv2D(expand, (3, 3), padding='same', name=s_id + "expand3x3")(x)
    right = Activation('relu', name=s_id + "relu_" + "expand3x3")(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x

def original_squeezenet(input_shape, output_size):

    img_input = Input(shape=input_shape, name='Input_New')
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)

    Dropout_10 = Dropout(name='Dropout_10', rate=0.5)(x)
    Convolution2D_41 = Convolution2D(name='Convolution2D_41', kernel_size=(1, 1), activation='linear', filters=1000)(Dropout_10)
    AveragePooling2D_1 = AveragePooling2D(name='AveragePooling2D_1', pool_size=(13, 13), strides=(1, 1))(Convolution2D_41)
    Flatten_1 = Flatten(name='Flatten_1')(AveragePooling2D_1)

    Dense_Output = Dense(name='Dense_Output', units=output_size, activation='softmax')(Flatten_1)  # new layer; must be trained

    model = Model([img_input], [Dense_Output])
    return model

def custom_squeezenet(x):
    """
    Squeezenet model to insert into wherever it's needed. Returns a convolution output on the standard sequence of fire modules
    :param input_shape:
    :return:
    """
    #img_input = Input(shape=input_shape, name='Input_New')
    x = Conv2D(64, kernel_size=3, strides=2, padding='valid', name='conv1')(x)
    x = Activation('relu', name='relu_conv1')(x)
    #x = MaxPooling2D(pool_size=3, strides=2, name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    #x = MaxPooling2D(pool_size=3, strides=2, name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    #x = MaxPooling2D(pool_size=3, strides=2, name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)

    Dropout_10 = Dropout(name='Dropout_10', rate=0.5)(x)
    Convolution2D_41 = Convolution2D(name='Convolution2D_41', kernel_size=1, activation='linear', filters=1000, padding='same')(Dropout_10)
    #Convolution2D_41 = Convolution2D(filters=256, kernel_size=1, strides=1, padding='valid', activation='relu', name='conv1')(Dropout_10)

    return Convolution2D_41

def DLS_squeezenet(input_shape, output_size):
    Input_1 = Input(shape=input_shape, name='Input_New')  # new layer; must be trained

    Convolution2D_14 = Conv2D(name='Convolution2D_14', strides=(2, 2), kernel_size=(7, 7), activation='relu',
                              data_format="channels_first", filters=96)(Input_1)
    MaxPooling2D_9 = MaxPooling2D(name='MaxPooling2D_9', pool_size=(3, 3), strides=(2, 2))(Convolution2D_14)
    Convolution2D_15 = Conv2D(name='Convolution2D_15', kernel_size=(1, 1), activation='relu', filters=16)(
        MaxPooling2D_9)
    Convolution2D_17 = Conv2D(name='Convolution2D_17', kernel_size=(3, 3), padding='same', activation='relu',
                              filters=64)(Convolution2D_15)
    Convolution2D_16 = Conv2D(name='Convolution2D_16', kernel_size=(1, 1), activation='relu', filters=64)(
        Convolution2D_15)
    merge_1 = concatenate([Convolution2D_17, Convolution2D_16], axis=1, name='merge_1')
    Convolution2D_20 = Conv2D(name='Convolution2D_20', kernel_size=(1, 1), activation='relu', filters=16)(merge_1)
    Convolution2D_22 = Conv2D(name='Convolution2D_22', kernel_size=(3, 3), padding='same', activation='relu',
                              filters=64)(Convolution2D_20)
    Convolution2D_21 = Conv2D(name='Convolution2D_21', kernel_size=(1, 1), activation='relu', filters=64)(
        Convolution2D_20)
    merge_2 = concatenate([Convolution2D_21, Convolution2D_22], axis=1, name='merge_2')
    MaxPooling2D_10 = MaxPooling2D(name='MaxPooling2D_10', pool_size=(3, 3), strides=(2, 2))(merge_2)
    Convolution2D_23 = Conv2D(name='Convolution2D_23', kernel_size=(1, 1), activation='relu', filters=32)(
        MaxPooling2D_10)
    Convolution2D_24 = Conv2D(name='Convolution2D_24', kernel_size=(1, 1), activation='relu', filters=128)(
        Convolution2D_23)
    Convolution2D_25 = Conv2D(name='Convolution2D_25', kernel_size=(3, 3), padding='same', activation='relu',
                              filters=128)(Convolution2D_23)
    merge_3 = concatenate([Convolution2D_24, Convolution2D_25], axis=1, name='merge_3')
    Convolution2D_26 = Conv2D(name='Convolution2D_26', kernel_size=(1, 1), activation='relu', filters=32)(merge_3)
    Convolution2D_27 = Conv2D(name='Convolution2D_27', kernel_size=(1, 1), activation='relu', filters=128)(
        Convolution2D_26)
    Convolution2D_28 = Conv2D(name='Convolution2D_28', kernel_size=(3, 3), padding='same', activation='relu',
                              filters=128)(Convolution2D_26)
    merge_4 = concatenate([Convolution2D_27, Convolution2D_28], axis=1, name='merge_4')
    MaxPooling2D_11 = MaxPooling2D(name='MaxPooling2D_11', pool_size=(3, 3), strides=(2, 2))(merge_4)
    Convolution2D_29 = Conv2D(name='Convolution2D_29', kernel_size=(1, 1), activation='relu', filters=48)(
        MaxPooling2D_11)
    Convolution2D_30 = Conv2D(name='Convolution2D_30', kernel_size=(1, 1), activation='relu', filters=192)(
        Convolution2D_29)
    Convolution2D_31 = Conv2D(name='Convolution2D_31', kernel_size=(3, 3), padding='same', activation='relu',
                              filters=192)(Convolution2D_29)
    merge_5 = concatenate([Convolution2D_30, Convolution2D_31], axis=1, name='merge_5')
    Convolution2D_32 = Conv2D(name='Convolution2D_32', kernel_size=(1, 1), activation='relu', filters=48)(merge_5)
    Convolution2D_33 = Conv2D(name='Convolution2D_33', kernel_size=(1, 1), activation='relu', filters=192)(
        Convolution2D_32)
    Convolution2D_34 = Conv2D(name='Convolution2D_34', kernel_size=(3, 3), padding='same', activation='relu',
                              filters=192)(Convolution2D_32)
    merge_6 = concatenate([Convolution2D_33, Convolution2D_34], axis=1, name='merge_6')
    Convolution2D_35 = Conv2D(name='Convolution2D_35', kernel_size=(1, 1), activation='relu', filters=64)(merge_6)
    Convolution2D_37 = Conv2D(name='Convolution2D_37', kernel_size=(3, 3), padding='same', activation='relu',
                              filters=256)(Convolution2D_35)
    Convolution2D_36 = Conv2D(name='Convolution2D_36', kernel_size=(1, 1), activation='relu', filters=256)(
        Convolution2D_35)
    merge_7 = concatenate([Convolution2D_37, Convolution2D_36], axis=1, name='merge_7')
    Convolution2D_38 = Conv2D(name='Convolution2D_38', kernel_size=(1, 1), activation='relu', filters=64)(merge_7)
    Convolution2D_40 = Conv2D(name='Convolution2D_40', kernel_size=(3, 3), padding='same', activation='relu',
                              filters=256)(Convolution2D_38)
    Convolution2D_39 = Conv2D(name='Convolution2D_39', kernel_size=(1, 1), activation='relu', filters=256)(
        Convolution2D_38)
    merge_8 = concatenate([Convolution2D_40, Convolution2D_39], axis=1, name='merge_8')
    Dropout_10 = Dropout(name='Dropout_10', rate=0.5)(merge_8)
    Convolution2D_41 = Convolution2D(name='Convolution2D_41', kernel_size=(1, 1), activation='linear', filters=1000)(
        Dropout_10)
    AveragePooling2D_1 = AveragePooling2D(name='AveragePooling2D_1', pool_size=(13, 13), strides=(1, 1))(
        Convolution2D_41)
    Flatten_1 = Flatten(name='Flatten_1')(AveragePooling2D_1)

    Dense_Output = Dense(name='Dense_Output', units=output_size, activation='softmax')(
        Flatten_1)  # new layer; must be trained

    model = Model([Input_1], [Dense_Output])

def get_model(input_shape, output_size, model_path=None):
    model = original_squeezenet(input_shape, output_size)

    if model_path is not None:
        model.load_weights(model_path, by_name=True) # by_name=True should only load weights for those layers with the same name - in this case all but the first (input) and last (output) layers

    return model