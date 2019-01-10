# General network for trzing a variety of different architectures

import numpy as np
from os import listdir, makedirs, path
import time
from datetime import datetime
from matplotlib import pyplot as plt

from keras import Sequential
from keras import layers, models, optimizers, losses
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array

from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
#from keras.engine.topology import merge
from keras.layers import concatenate, Conv2D, Activation, Add
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.pooling import AveragePooling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.models import Model
from keras.regularizers import *
from keras.callbacks import Callback
from keras.layers import Reshape

from architectures.squeezenet.skeleton_squeezenet import fire_module

class Autoencoder():
    def __init__(self,  modelDetails, modelName, modelPath):
        self.modelName = modelName
        inputShape = modelDetails.get('inputShape', (32, 32, 3))  # default image shape assumed to be (32,32,3)
        self.encoder, self.autoencoder = self.create_model(inputShape, min_dim=80)

        # PREPARE READOUT DIRECTORIES
        if modelPath is not None and path.isfile(modelPath):  # init the model weights with provided one
            try:
                self.autoencoder.load_weights(modelPath)
                print("Loading model weights from {}".format(modelPath))
            except:
                print("Error loading model. Starting with randomly initialised weights.")

        dirname = "logs/autoencoder_logs/" + modelDetails['modelName']
        if not path.exists(dirname):
            self.num_files = 0
        else:
            self.num_files = len(listdir(dirname))
        name = dirname + '/run{}'.format(self.num_files)
        makedirs(name) # for storing output logs

        modelDetails["num_files"] = self.num_files

    def create_model(self, input_shape, min_dim=64):
        """
        Creates autoencoder models
        :param inputShape:
        :param nClass:
        :return: a Keras model
        """
        x = layers.Input(shape=input_shape)
        print(input_shape, np.prod(input_shape))
        encoded = Flatten()(x)
        #encoded = Dense(1458, activation='relu', input_dim=np.prod(input_shape))(encoded)
        encoded = Dense(1024, activation='relu', input_dim=np.prod(input_shape))(encoded)
        encoded = Dense(512, activation='relu')(encoded)
        encoded = Dense(256, activation='relu')(encoded)
        encoded = Dense(128, activation='relu')(encoded)
        encoded = Dense(min_dim, activation='relu')(encoded)

        decoded = Dense(128, activation='relu')(encoded)
        decoded = Dense(256, activation='relu')(decoded)
        decoded = Dense(512, activation='relu')(decoded)
        decoded = Dense(1024, activation='relu')(decoded)
        decoded = Dense(np.prod(input_shape), activation='relu')(decoded)
        decoded = Dense(np.prod(input_shape), activation='linear')(decoded)
        decoded = Reshape(target_shape=input_shape, name='out_recon')(decoded)

        encoder = Model(x, encoded)
        autoencoder = Model(x, decoded)
        return encoder, autoencoder

    def train(self, model, data, modelVars):
        """
        Training the net
        :param model: an autoencoder model
        :param data: a tuple containing training & testing data, e.g. ((x_train, y_train), (x_test, y_test))
        :param modelVars: a dictionary of the model variables
        :return: the trained model
        """
        # unpack the data
        (x_train, y_train), (x_test, y_test) = data

        disp_callback = DisplayCallback(x_test)  # display accuracy and loss during training (after each epoch)

        # compile the data
        #model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.SGD(lr=0.01), metrics=['accuracy'])
        #model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
        model.compile(optimizer=optimizers.Adam(lr=modelVars['learningRate']), loss=losses.mse, metrics=['accuracy'])
        #model.compile(optimizer=optimizers.SGD(lr=modelVars['learningRate'], momentum=0.9, nesterov=True), loss=losses.categorical_crossentropy, metrics=['accuracy'])

        """# train the model
        history = model.fit(x_train, x_train,
                  batch_size=modelVars['batchSize'],
                  epochs=modelVars['epochs'],
                  validation_data=(x_test, x_test),
                            callbacks=[disp_callback]
                            )
        self.store_log(history.history)
        model.save_weights('models/autoencoder_models/' + modelVars['modelName'] + '_trained_model.h5')"""

        return model

    def test(self, model, data, modelVars):
        """
        Test the model to determine classification accuracy
        :param model:
        :param data:
        :param modelVars:
        :return:
        """
        print("STARTING TEST")
        (x_test, y_test) = data
        score = model.evaluate(x_test, x_test, verbose=0, batch_size=modelVars['batchSize'])
        print('TEST LOSS: ', score[0])
        print('TEST ACCURACY: ', score[1])

        self.autoencoder.save('models/autoencoder_models/' + modelVars['modelName'] + '_autoencoder.h5')
        self.encoder.save('models/autoencoder_models/' + modelVars['modelName'] + '_encoder.h5')

    def store_log(self, history):
        print(history.keys())
        keys = [key for key in history.keys()]
        training_log = ""
        for item in keys:
            training_log += item + ','
        training_log = training_log[:-1]
        for i in range(len(history[keys[0]])):
            training_log += '\n'
            for j in range(len(keys)):
                training_log += str(history[keys[j]][i]) + ','  # the ith element of the jth key
            training_log = training_log[:-1]


        with open("logs/autoencoder_logs/{}/run{}/{:%Y-%m-%d %H-%M-%S}-{}-run-log.csv".format(self.modelName, self.num_files, datetime.now(), self.modelName), 'w') as f:
            f.write(training_log)

class DisplayCallback(Callback):
    def __init__(self, valid_data, disp_range=10):
        self.x_test = valid_data
        self.d_range = disp_range

        self.val_acc_log = []
        self.val_loss_log = []
        self.epoch_count = 0

        self.in_sample = self.x_test[:disp_range]
        self.out_sample = self.x_test[:disp_range]

    def on_train_begin(self, logs=None):
        plt.ion()
        plt.show()

    def on_epoch_end(self, epoch, logs={}):
        self.val_acc_log.append(logs['val_acc'])
        self.val_loss_log.append(logs['val_loss'])

        x_recon = self.model.predict(self.in_sample)

        display_im = np.concatenate((self.in_sample[0], x_recon[0]), axis=1)
        for i in range(1, self.d_range):
            out = np.concatenate((self.in_sample[i], x_recon[i]), axis=1)
            display_im = np.concatenate((display_im, out))
        display_im = np.ndarray.astype(display_im, int)

        plt.clf()
        plt.subplot(1, 3, 1)
        plt.imshow(display_im)
        plt.subplot(1, 3, 2)
        plt.plot(self.val_acc_log)
        plt.ylim(ymin=0)
        plt.subplot(1, 3, 3)
        plt.plot(self.val_loss_log)
        plt.ylim(ymin=0)
        plt.draw()
        plt.pause(0.0005)

        self.epoch_count += 1

    def on_train_end(self, logs=None):
        plt.ioff()