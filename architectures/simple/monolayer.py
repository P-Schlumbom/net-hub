# General network for trzing a variety of different architectures

import numpy as np
from os import listdir, makedirs, path
from time import time
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

from architectures.squeezenet.skeleton_squeezenet import fire_module

class MonoNetwork():
    def __init__(self,  modelDetails, modelName, modelPath):
        self.modelName = modelName
        inputShape = modelDetails.get('inputShape', (32, 32, 3))  # default image shape assumed to be (32,32,3)
        nClass = modelDetails.get('nClass', 10)  # default # of classes is 10
        self.model = self.create_model(inputShape, nClass)
        self.info = modelDetails.get('info', 'no info available')

        # PREPARE READOUT DIRECTORIES
        if modelPath is not None and path.isfile(modelPath):  # init the model weights with provided one
            try:
                self.model.load_weights(modelPath)
                print("Loading model weights from {}".format(modelPath))
            except:
                print("Error loading model. Starting with randomly initialised weights.")
        #if not path.isfile('models/mono_models'):
        #    makedirs('models/mono_models')

        dirname = "logs/mono_net_logs/" + modelDetails['modelName']  # folder to store testing images in
        if not path.exists(dirname):
            self.num_files = 0
            makedirs(dirname)
            # create meta file to store all info
            with open(dirname + '/meta.csv', 'a+') as f:
                f.write("run, info, epochs, batch size, learning rate, training set size, testing set size, number of classes, train time, test time, train time per item, test time per item, test loss, test accuracy\n")
        else:
            self.num_files = len(listdir(dirname))
        name = dirname + '/run{}'.format(self.num_files)
        makedirs(name) # for storing output logs

        modelDetails["num_files"] = self.num_files

    def create_model(self, input_shape, output_size):
        """
        Creates a small convolutional model
        :param inputShape:
        :param nClass:
        :return: a Keras model
        """
        img_input = Input(shape=input_shape, name='Input_New')
        x = Flatten()(img_input)
        x = Dense(activation='sigmoid', name='layer_1', units=(np.prod(input_shape) + output_size)//2)(x) # the middle layer

        Dense_Output = Dense(name='Dense_Output', units=output_size, activation='softmax')(x)

        model = Model([img_input], [Dense_Output])
        return model

    def train(self, model, data, modelVars):
        """
        Training the net
        :param model: a convnet model
        :param data: a tuple containing training & testing data, e.g. ((x_train, y_train), (x_test, y_test))
        :param modelVars: a dictionary of the model variables
        :return: the trained model
        """
        start = time()
        # unpack the data
        (x_train, y_train), (x_test, y_test) = data
        self.trainSize = len(x_train)

        disp_callback = DisplayCallback(modelVars)  # display accuracy and loss during training (after each epoch)

        # compile the data
        model.compile(optimizer=optimizers.Adam(lr=modelVars['learningRate']), loss=losses.categorical_crossentropy, metrics=['accuracy'])

        # train the model
        history = model.fit(x_train, y_train,
                  batch_size=modelVars['batchSize'],
                  epochs=modelVars['epochs'],
                  validation_data=(x_test, y_test),
                            callbacks=[disp_callback]
                            )
        self.trainTime=time() - start
        self.store_log(history.history)
        model.save_weights('models/mono_models/' + modelVars['modelName'] + '_trained_model.h5')

        return model

    def test(self, model, data, modelVars):
        """
        Test the model to determine classification accuracy
        :param model:
        :param data:
        :param modelVars:
        :return:
        """
        start = time()
        print("STARTING TEST")
        (x_test, y_test) = data
        score = model.evaluate(x_test, y_test, verbose=0, batch_size=modelVars['batchSize'])
        self.testTime = time() - start
        print('TEST LOSS: ', score[0])
        print('TEST ACCURACY: ', score[1])

        report = "{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(self.num_files, self.info, modelVars['epochs'], modelVars['batchSize'], modelVars['learningRate'], self.trainSize, len(x_test), self.trainTime, self.testTime, self.trainTime/self.trainSize, self.testTime/len(x_test), score[0], score[1])
        with open('logs/mono_net_logs/{}/meta.csv'.format(self.modelName), 'a+') as f:
            f.write(report)

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


        with open("logs/mono_net_logs/{}/run{}/{:%Y-%m-%d %H-%M-%S}-{}-run-log.csv".format(self.modelName, self.num_files, datetime.now(), self.modelName), 'w') as f:
            f.write(training_log)

class DisplayCallback(Callback):
    def __init__(self, modelVars):
        self.modelVars = modelVars

        self.val_acc_log = []
        self.val_loss_log = []
        self.epoch_count = 0

    def on_train_begin(self, logs=None):
        plt.ion()
        plt.show()

    def on_epoch_end(self, epoch, logs={}):
        self.val_acc_log.append(logs['val_acc'])
        self.val_loss_log.append(logs['val_loss'])

        plt.clf()
        plt.subplot(1, 2, 1)
        plt.plot(self.val_acc_log)
        plt.ylim(ymin=0)
        plt.subplot(1, 2, 2)
        plt.plot(self.val_loss_log)
        plt.ylim(ymin=0)
        plt.draw()
        plt.pause(0.0005)

        self.epoch_count += 1

    def on_train_end(self, logs=None):
        plt.ioff()