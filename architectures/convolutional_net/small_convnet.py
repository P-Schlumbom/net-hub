import numpy as np
from datetime import datetime
from keras import Sequential
from keras import layers, models, optimizers, losses
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array

class ConvolutionalNetwork():
    def __init__(self,  modelDetails, modelName, modelPath):
        self.modelName = modelName
        inputShape = modelDetails.get('inputShape', (32, 32, 1))  # default image shape assumed to be (32,32,3)
        nClass = modelDetails.get('nClass', 10)  # default # of classes is 10
        self.model = self.create_convnet_model(inputShape, nClass)

    def create_convnet_model(self, inputShape, nClass):
        """
        Creates a small convolutional model
        :param inputShape:
        :param nClass:
        :return: a Keras model
        """
        model = Sequential()
        #model.add(layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1', input_shape=inputShape))
        model.add(layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', name='conv1',input_shape=inputShape))
        model.add(layers.Activation('relu', name='conv1_act'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2'))
        #model.add(layers.Conv2D(filters=64, kernel_size=5, padding='same', activation='relu', name='conv3'))
        model.add(layers.Conv2D(filters=64, kernel_size=5, padding='same', name='conv3'))
        model.add(layers.Activation('relu', name='conv3_act'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool4'))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(nClass, activation='softmax'))

        return model

    def train(self, model, data, modelVars):
        """
        Training the convnet
        :param model: a convnet model
        :param data: a tuple containing training & testing data, e.g. ((x_train, y_train), (x_test, y_test))
        :param modelVars: a dictionary of the model variables
        :return: the trained model
        """
        # unpack the data
        (x_train, y_train), (x_test, y_test) = data

        # compile the data
        #model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.SGD(lr=0.01), metrics=['accuracy'])
        #model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
        model.compile(optimizer=optimizers.Adam(lr=0.00001), loss='mse', metrics=['accuracy'])

        # train the model
        history = model.fit(x_train, y_train,
                  batch_size=modelVars['batchSize'],
                  epochs=modelVars['epochs'],
                  validation_data=(x_test, y_test))
        self.store_log(history.history)
        model.save_weights('models/convnet_models/' + modelVars['modelName'] + '_trained_convnet_model.h5')

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
        score = model.evaluate(x_test, y_test, verbose=0, batch_size=modelVars['batchSize'])
        print('TEST LOSS: ', score[0])
        print('TEST ACCURACY: ', score[1])

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
        with open("logs/convolutional_net_logs/{:%Y-%m-%d %H-%M-%S}-{}-run-log.csv".format(datetime.now(), self.modelName), 'w') as f:
            f.write(training_log)