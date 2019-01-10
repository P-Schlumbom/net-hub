"""
Keras implementation of Sabour, Frosst & Hinton's Capsule Network (https://arxiv.org/abs/1710.09829),
based very heavily on Xifeng Guo"s implementation: https://github.com/XifengGuo/CapsNet-Keras
"""
import numpy as np
from os import listdir, makedirs, path
from time import time
from datetime import datetime
from PIL import Image

from keras import layers, models, optimizers
from keras import backend as K # don't like the look of this, replaced it with numpy functions because I can
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from architectures.capsule_net.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.models import save_model

from matplotlib import pyplot as plt

from architectures.squeezenet.skeleton_squeezenet import fire_module, custom_squeezenet

class SimpleCapsuleNetwork():
    def __init__(self, modelDetails, modelName, modelPath):
        self.modelName = modelName
        inputShape = modelDetails.get('inputShape', (32, 32, 3)) # default image shape assumed to be (32,32,3)
        nClass = modelDetails.get('nClass', 10) # default # of classes is 10
        routings = modelDetails.get('routings', 3) # default # of routing iterations is 3, as suggested by the paper.
        self.model = self.create_capsnet_model(inputShape, nClass, routings)
        self.info = modelDetails.get('info', 'no info available')

        # PREPARE READOUT DIRECTORIES
        if modelPath is not None and path.isfile(modelPath):  # init the model weights with provided one
            try:
                self.model.load_weights(modelPath)
                print("Loading model weights from {}".format(modelPath))
            except:
                print("Error loading model. Starting with randomly initialised weights.")

        dirname = "results/simple_capsule_net_results/" + modelDetails['modelName']  # folder to store testing images in
        if not path.exists(dirname):
            self.num_files = 0
            makedirs(dirname)
            # create meta file to store all info
            with open(dirname + '/meta.csv', 'a+') as f:
                f.write(
                    "run, info, epochs, batch size, learning rate, training set size, testing set size, number of classes, train time, test time, train time per item, test time per item, test accuracy\n")
        else:
            self.num_files = len(listdir(dirname))
        name = dirname + '/run{}'.format(self.num_files)
        makedirs(name) # for storing output images
        makedirs(name + "/progress") # for storing images of reconstructions made each epoch

        dirname = "logs/simple_capsule_net_logs/" + name
        if not path.exists(dirname):
            makedirs(dirname)

        modelDetails["num_files"] = self.num_files
    #def create_models(self, modelDetails):


    def create_capsnet_model(self, inputShape, nClass, routings, caps_dim=16):
        """
        Creates a Capsule Network model.
        :param inpuShape:
        :param nClass:
        :param routings:
        :return: a Keras model
        """
        x = layers.Input(shape=inputShape)

        # layer 2: Conv2D with 'squash' activation, reshaped to [None, num_capsule, dim_capsule]
        #primaryCaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
        primaryCaps = PrimaryCap(x, dim_capsule=8, n_channels=64, kernel_size=9, strides=2, padding='valid') # edited to have twice the number of capsules

        # layer 3: capsule layer utilising routing algorithm.
        digitCaps = CapsuleLayer(num_capsule=nClass, dim_capsule=caps_dim, routings=routings, name='digitcaps')(primaryCaps)

        # layer 4: auxiliary layer replacing each capsule with its length. Supposedly not necessary for TensorFlow? Must check later.
        outCaps = Length(name='capsnet')(digitCaps)

        """# ------------DECODER network starts here------------- #
        # Note: the mask hides the output of every capsule except the chosen one. Hence reconstruction is based on the pose matrix produced by the chosen capsule.
        # When training, we use Y to decide which capsules to mask.
        # During evaluation, we use the capsule that produced the longest vector to mask the capsule output.
        y = layers.Input(shape=(nClass,))
        maskedByY = Mask()([digitCaps, y]) # true label used to mask output of capsule layer for training.
        masked = Mask()(digitCaps) # mask using the capsule with maximum length (for prediction).

        # Decoder model shared for training and prediction.
        decoder = models.Sequential(name='decoder')
        decoder.add(layers.Dense(512, activation='relu', input_dim=caps_dim*nClass))
        decoder.add(layers.Dense(np.prod(inputShape), activation='relu')) # CHANGED SIGMOID TO RELU 3
        decoder.add(layers.Dense(np.prod(inputShape), activation='linear')) # ADDED LAYER 2
        decoder.add(layers.Reshape(target_shape=inputShape, name='out_recon'))

        # models for training and evaluation
        train_model = models.Model([x, y], [outCaps, decoder(maskedByY)])
        eval_model = models.Model(x, [outCaps, decoder(masked)])

        # manipulate model
        noise = layers.Input(shape=(nClass, caps_dim))
        noisedDigitcaps = layers.Add()([digitCaps, noise])
        maskedNoisedY = Mask()([noisedDigitcaps, y])
        manipulateModel = models.Model([x, y, noise], decoder(maskedNoisedY))
        return train_model, eval_model, manipulateModel"""

        model = models.Model(x, outCaps)
        return model

    def train(self, model, data, modelVars):
        """
        Training the Capsnet
        :param model: a capsnet model
        :param data: a tuple containing training & testing data, e.g. ((x_train, y_train), (x_test, y_test))
        :param modelVars: a dictionary of the model variables
        :return: the trained model
        """
        start = time()
        # unpack the data
        (x_train, y_train), (x_test, y_test) = data
        self.trainSize = len(x_train)

        # logging stuff, worry 'bout it later
        # PLACEHOLDER DO LOGGING HERE MAYBE------------------------------
        disp_callback = DisplayCallback(modelVars) # display reconstructed images during training (after each epoch)

        # compile model
        model.compile(optimizer=optimizers.Adam(lr=modelVars['learningRate'], clipvalue=1e8),
                      loss=[self.margin_loss, 'mse'],
                      loss_weights=[1., modelVars['decoderLossCoefficient']],
                      metrics={'capsnet':'accuracy'})

        # training with data augmentation
        def train_generator(x, y, batchSize, shiftFraction=0.):
            train_datagen = ImageDataGenerator(width_shift_range=shiftFraction,
                                               height_shift_range=shiftFraction)
            generator = train_datagen.flow(x, y, batch_size=batchSize)
            while 1:
                x_batch, y_batch = generator.next()
                yield (x_batch, y_batch)

        # note: if shiftFraction=0, no data augmentation is carried out. Xifeng recommends a maximum of 2 for the mnist dataset.
        history = model.fit_generator(generator=train_generator(x_train, y_train, modelVars['batchSize'], modelVars['shiftFraction']),
                            steps_per_epoch=int(y_train.shape[0] / modelVars['batchSize']),
                            epochs=modelVars['epochs'],
                            validation_data=[x_test, y_test],
                            callbacks=[disp_callback]
                            )
        self.trainTime = time() - start
        self.store_log(history.history, modelVars['modelName'])
        model.save_weights('models/simple_capsule_models/' + modelVars['modelName'] + '_trained_capsnet_model.h5')

        return model

    def store_models(self):
        dirname = "models/simple_capsule_models/" + self.modelName
        if not path.exists(dirname):
            makedirs(dirname)

        self.model.save(dirname + '/{}_run{}.h5'.format(self.modelName, self.num_files//3))

    def test(self, model, data, modelVars):
        """
        Test the model to determine classification accuracy.
        :param model: a capsnet model
        :param data: a tuple containing testing data: (x_test, y_test)
        :param modelVars: a dictionary of the model variables
        :return:
        """
        start = time()
        dirname = "results/simple_capsule_net_results/" + modelVars['modelName'] # folder to store testing images in
        name = dirname + '/run{}'.format(self.num_files) # directory now made at initialisation
        #makedirs(name)

        print("STARTING TEST")
        (x_test, y_test) = data
        y_pred = model.predict(x_test, batch_size=modelVars['batchSize'])
        self.testTime = time() - start

        acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0]
        print("TEST ACCURACY: {}".format(acc))

        report = "{},{},{},{},{},{},{},{},{},{},{},{}\n".format(self.num_files, self.info, modelVars['epochs'],
                                                                modelVars['batchSize'], modelVars['learningRate'],
                                                                self.trainSize, len(x_test), self.trainTime,
                                                                self.testTime, self.trainTime / self.trainSize,
                                                                self.testTime / len(x_test), acc)
        with open('logs/mono_net_logs/{}/meta.csv'.format(self.modelName), 'a+') as f:
            f.write(report)


    def margin_loss(self, y_true, y_pred):
        """
        Margin loss for Eq. (4). When y_true[i, :] contains more than one '1', this should work too, although apparently it hasn't been tested yet.
        :param y_true: [None, n_classes]
        :param y_pred: [None, num_capsules]
        :return: a scalar loss value
        """
        L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
        result = K.mean(K.sum(L, 1))
        return result

    def store_log(self, history, name='test'):
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

        dirname = "logs/simple_capsule_net_logs/"+name
        if not path.exists(dirname):
            makedirs(dirname)

        with open(dirname + "/run{}-log_{:%Y-%m-%d_%H-%M-%S}_{}.csv".format(self.num_files, datetime.now(), self.modelName), 'w') as f:
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