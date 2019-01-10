"""
Selects desired model, initialises it and sends it away for training.
"""
import numpy as np
from os.path import isfile

from matplotlib import  pyplot as plt
from PIL import  Image

from keras.utils import plot_model
from keras import optimizers
from keras import backend as K

from architectures.capsule_net.capsnet import CapsuleNetwork
from architectures.convolutional_net.small_convnet import ConvolutionalNetwork
from architectures.general import GeneralNetwork
from architectures.autoencoder.autoencoder import Autoencoder
from architectures.simple.kNN import KNN
from architectures.simple.monolayer import MonoNetwork
from architectures.simple.simple_capsnet import SimpleCapsuleNetwork

class ModelManager():
    def __init__(self, modelDetails, modelName, modelPath=None):
        """
        Given the details of a model to instantiate, instantiate the appropriate model, load any relevant info,
        establish storing of model data and return the compiled model, ready to rumble.
        :param modelDetails: Dictionary containing model type and, optionally, hyperparameter values of the chosen model.
        :param modelName: Name to use when storing information about the model.
        :param modelPath: Path to weights of trained model; use to continue training an already trained model
        :return: an initialised and compiled keras model
        """
        self.modelPath = modelPath
        self.mType = modelDetails['mType']
        self.modelVars = self.collate_data(modelDetails)
        modelSelection = {
            'capsnet': {
                'initialise': CapsuleNetwork,
                'train': train_capsnet,
                'test': test_capsnet
            },
            'convnet': {
                'initialise': ConvolutionalNetwork,
                'train': train_convnet,
                'test': test_convnet,
                'assess': assess_convnet
            },
            'gennet': {
                'initialise': GeneralNetwork,
                'train': train_gennet,
                'test': test_gennet,
                'assess': assess_gennet
            },
            'autoencoder': {
                'initialise': Autoencoder,
                'train': train_autoencoder,
                'test': test_autoencoder,
                'assess': assess_autoencoder
            },
            'knn': {
                'initialise': KNN,
                'train': train_knn,
                'test': test_knn
            },
            'mononet': {
                'initialise': MonoNetwork,
                'train': train_mononet,
                'test': test_mononet
            },
            'simplecaps': {
                'initialise': SimpleCapsuleNetwork,
                'train': train_simplecaps,
                'test': test_simplecaps
            }
        }
        self.modelOptions = modelSelection[self.mType]
        self.Learner = self.modelOptions['initialise'](modelDetails, modelName, modelPath)

    def collate_data(self, values):
        """
        Describes all the variables associated with each model type, and assigns their default values. Produces a dictionary where default values have been overwritten by those values provided by user.
        :param values: Dictionary of values supplied by user. Not all variables have to be contained here; missing variables will be filled in with default values.
        :return: dictionary containing all variables required by the selected model.
        """
        modelVariables = {
            'capsnet': {
                'learningRate': values.get('learningRate', 0.001),
                'decoderLossCoefficient': values.get('decoderLossCoefficient', 0.                                                                                                  ),
                'shiftFraction': values.get('shiftFraction', 0.),
                'batchSize': values.get('batchSize', 256),
                'epochs': values.get('epochs', 10),
                'modelName': values.get('modelName', 'model'),
                'inputShape': values.get('inputShape', np.asarray([32, 32, 3])),
                'nClass': values.get('nClass', 10),
                'routings': values.get('routings', 3)
            },
            'convnet': {
                'learningRate': values.get('learningRate', 0.001),
                'batchSize': values.get('batchSize', 256),
                'epochs': values.get('epochs', 10),
                'modelName': values.get('modelName', 'model'),
                'inputShape': values.get('inputShape', np.asarray([32, 32, 1])),
                'nClass': values.get('nClass', 10)
            },
            'gennet': {
                'learningRate': values.get('learningRate', 0.001),
                'batchSize': values.get('batchSize', 256),
                'epochs': values.get('epochs', 10),
                'modelName': values.get('modelName', 'model'),
                'inputShape': values.get('inputShape', np.asarray([32, 32, 3])),
                'nClass': values.get('nClass', 10)
            },
            'autoencoder': {
                'learningRate': values.get('learningRate', 0.001),
                'batchSize': values.get('batchSize', 256),
                'epochs': values.get('epochs', 10),
                'modelName': values.get('modelName', 'model'),
                'inputShape': values.get('inputShape', np.asarray([32, 32, 3]))
            },
            'knn': {
                'modelName': values.get('modelName', 'model'),
                'k': values.get('k', 5),
                'algorithm' : values.get('algorithm', 'ball_tree')
            },
            'mononet': {
                'learningRate': values.get('learningRate', 0.001),
                'batchSize': values.get('batchSize', 128),
                'epochs': values.get('epochs', 10),
                'modelName': values.get('modelName', 'model'),
                'inputShape': values.get('inputShape', np.asarray([32, 32, 3])),
                'nClass': values.get('nClass', 10)
            },
        }
        return modelVariables[self.mType]

    def train(self, data):
        self.modelOptions['train'](self.Learner, self.modelVars, data, self.modelPath)

    def test(self, data):
        self.modelOptions['test'](self.Learner, self.modelVars, data, self.modelPath)

    def assess(self, data):
        self.modelOptions['assess'](self.Learner, self.modelVars, data, self.modelPath)

def train_capsnet(Learner, modelVars, data, modelPath):
    """
    Manage training for the capsule network
    :param Learner: a CapsuleNetwork object from capsnet.py
    :param modelVars: dictionary of the model variable values
    :param data: a tuple containing training & testing data, e.g. ((x_train, y_train), (x_test, y_test))
    :param modelPath: path of the model weights to use (if None, a new model is trained)
    :return:
    """
    #print("HI THERE", data[0][0].shape[1:])
    #Learner.create_models(modelVars, data[0][0].shape[1:])

    model = Learner.trainModel

    """if modelPath is not None and isfile(modelPath):  # init the model weights with provided one
        model.load_weights(modelPath)""" # handled by initialisation now
    Learner.train(model=model, data=data, modelVars=modelVars)

def test_capsnet(Learner, modelVars, data, modelPath):
    """
    Manage testing/evaluation for the capsule network
    :param Learner: a CapsuleNetwork object from capsnet.py
    :param modelVars: dictionary of the model variable values
    :param data: data to test on; tuple (x_test, y_test)
    :param modelPath: path of the model weights to use (if None, a new model is instantiated)
    :return:
    """
    #x_test, y_test = data
    eval_model, manipulate_model = Learner.evalModel, Learner.manipModel
    if modelPath is None:
        print('No weights are provided. Will test using random initialized weights.')
    #manipulate_latent(manipulate_model, (x_test, y_test), modelVars)
    Learner.test(model=eval_model, data=data, modelVars=modelVars)

def train_convnet(Learner, modelVars, data, modelPath):
    """
    Manage training for convolutional network
    :param Learner: a ConvolutionalNet object
    :param modelVars: dictionary of model variable values
    :param data: a tuple containing training & testing data, e.g. ((x_train, y_train), (x_test, y_test))
    :param modelPath: path of the model weights to use (if None, a new model is trained)
    :return:
    """
    model = Learner.model
    if modelPath is not None and isfile(modelPath):  # init the model weights with provided one
        model.load_weights(modelPath)
    Learner.train(model=model, data=data, modelVars=modelVars)

def test_convnet(Learner, modelVars, data, modelPath):
    """
    Manage testing/evaluation of convolutional network
    :param Learner: a ConvolutionalNet object
    :param modelVars: dictionary of model variable values
    :param data: a tuple containing training & testing data, e.g. ((x_train, y_train), (x_test, y_test))
    :param modelPath: path of the model weights to use (if None, a new model is trained)
    :return:
    """
    eval_model = Learner.model
    if modelPath is None:
        print('No weights are provided. Will test using random initialized weights.')
    Learner.test(model=eval_model, data=data, modelVars=modelVars)

def assess_convnet(Learner, modelVars, data, modelPath):
    """
    hopefully visualise layers
    :param Learner:
    :param modelVars:
    :param data:
    :param modelPath:
    :return:
    """
    (x_test, y_test) = data
    model = Learner.model
    if modelPath is not None and isfile(modelPath):  # init the model weights with provided one
        model.load_weights(modelPath)

    model.compile(optimizer=optimizers.Adam(lr=0.00001), loss='mse', metrics=['accuracy'])
    Learner.test(model=model, data=data, modelVars=modelVars)

    id=7
    img_to_visualise = x_test[id]
    print(y_test[id])
    vis_img_to_visualise = np.resize(img_to_visualise, (32, 32))
    print(np.uint8(vis_img_to_visualise.T))
    print(vis_img_to_visualise.shape)
    vis_img_to_visualise = Image.fromarray(np.uint8(vis_img_to_visualise))
    plt.imshow(vis_img_to_visualise, cmap='gray')
    plt.savefig('results/example'+str(id)+'.png', dpi=900)
    plt.show()
    img_to_visualise = np.expand_dims(img_to_visualise, axis=0)
    target = 'conv1_act'
    layer_to_test = model.get_layer(target)
    layer_to_visualize(model, layer_to_test, img_to_visualise, target)

def train_gennet(Learner, modelVars, data, modelPath):
    """
    Manage training for convolutional network
    :param Learner: a ConvolutionalNet object
    :param modelVars: dictionary of model variable values
    :param data: a tuple containing training & testing data, e.g. ((x_train, y_train), (x_test, y_test))
    :param modelPath: path of the model weights to use (if None, a new model is trained)
    :return:
    """
    model = Learner.model
    if modelPath is not None and isfile(modelPath):  # init the model weights with provided one
        model.load_weights(modelPath)
    Learner.train(model=model, data=data, modelVars=modelVars)

def test_gennet(Learner, modelVars, data, modelPath):
    """
    Manage testing/evaluation of convolutional network
    :param Learner: a ConvolutionalNet object
    :param modelVars: dictionary of model variable values
    :param data: a tuple containing training & testing data, e.g. ((x_train, y_train), (x_test, y_test))
    :param modelPath: path of the model weights to use (if None, a new model is trained)
    :return:
    """
    eval_model = Learner.model
    if modelPath is None:
        print('No weights are provided. Will test using random initialized weights.')
    Learner.test(model=eval_model, data=data, modelVars=modelVars)

def assess_gennet(Learner, modelVars, data, modelPath):
    """
    hopefully visualise layers
    :param Learner:
    :param modelVars:
    :param data:
    :param modelPath:
    :return:
    """
    (x_test, y_test) = data
    model = Learner.model
    if modelPath is not None and isfile(modelPath):  # init the model weights with provided one
        model.load_weights(modelPath)

    model.compile(optimizer=optimizers.Adam(lr=0.00001), loss='mse', metrics=['accuracy'])
    Learner.test(model=model, data=data, modelVars=modelVars)

def train_autoencoder(Learner, modelVars, data, modelPath):
    """
    Manage training for autoencoder
    :param Learner: an Autoencoder object
    :param modelVars: dictionary of model variable values
    :param data: a tuple containing training & testing data, e.g. ((x_train, y_train), (x_test, y_test))
    :param modelPath: path of the model weights to use (if None, a new model is trained)
    :return:
    """
    model = Learner.autoencoder
    if modelPath is not None and isfile(modelPath):  # init the model weights with provided one
        model.load_weights(modelPath)
    Learner.train(model=model, data=data, modelVars=modelVars)

def test_autoencoder(Learner, modelVars, data, modelPath):
    """
    Manage testing/evaluation of convolutional network
    :param Learner: a ConvolutionalNet object
    :param modelVars: dictionary of model variable values
    :param data: a tuple containing training & testing data, e.g. ((x_train, y_train), (x_test, y_test))
    :param modelPath: path of the model weights to use (if None, a new model is trained)
    :return:
    """
    eval_model = Learner.autoencoder
    if modelPath is None:
        print('No weights are provided. Will test using random initialized weights.')
    Learner.test(model=eval_model, data=data, modelVars=modelVars)

def assess_autoencoder(Learner, modelVars, data, modelPath):
    """
    hopefully visualise layers
    :param Learner:
    :param modelVars:
    :param data:
    :param modelPath:
    :return:
    """
    (x_test, y_test) = data
    model = Learner.autoencoder
    if modelPath is not None and isfile(modelPath):  # init the model weights with provided one
        model.load_weights(modelPath)

    model.compile(optimizer=optimizers.Adam(lr=0.00001), loss='mse', metrics=['accuracy'])
    Learner.test(model=model, data=data, modelVars=modelVars)

def train_knn(Learner, modelVars, data, modelPath):
    Learner.train(data, modelVars)

def test_knn(Learner, modelVars, data, modelPath):
    Learner.test(data, modelVars)

def train_mononet(Learner, modelVars, data, modelPath):
    """
    Manage training for mono-layer network
    :param Learner: a MonoNetwork object
    :param modelVars: dictionary of model variable values
    :param data: a tuple containing training & testing data, e.g. ((x_train, y_train), (x_test, y_test))
    :param modelPath: path of the model weights to use (if None, a new model is trained)
    :return:
    """
    model = Learner.model
    Learner.train(model=model, data=data, modelVars=modelVars)

def test_mononet(Learner, modelVars, data, modelPath):
    """
    Manage testing/evaluation of mono-layer network
    :param Learner: a MonoNetwork object
    :param modelVars: dictionary of model variable values
    :param data: a tuple containing training & testing data, e.g. ((x_train, y_train), (x_test, y_test))
    :param modelPath: path of the model weights to use (if None, a new model is trained)
    :return:
    """
    eval_model = Learner.model
    if modelPath is None:
        print('No weights are provided. Will test without pre-trained weights.')
    Learner.test(model=eval_model, data=data, modelVars=modelVars)

def train_simplecaps(Learner, modelVars, data, modelPath):
    """
    Manage training for simple capsule network
    :param Learner: a SimpleCapsuleNetwork object
    :param modelVars: dictionary of model variable values
    :param data: a tuple containing training & testing data, e.g. ((x_train, y_train), (x_test, y_test))
    :param modelPath: path of the model weights to use (if None, a new model is trained)
    :return:
    """
    model = Learner.model
    Learner.train(model=model, data=data, modelVars=modelVars)

def test_simplecaps(Learner, modelVars, data, modelPath):
    """
    Manage testing/evaluation of simple capsule network
    :param Learner: a SimpleCapsuleNetwork object
    :param modelVars: dictionary of model variable values
    :param data: a tuple containing training & testing data, e.g. ((x_train, y_train), (x_test, y_test))
    :param modelPath: path of the model weights to use (if None, a new model is trained)
    :return:
    """
    eval_model = Learner.model
    if modelPath is None:
        print('No weights are provided. Will test without pre-trained weights.')
    Learner.test(model=eval_model, data=data, modelVars=modelVars)

def layer_to_visualize(model, layer, img_to_visualise, name):
    """
    ripped straight from https://github.com/yashk2810/Visualization-of-Convolutional-Layers/blob/master/Visualizing%20Filters%20Python3%20Theano%20Backend.ipynb
    :param layer:
    :return:
    """
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])

    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualise)
    convolutions = np.squeeze(convolutions)

    print('Shape of conv:', convolutions.shape)

    convolutions = convolutions.T

    n = convolutions.shape[0]
    n = int(np.ceil(np.sqrt(n)))

    # Visualization of each filter of the layer
    """fig = plt.figure(figsize=(12, 8))
    for i in range(len(convolutions)):
        ax = fig.add_subplot(n, n, i + 1)
        ax.imshow(convolutions[i], cmap='hot')
    fig.savefig(name + '_testing-vis.png', dpi=900)
    plt.show()"""
    """plt.imshow(convolutions[2], cmap='hot')
    plt.savefig('results/' + name + '_2_single_testing-vis.png', dpi=900)
    plt.show()
    plt.imshow(convolutions[3], cmap='hot')
    plt.savefig('results/' + name + '_3_single_testing-vis.png', dpi=900)
    plt.show()
    plt.imshow(convolutions[4], cmap='hot')
    plt.savefig('results/' + name + '_4_single_testing-vis.png', dpi=900)
    plt.show()
    plt.imshow(convolutions[5], cmap='hot')
    plt.savefig('results/' + name + '_5_single_testing-vis.png', dpi=900)
    plt.show()
    plt.imshow(convolutions[6], cmap='hot')
    plt.savefig('results/' + name + '_6_single_testing-vis.png', dpi=900)
    plt.show()"""
    for i in range(len(convolutions)):
        print(i, end='...')
        plt.imshow(convolutions[i], cmap='hot')
        plt.savefig('results/conv1/' + name + '_' + str(i) + '_single_testing-vis.png', dpi=900)
        plt.show()
