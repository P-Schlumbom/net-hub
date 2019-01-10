"""
This should be used for actual training.
Manages everything from data pre-processing, training regimes and the recording of results.
"""

import sys
import json
import numpy as np
from model_hub import ModelManager
from data_hub import get_data
from time import time
from datetime import timedelta

def create_one_hot(val, max):
    oneHot = np.zeros(max)
    oneHot[val] = 1
    return oneHot

def json_to_dict(jsonpath):
    """
    Given path to some json file, returns dictionary stored there.
    :param jsonpath: Path of .json file
    :return: dictionary stored by that .json file.
    """
    try:
        with open(jsonpath) as data:
            jsondata = json.loads(data.read())
        return jsondata
    except:
        print("ERROR COULD NOT LOAD JSON")
        return None

def get_json_values(jsonpath):
    """
    Given the path to a json file, uses it to fill out the values of a dictionary and returns the dictionary.
    :param jsonpath:
    :return: dictionary of values
    """
    try:
        jsondata = json_to_dict(jsonpath)
        if 'dataPath' not in jsondata.keys():
            print("ERROR: 'datapath' NOT FOUND IN JSON FILE")
            a = 1/0
        if 'epochs' in jsondata.keys() and 'steps' in jsondata.keys():
            del(jsondata['steps'])
        jsondata['modelDetails']['modelName'] = jsondata['modelDetails'].get('modelName', jsondata['modelName'])
        jsondata['modelDetails']['batchSize'] = jsondata['modelDetails'].get('batchSize', jsondata['batchSize'])
        if 'epochs' in jsondata.keys():
            jsondata['modelDetails']['epochs'] = jsondata['modelDetails'].get('epochs', jsondata['epochs'])
        else:
            jsondata['modelDetails']['steps'] = jsondata['modelDetails'].get('steps', jsondata['steps'])
        return jsondata
    except:
        if 'y' in input("WARNING COULD NOT LOAD JSON. ENTER PATH MANUALLY? y/n: "):
            return get_json_values(input("Enter path of json file: "))
        else:
            print("STOPPING")
            sys.exit()

def get_setup_values():
    """
    Uses stdin input to determine training parameters. Parameters can be manually entered, or the path to a json
    file can be provided that contains all relevant information.
    :return: a dictionary of acquired parameters
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datapath', type=str, default='__', help='The path of the dataset.')
    parser.add_argument('-m', '--modelpath', type=str, default='__', help='Path to model to use for training. If no value is provided, a new model will be trained.')
    parser.add_argument('-n', '--modelname', type=str, default='trained_model', help='Name to used to identify stored models and data logs.')
    parser.add_argument('-b', '--batchsize', type=int, default=256, help='Size of the batches to train on.')
    parser.add_argument('-e', '--epochs', type=int, default=-1, help='The number of epochs to train for.')
    parser.add_argument('-s', '--steps', type=int, default=100, help='Number of steps to train on. Mutually exclusive with epochs, choose one or the other. Epochs will be given priority if both are provided.')
    parser.add_argument('-j', '--jsonpath', type=str, default='__', help='Path to json file of parameters. If this is provided, all other entries WILL BE IGNORED.')
    parser.add_argument('-md', '--modeldetails', type=str, default='__', help='Path to json file containing details of the model to load, i.e. model type and optionally, hyperparameters.')
    parser.add_argument('-dd', '--datadetails', type=str, default='__', help='Path to json file containing details of training regime. Optional; standard batch-wise training with uniform sampling carried out if nothing is provided.')
    args = parser.parse_args()

    if args.jsonpath != '__':
        return get_json_values(args.jsonpath)
    else:
        # get datapath
        if args.datapath == '__':
            print("ERROR: NO DATAPATH PROVIDED. RUN WITH -h FOR LIST OF INPUT ARGUMENTS.")
            sys.exit()
        values_dict = {}
        values_dict['dataPath'] = args.datapath

        # get modelpath
        if args.modelpath != '__':
            values_dict['modelPath'] = args.modelpath

        # get model details
        if args.modeldetails == '__':
            print("ERROR NO MODEL DETAILS PROVIDED. ENTER PATH TO JSON FILE CONTAINING RELEVANT DATA, OR ALTERNATIVELY ENTER NAME OF MODEL TYPE TO USE.")
            sys.exit()
        else:
            if '.json' not in args.modeldetails:
                values_dict['modelDetails'] = {'mType': args.modeldetails}
            else:
                values_dict['modelDetails'] = json_to_dict(args.modeldetails)
                if values_dict['modelDetails'] == None:
                    sys.exit()

        # get data details
        if args.datadetails != '__':
            if '.json' not in args.datadetails:
                values_dict['dataDetails'] = {'mType': args.datadetails}
            else:
                values_dict['dataDetails'] = json_to_dict(args.datadetails)
                if values_dict['dataDetails'] == None:
                    sys.exit()

        # get model name
        values_dict['modelName'] = args.modelname
        values_dict['modelDetails']['modelName'] = values_dict['modelDetails'].get('modelName', args.modelname)

        # get batch size
        values_dict['batchSize'] = args.batchsize
        values_dict['modelDetails']['batchSize'] = values_dict['modelDetails'].get('batchSize', args.batchsize)

        # get epochs or steps
        if args.epochs != -1:
            values_dict['epochs'] = args.epochs
            values_dict['modelDetails']['epochs'] = values_dict['modelDetails'].get('epochs', args.epochs)
        else:
            values_dict['steps'] = args.steps
            values_dict['modelDetails']['steps'] = values_dict['modelDetails'].get('steps', args.steps)

        return values_dict

def load_data(dataPath):
    """
    Loads data stored at some location and returns it. Currently hard-coded to load CIFAR-10 dataset.
    :param dataPath: path to dataset(s)
    :return: train, validation and evaluation data
    """
    trainsplit = 0.8

    dataPath="data/cifar-10-batches-py/data_batch_1"
    import pickle
    with open(dataPath, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    ims = dict[b'data']
    labels = np.asarray([create_one_hot(v, 10) for v in dict[b'labels']])
    print(labels.shape)
    reshapedIms = np.rot90(np.reshape(ims, (ims.shape[0], 32, 32, 3), order='F'), 3, axes=(1, 2))

    trainsplit = int(trainsplit*len(reshapedIms))
    valsplit = (len(reshapedIms) - trainsplit) // 2

    trainData = (reshapedIms[:trainsplit], labels[:trainsplit])
    valData = (reshapedIms[trainsplit:trainsplit + valsplit], labels[trainsplit:trainsplit + valsplit])
    evalData = (reshapedIms[trainsplit + valsplit:], labels[trainsplit + valsplit:])

    return trainData, valData, evalData

def auto_tester(valuesDict):
    datapaths = ["../data/tiny-imagenet-200/train", "../data/tiny-imagenet-200_sift/train", "../data/tiny-imagenet-200_surf/train", "../data/tiny-imagenet-200_autoencoder/train", 
    "../data/tiny-imagenet-200_capsencoder/train", "../data/Omniglot", "../data/Omniglot_autoencoder", "../data/Omniglot_capsencoder", "../data/Omniglot_sift", "../data/Omniglot_surf",
    "../data/101_ObjectCategories_im-cropped", "../data/101_ObjectCategories_im_autoencoder-cropped", "../data/101_ObjectCategories_im_capsencoder-cropped", "../data/101_ObjectCategories_im_sift-cropped", "../data/101_ObjectCategories_im_surf-cropped"]

    #lrs = json_to_dict("inputs/references/mononet_lrs.json")
    lrs = json_to_dict(valuesDict['dataPath'])

    #datapaths = ["../data/101_ObjectCategories_im_capsencoder-cropped"]
    datapaths = list(lrs.keys()) # allows for the deletion of entries...

    for datapath in datapaths:
        for i in range(1, 10):
            datasplit = i * 0.1
            trainData, validationData, evaluationData = get_data(valuesDict['dataType'], datasplit, datapath)

            print("using lr={}".format(lrs[datapath]))
            valuesDict['modelDetails']['learningRate'] = lrs[datapath]
            valuesDict['modelDetails']['info'] = datapath + '--{}'.format(datasplit)
            valuesDict['modelDetails']['inputShape'] = trainData[0][0].shape
            valuesDict['modelDetails']['nClass'] = trainData[1][0].shape[0]

            # prepare model
            manager = ModelManager(valuesDict['modelDetails'], valuesDict['modelName'],
                                   valuesDict.get('modelPath', None))

            # train model
            start = time()
            manager.train((trainData, validationData))
            end = time()
            print("time taken to train: {}".format(timedelta(seconds=int(end - start))))

            # test model
            start = time()
            manager.test(evaluationData)
            end = time()
            print("time taken to test: {}".format(timedelta(seconds=int(end - start))))

def run_function(valuesDict, trainData, validationData, evaluationData, datasplit=0.1):
    valuesDict['modelDetails']['info'] = valuesDict['dataPath'] + '--{}'.format(datasplit)
    valuesDict['modelDetails']['inputShape'] = trainData[0][0].shape
    valuesDict['modelDetails']['nClass'] = trainData[1][0].shape[0]

    # prepare model
    manager = ModelManager(valuesDict['modelDetails'], valuesDict['modelName'], valuesDict.get('modelPath', None))

    # train model
    start = time()
    manager.train((trainData, validationData))
    end = time()
    print("time taken to train: {}".format(timedelta(seconds=int(end - start))))

    # test model
    start = time()
    manager.test(evaluationData)
    end = time()
    print("time taken to test: {}".format(timedelta(seconds=int(end - start))))
    # manager.assess(evaluationData)


def custom_expt(valuesDict):
    print("starting custom experiment:")
    datasplit = 0.1
    # load data
    trainData, validationData, evaluationData = get_data(valuesDict['dataType'], datasplit, valuesDict['dataPath'])
    lr=1e-1
    for i in range(6):
        print("using lr={}".format(lr))
        valuesDict['modelDetails']['learningRate'] = lr
        run_function(valuesDict, trainData, validationData, evaluationData)
        lr *=0.1

if __name__ == '__main__':
    """
    main function
    """

    # get starting values
    valuesDict = get_setup_values()
    autorun = False
    custom_run=False
    if autorun:
        auto_tester(valuesDict)
    elif custom_run:
        custom_expt(valuesDict)
    else:
        datasplit = 0.8
        # load data
        trainData, validationData, evaluationData = get_data(valuesDict['dataType'], datasplit, valuesDict['dataPath'])

        valuesDict['modelDetails']['info'] = valuesDict['dataPath'] + '--{}'.format(datasplit)
        valuesDict['modelDetails']['inputShape'] = trainData[0][0].shape
        valuesDict['modelDetails']['nClass'] = trainData[1][0].shape[0]

        # prepare model
        manager = ModelManager(valuesDict['modelDetails'], valuesDict['modelName'], valuesDict.get('modelPath', None))

        # train model
        start = time()
        manager.train((trainData, validationData))
        end=time()
        print("time taken to train: {}".format(timedelta(seconds=int(end - start))))

        # test model
        start = time()
        manager.test(evaluationData)
        end = time()
        print("time taken to test: {}".format(timedelta(seconds=int(end - start))))
        #manager.assess(evaluationData)


    #auto_tester(valuesDict)
