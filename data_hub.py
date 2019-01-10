"""
Selects, loads and prepares data for training
"""
import pickle
import h5py
import numpy as np
from os import listdir, walk, path
from scipy.misc import imread
from sys import stdout

def create_one_hot(val, max):
    oneHot = np.zeros(max)
    oneHot[val] = 1
    return oneHot

def load_cifar10(datasplit=0.8, datapath=""):
    dataPath = "data/cifar-10-batches-py/"
    files = listdir(dataPath)

    ims = []
    labels = []

    for file in files:
        if "." not in file:
            with open(dataPath + file, 'rb') as f:
                dict = pickle.load(f, encoding='bytes')
            ims.extend(dict[b'data'])
            labels.extend(dict[b'labels'])

    labels = np.asarray([create_one_hot(v, 10) for v in labels])
    ims = np.asarray(ims)
    reshapedIms = np.rot90(np.reshape(ims, (ims.shape[0], 32, 32, 3), order='F'), 3, axes=(1, 2))
    print(reshapedIms.shape, labels.shape)

    # 60000 images total
    trainData = (reshapedIms[:40000], labels[:40000])
    valData = (reshapedIms[40000:50000], labels[40000:50000])
    evalData = (reshapedIms[50000:], labels[50000:])

    return trainData, valData, evalData

def load_worms(datasplit=0.8, datapath=""):
    f = h5py.File("data/worms/full_mono_worm_data.hdf5")
    ftrain_x = f['train/data'][()]
    ftrain_y = f['train/labels'][()]
    feval_x = f['eval/data'][()]
    feval_y = f['eval/labels'][()]

    ftrain_x = np.reshape(ftrain_x, (ftrain_x.shape[0], 32, 32, 1), order='F')
    feval_x = np.reshape(feval_x, (feval_x.shape[0], 32, 32, 1), order='F')
    train_labels = np.asarray([create_one_hot(int(v), 2) for v in ftrain_y])
    eval_labels = np.asarray([create_one_hot(int(v), 2) for v in feval_y])

    trainsplit = int(len(ftrain_x)*datasplit)
    trainData = (ftrain_x[:trainsplit], train_labels[:trainsplit])
    valData = (ftrain_x[trainsplit:], train_labels[trainsplit:])
    evalData = (feval_x, eval_labels)

    return trainData, valData, evalData

def load_atomic(datasplit=0.8, datapath="", small=True):
    if small:
        #datapath = "e:/atomic concept images/atomic image file_5_balanced.csv"
        datapath = "e:/atomic concept images/atomic-displaced_5_balanced.csv"
    else:
        datapath = "e:/atomic concept images/atomic images/"

    basepath="e:/atomic concept images/"

    mappings={'cone':0,
              'cube':1,
              'cylinder':2,
              'sphere':3,
              'tetrahedron':4}

    with open(datapath) as f:
        data = f.readlines()

    cases = []
    classes = []

    for i in range(1, len(data)):
        vals = data[i].split(',')
        cases.append(imread(basepath + vals[0]))
        if np.amax(cases[-1]) == 0:
            print("ZEROS!")
        """for j in cases[-1]:
            if not isinstance(j, int):
                print("MAD LAD DETECTED: ", cases[-1], j, type(j))"""

        classes.append(create_one_hot(mappings[str.strip(vals[1])], 5))
        if np.amax(classes[-1]) == 0 or np.amax(classes[-1]) != 1:
            print("CLASS ZEROS!")
        for j in classes[-1]:
            if not isinstance(j, float):
                print("MAD LAD DETECTED: ", classes[-1])

    #datapath = "e:/atomic concept images/atomic-chaotic-file_5_balanced.hdf5"

    #datapath = "e:/atomic concept images/atomic image file_5_balanced.hdf5"
    #f = h5py.File(datapath, "r")

    cases, classes = np.asarray(cases), np.asarray(classes)
    cases = cases.astype(float)
    #cases, classes = f["images"], f["labels"]

    #cases, classes = cases[:len(cases)//5], classes[:len(classes)//5] # FOR TESTING ONLY, TO SPEED THINGS UP
    sub_frac = 1
    cases, classes = cases[:int(len(cases)*sub_frac)], classes[:int(len(classes)*sub_frac)]  # FOR TESTING ONLY, TO SPEED THINGS UP

    print(cases.shape)

    trainingsplit = int(len(cases) * datasplit)
    validationsplit = (len(cases) - trainingsplit) // 2

    training=(cases[:trainingsplit], classes[:trainingsplit])
    validation=(cases[trainingsplit:trainingsplit+validationsplit], classes[trainingsplit:trainingsplit+validationsplit])
    evaluation=(cases[trainingsplit+validationsplit:], classes[trainingsplit+validationsplit:])

    return training, validation, evaluation

def load_autoencoder_images(datasplit=0.8, datapath="", small=True):
    if small:
        #datapath = "e:/atomic concept images/atomic image file_5_balanced.csv"
        datapath = "e:/COCO 2017 training/train2017/vig1e3"
    else:
        datapath = "e:/COCO 2017 training/train2017/vig1e3"

    basepath="e:/COCO 2017 training/train2017/vig1e6/"

    images = listdir(basepath)

    cases = []
    classes = []

    for image in images:
        cases.append(imread(basepath + image))

    cases, classes = np.asarray(cases), np.asarray(classes)
    cases = cases.astype(float)

    sub_frac = 1
    cases, classes = cases[:int(len(cases)*sub_frac)], classes[:int(len(classes)*sub_frac)]  # FOR TESTING ONLY, TO SPEED THINGS UP

    print(cases.shape)

    trainingsplit = int(len(cases) * datasplit)
    validationsplit = (len(cases) - trainingsplit) // 2

    training=(cases[:trainingsplit], classes[:trainingsplit])
    validation=(cases[trainingsplit:trainingsplit+validationsplit], classes[trainingsplit:trainingsplit+validationsplit])
    evaluation=(cases[trainingsplit+validationsplit:], classes[trainingsplit+validationsplit:])

    return training, validation, evaluation

def load_general(datasplit=0.8, datapath="../../data/Omniglot Set/Omniglot", flatten=False, pad=True):

    examples = {}

    print("Assembling dataset...")
    count=0
    dims = []
    for dirpath, dirnames, filenames in walk(datapath):
        stdout.write("\r" + dirpath)
        stdout.flush()
        for filename in filenames:
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith(
                    '.jpeg') or filename.endswith('.JPEG'):
                item = imread(path.join(dirpath, filename))
                dims.append(item.shape[0])
                dims.append(item.shape[1])
                if item.shape[0] == 3999 or item.shape[1] == 3999:
                    print(dirpath)
                pathKey = dirpath[len(datapath):]
                if len(item.shape) == 2:
                    item = np.expand_dims(item, axis=2)
                if item.shape[2] == 1:  # i.e. if it's a greyscale image, convert it into a 3-channel image
                    item = np.concatenate((item, item, item), axis=2)

                if flatten:
                    item = item.flatten()
                item = item / 255
                if pathKey in examples.keys():
                    examples[pathKey].append((pathKey, item))
                else:
                    examples[pathKey] = [(pathKey, item)]
            elif filename.endswith('.npy'):
                item = np.load(path.join(dirpath, filename))
                dims.append(item.shape[0])
                dims.append(item.shape[1])

                if len(item.shape) == 4:
                    item = np.reshape(item, (item.shape[0], item.shape[1], item.shape[2] * item.shape[3]))
                if flatten:
                    item = item.flatten()
                if pad and (item.shape[0] < 3 or item.shape[1] < 3):
                    padded = np.zeros((3, 3, item.shape[2]))
                    padded[0:item.shape[0], 0:item.shape[1], :] = item
                    item = padded
                pathKey = dirpath[len(datapath):]
                if pathKey in examples.keys():
                    examples[pathKey].append((pathKey, item))
                else:
                    examples[pathKey] = [(pathKey, item)]

        #count += 1 ##--------------------DON'T FORGET THIS IS LIMITING THE DATASET!!!!!!!--------------
        #if count > 50:
        #    break
    print("\ndone!")
    print(max(dims), min(dims), np.mean(dims), np.median(dims))

    key_assignments = {}
    for i, key in enumerate(examples.keys()): # for neural networks, need to construct one-hot vector encodings of the classes
        key_assignments[key] = i

    print("Splitting data, using {} of the examples available for each category for training.".format(datasplit))
    training = []
    validation = []
    testing = []
    for key in examples.keys():
        num_samples = len(examples[key])
        cutoff = int(np.ceil(num_samples * datasplit))
        valsplit = (num_samples - cutoff) // 2

        training.extend(examples[key][:cutoff])
        validation.extend(examples[key][cutoff:cutoff+valsplit])
        testing.extend(examples[key][cutoff+valsplit:])
        print('.',end='')
    np.random.shuffle(training)
    np.random.shuffle(testing)
    print("\ndone!")

    train = (np.asarray([a[1] for a in training]), np.asarray([create_one_hot(key_assignments[a[0]], len(examples.keys())) for a in training]))
    valid = (np.asarray([a[1] for a in validation]), np.asarray([create_one_hot(key_assignments[a[0]], len(examples.keys())) for a in validation]))
    eval = (np.asarray([a[1] for a in testing]), np.asarray([create_one_hot(key_assignments[a[0]], len(examples.keys())) for a in testing]))

    return train, valid, eval





def get_data(target="general", datasplit=0.8, datapath=""):
    assignments = {
        'cifar': load_cifar10,
        'worms': load_worms,
        'atomic': load_atomic,
        'auto': load_autoencoder_images,
        'general': load_general
    }
    return assignments[target](datasplit=datasplit, datapath=datapath)
