from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from os import listdir, makedirs, path

import numpy as np

class KNN():
    def __init__(self, modelDetails, modelName, modelPath):
        self.model = None
        self.training_size=0
        self.testing_size=0

        dirname = "logs/kNN_logs/" + modelDetails['modelName']
        if not path.exists(dirname):
            self.num_files = 0
            makedirs(dirname)
        else:
            self.num_files = len(listdir(dirname))

        modelDetails["num_files"] = self.num_files
        self.dirname = dirname

    def train(self, data, modelVars):
        (x_train, y_train), (x_test, y_test) = data
        self.model = KNeighborsClassifier(n_neighbors= modelVars['k'], algorithm=modelVars['algorithm'])
        self.model.fit(x_train, y_train)

        print("kNN model trained!")
        self.training_size = x_train.shape[0]

    def test(self, data, modelVars):
        (x_test, y_test) = data
        y_pred = self.model.predict(x_test)
        acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0]
        print("Test accuracy: {}".format(acc))
        self.testing_size = x_test.shape[0]

        report = "training set size:,{}\ntesting set size:,{}\ntest accuracy:,{}".format(self.training_size, self.testing_size, acc)
        with open(self.dirname + "/run{}-result_{:%Y-%m-%d_%H-%M-%S}.csv".format(self.num_files, datetime.now()), 'w') as f:
            f.write(report)