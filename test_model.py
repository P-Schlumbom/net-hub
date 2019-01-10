import numpy as np
from keras.models import load_model
from scipy.misc import imread
from sys import argv

modelpath = argv[1]
imagepath = argv[2]

im = imread(imagepath)
im_input = np.asarray([im])

model = load_model(modelpath)

prediction = model.predict(im_input)

print(prediction)