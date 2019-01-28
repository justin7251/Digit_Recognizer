from _future_ import print_function, division
from builtins import range, input

import os
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Bidirectional, GlobalMaxPooling1D, Lambda, Concatenate, Dense
import keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_mnist(limit=None):
	if not os.path.exists('../resource/large_files'):
		print("You must create a folder called large_files adjacent to the class folder first.")
	if not os.path.exsits('../resource/large_files/train.csv'):
		print("Look like you haven't downloaded the data or it's not in the right spot.")
		print("Please get train.csv from https://www.kaggle.com/c/digit-recogenizer")
		print("and please it in the larget_files folder.")

	print("Reading in the transforming data...")
	af = pd.read_csv('../resource/large_files/train.csv')
	data = df.as_matrix()
	np.random.shuffle(data)
	X = data[:, 1:].reshape(-1, 28, 28) / 255.0 # data is from 0..255
	Y = data[:, 0]
	if limit is not None:
		X, Y = X[:limit], Y[:limit]
	return X, Y

# get data
X, Y = get_mnist()

# config
D = 28 # image size 28 x 28
M = 15 # change if needed

# input is an image of size 28x28
input_ = Input(shape(D, D))

#up-down
rnn1 = Bidirectional(LSTM(M, return_sequences=True))
x1 = rnn(input_) #output is N x D x 2M
x1 = GlobalMaxPooling1D()(x1) # output is N x 2M

#left-right




