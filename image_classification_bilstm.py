from __future__ import print_function, division
from builtins import range, input
# sudo pip install -U future

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
	if not os.path.exists('../resource/large_files/train.csv'):
		print("Look like you haven't downloaded the data or it's not in the right spot.")
		print("Please get train.csv from https://www.kaggle.com/c/digit-recogenizer")
		print("and please it in the larget_files folder.")

	print("Reading in the transforming data...")
	df = pd.read_csv('../resource/large_files/train.csv')
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
input_ = Input(shape=(D, D))

#up-down
rnn1 = Bidirectional(LSTM(M, return_sequences=True))
x1 = rnn1(input_) #output is N x D x 2M
x1 = GlobalMaxPooling1D()(x1) # output is N x 2M

#left-right
rnn2 = Bidirectional(LSTM(M, return_sequences=True))

#custom layer
permutor = Lambda(lambda t: K.permute_dimensions(t, pattern=(0, 2, 1)))

x2 = permutor(input_)
x2 = rnn2(x2) # output is N x D x 2M
x2 = GlobalMaxPooling1D()(x2) # output is N x 2M

# put them together
concatenator = Concatenate(axis=1)
x = concatenator([x1, x2]) # output is N x 4M

# final dense layer
output = Dense(10, activation='softmax')(x)
model = Model(input=input_, output=output)

# testing
# o = model.predict(X)
# print("o.shape:", o.shape)

# compile
model.compile(
	loss='sparse_categorical_crossentropy',
	optimizer='adam',
	metrics=['accuracy']
)

# train
print('Training model...')
r = model.fit(X, Y, batch_size=32, validation_split=0.3)

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()




