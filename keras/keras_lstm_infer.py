# built from:
# https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# https://www.kaggle.com/kredy10/simple-lstm-for-text-classification
# https://realpython.com/python-keras-text-classification/

# ----------

filename = 'tcat_trump_multi.csv'
col_text = 'text'

# ----------

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import load_model


# read data file
df = pd.read_csv(filename,delimiter=',',encoding='utf-8')

# some info about the dataset, in particular column names
print(df.info())

# data selection and basic transformations
X = df.text					# text column

# we load the labelencoder from the training process to reestablish the labels
with open('saved_labelencoder.pickle', 'rb') as handle:
	le = pickle.load(handle)

# the tokenizer also needs to be loaded
max_len = 150
with open('saved_tokenizer.pickle', 'rb') as handle:
	tok = pickle.load(handle)

# we prepare the new data the same way as in training
sequences = tok.texts_to_sequences(X)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

# loading the trained model
model = load_model('saved_model.h5')

# infering and label translation
pred_probs = model.predict(sequences_matrix,verbose=1)
pred_classes = le.inverse_transform(pred_probs.argmax(axis=-1))

# adding the new column to the dataframe
df = df.assign(infclass=pred_classes)

# write an output
df.to_csv('output.csv', sep=',', encoding='utf-8')