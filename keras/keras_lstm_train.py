# working with:
# https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# https://www.kaggle.com/kredy10/simple-lstm-for-text-classification
# https://realpython.com/python-keras-text-classification/


# ----------

filename = '../tcat_trump_multi.csv'
col_text = 'text'
col_label = 'source'

# ----------

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


# read data file
df = pd.read_csv(filename,delimiter=',',encoding='utf-8')

# clean out the HTML for twitter source variable
df.source.replace("<.+?>","", regex=True, inplace=True)

# some info about the dataset, in particular column names
print(df.info())

# vlaue stats of the source column
print(df.source.value_counts())
no_categories = len(df.source.value_counts())


# data selection and basic transformations
X = df[col_text]				# text column
Y = df[col_label]				# label column

le = LabelEncoder()				# transforms categories into numbers
Y = le.fit_transform(Y)
Y = to_categorical(Y)

# we need to save the label encoder
with open('saved_labelencoder.pickle','wb') as handle:							# save the tokenizer word<=>id pairs for inference
	pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

# cutting the dataset into training (85%) and testing (15%) data 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)

# preparing word dict
max_len = 150									# maximum length of sequence
max_words = 10000								# how big the vocabulary should be; "the maximum number of words to keep, based on word frequency" https://keras.io/preprocessing/text/
tok = Tokenizer(num_words=max_words)			# more info on Keras' somewhat basic tokenizer: https://keras.io/preprocessing/text/
tok.fit_on_texts(X)
with open('saved_tokenizer.pickle', 'wb') as handle:							# save the tokenizer word<=>id pairs for inference
	pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

# defining model
model = Sequential()
model.add(Embedding(max_words, 32, input_length=max_len))				# for the embedding layer see https://keras.io/layers/embeddings/
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(no_categories, activation='softmax'))

# compiling model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# fitting model: for parameters explanations check https://keras.io/models/model/
model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

# evaluate the model with test data
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
accr = model.evaluate(test_sequences_matrix,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

model.save('saved_model.h5')