
# built from:
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# ---------- modify these lines to adapt to your data:

filename_train = '../tcat_trump_full.csv'				# file to use for training
col_train_text = 'text'									# name of the text column
col_train_label = 'source'								# name of the label column

filename_infer = '../tcat_trump_full.csv'				# file to label
col_infer_text = 'text'									# name of the text column

type_classifier = 'svm'									# use 'bayes' for the multinominal Bayes classifier and 'svm' for support vector machine
use_tfidf = False										# whether to use tf*idf term weighting
frequency_cutoff = 3									# minimum frequency for word (and bigrams) to take into account



# ---------- only modify what follows if you are confident

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


# read training data file
df = pd.read_csv(filename_train,delimiter=',',encoding='utf-8')

# data selection
X = df[col_train_text].astype('U')				# text column
Y = df[col_train_label].astype(str)				# label column

# clean out the HTML for twitter source variable
Y.replace("<.+?>","", regex=True, inplace=True)

# transform categories into numbers
le = LabelEncoder()				
Y = le.fit_transform(Y)

# see: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
if use_tfidf == False:
	count_vect = CountVectorizer(ngram_range=(1,2),min_df=frequency_cutoff,stop_words='english')		# 1- and 2-gram vectorizer
else:
	count_vect = TfidfVectorizer(ngram_range=(1, 2),min_df=frequency_cutoff,stop_words='english')		# 1- and 2-gram vectorizer with tf-idf transformation (depending on the data, this may work better or not)

# vectorize and weigh training data 
X_counts = count_vect.fit_transform(X)

# train the classifier
if type_classifier == 'bayes':
	clf = MultinomialNB()
else:
	clf = SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,random_state=42,max_iter=50,tol=1e-3)

# train the model (use X_counts for unweighted and X_tfidf for tfidf)
clf.fit(X_counts,Y)

# read inference data file
df = pd.read_csv(filename_infer,delimiter=',',encoding='utf-8')

# data selection
X = df[col_infer_text].astype('U')				# text column

# vectorize and weigh training data 
X_counts = count_vect.transform(X)

# apply model to the test data
predicted = clf.predict(X_counts)

# create output to get an idea
for doc, category in zip(X, predicted):
	print('%r => %s' % (doc, le.classes_[category]))

# adding a new column with category labels to the to the dataframe
pred_classes = le.inverse_transform(predicted)
df = df.assign(inferred_label=pred_classes)

# write an output from the extended dataframe
df.to_csv('output.csv', sep=',', encoding='utf-8')