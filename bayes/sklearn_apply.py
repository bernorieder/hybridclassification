
# built from:
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# ----------

filename_train = '../tcat_trump_full.csv'
col_train_text = 'text'
col_train_label = 'source'

filename_infer = '../tcat_trump_full.csv'
col_infer_text = 'text'

# ----------

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


# read training data file
df = pd.read_csv(filename_train,delimiter=',',encoding='utf-8')

# data selection
X = df[col_train_text]				# text column
Y = df[col_train_label]				# label column

# clean out the HTML for twitter source variable
Y.replace("<.+?>","", regex=True, inplace=True)

# transform categories into numbers
le = LabelEncoder()				
Y = le.fit_transform(Y)

# objects for the vectorizer and the frequency transformer
count_vect = CountVectorizer(stop_words='english')
tfidf_transformer = TfidfTransformer()

# vectorize and weigh training data 
X_counts = count_vect.fit_transform(X)
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# train the classifier (select either multinominar naive Bayes or Support Vector Machine)
clf = MultinomialNB()
clf = SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,random_state=42,max_iter=20,tol=None)

# train the model (use X_counts for unweighted and X_tfidf for tfidf)
clf.fit(X_counts,Y)


# read inference data file
df = pd.read_csv(filename_infer,delimiter=',',encoding='utf-8')

# data selection
X = df[col_infer_text]				# text column

# vectorize and weigh training data 
X_counts = count_vect.transform(X)
X_tfidf = tfidf_transformer.transform(X_counts)

# apply model to the test data (use X_counts for unweighted and X_tfidf for tfidf)
predicted = clf.predict(X_counts)

# create output to get an idea
for doc, category in zip(X, predicted):
	print('%r => %s' % (doc, le.classes_[category]))

# adding a new column with category labels to the to the dataframe
pred_classes = le.inverse_transform(predicted)
df = df.assign(infclass=pred_classes)

# write an output from the extended dataframe
df.to_csv('output.csv', sep=',', encoding='utf-8')