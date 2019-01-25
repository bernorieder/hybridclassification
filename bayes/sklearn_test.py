# built on the basis of:
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# ---------- modify the following lines to adapt to your data

filename = '../tcat_trump_full.csv'						# file to use for training and testing
col_text = 'text'										# name of the text column
col_label = 'source'									# name of the label column
type_classifier = 'svm'									# use 'bayes' for the multinominal Bayes classifier and 'svm' for support vector machine
use_tfidf = False										# whether to use tf*idf term weighting (depending on the data, one or the other may work better)
frequency_cutoff = 3									# minimum frequency for word (and bigrams) to take into account
no_features = 10										# number of most important features to show



# ---------- only modify what follows if you are confident

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


# read data file
df = pd.read_csv(filename,delimiter=',',encoding='utf-8')

# some info about the dataset, in particular column names
print(df.info(),'\n\n---')

# data selection
X = df[col_text].astype('U')			# text column
Y = df[col_label].astype(str)			# label column

# clean out the HTML for twitter source variable
Y.replace("<.+?>","", regex=True, inplace=True)

# transform categories into numbers
le = LabelEncoder()				
Y = le.fit_transform(Y)

# cutting the dataset into training (85%) and testing (15%) data 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)

# see: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
if use_tfidf == False:
	count_vect = CountVectorizer(ngram_range=(1,2),min_df=frequency_cutoff,stop_words='english')		# 1- and 2-gram vectorizer
else:
	count_vect = TfidfVectorizer(ngram_range=(1, 2),min_df=frequency_cutoff,stop_words='english')		# 1- and 2-gram vectorizer with tf-idf transformation (depending on the data, this may work better or not)

# vectorize and weigh training data 
X_train_counts = count_vect.fit_transform(X_train)

# vectorize and weigh test data
X_test_counts = count_vect.transform(X_test)

# train the classifier
if type_classifier == 'bayes':
	clf = MultinomialNB()
else:
	clf = SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,random_state=42,max_iter=50,tol=1e-3)

# train the model
clf.fit(X_train_counts,Y_train)

# apply model to the test data
predicted = clf.predict(X_test_counts)

# create output to get an idea
counter = 0
for doc, category in zip(X_test, predicted):
	print('\n%r => %s' % (doc, le.classes_[category]))
	counter += 1
	if(counter > 5):
		break;

# calculate and print accuracy score
print('\n---\n\naccuracy score: %r\n\n---' % accuracy_score(Y_test,predicted))

# show most informative features (may fail if there are sparse labels)
feature_names = count_vect.get_feature_names()

print('\nmost informative features (high to low):\n')

# two paths needed since binary and multiclass result structures are quite different
if len(le.classes_) == 2:
	out = "\t"
	for label in le.classes_:
		out += '%-28s' % label
	print(out)
	coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
	top = zip(coefs_with_fns[:no_features], coefs_with_fns[:-(no_features + 1):-1])
	for (coef_1, fn_1), (coef_2, fn_2) in top:
		print('\t%.4f\t%-15s\t\t%.4f\t%-15s' % (coef_1, fn_1, coef_2, fn_2))
else:
	longest = len(max(le.classes_, key=len))
	for i, class_label in enumerate(le.classes_):
		top = np.argsort(clf.coef_[i])[-no_features:]
		print('{0: <{1}}'.format(class_label, longest)," ".join(feature_names[j] for j in top[::-1]))