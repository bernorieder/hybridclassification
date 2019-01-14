
# built from:
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# ----------

filename = '../tcat_trump_full.csv'						# file to use for training and testing
col_text = 'text'										# name of the text column
col_label = 'source'									# name of the label column

# ----------

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
print(df.info())

# data selection
X = df[col_text]				# text column
Y = df[col_label]				# label column

# clean out the HTML for twitter source variable
Y.replace("<.+?>","", regex=True, inplace=True)

# transform categories into numbers
le = LabelEncoder()				
Y = le.fit_transform(Y)

# cutting the dataset into training (85%) and testing (15%) data 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)

# see: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# objects for the vectorizer and the frequency transformer (select one):
# count_vect = CountVectorizer(stop_words='english')									# 1-gram vectorizer
count_vect = CountVectorizer(ngram_range=(1, 2),min_df=2,stop_words='english')			# 1- and 2-gram vectorizer
# count_vect = TfidfVectorizer(ngram_range=(1, 2),min_df=2,stop_words='english')		# 1- and 2-gram vectorizer with tf-idf transformation (depending on the data, this may work better or not)


# vectorize and weigh training data 
X_train_counts = count_vect.fit_transform(X_train)

# vectorize and weigh test data
X_test_counts = count_vect.transform(X_test)

# train the classifier (select either multinominal naive Bayes or Support Vector Machine)
#clf = MultinomialNB()
clf = SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,random_state=42,max_iter=20,tol=None)

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
print('\naccuracy score: %r' % accuracy_score(Y_test,predicted))

# show most informative features (may fail if there are sparse labels)
n=10
feature_names = count_vect.get_feature_names()

print('\nmost informative features (high to low):\n')

# two paths needed since binary and multiclass result structures are quite different
if len(le.classes_) == 2:
	out = "\t"
	for label in le.classes_:
		out += '%-28s' % label
	print(out)
	coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
	top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
	for (coef_1, fn_1), (coef_2, fn_2) in top:
		print('\t%.4f\t%-15s\t\t%.4f\t%-15s' % (coef_1, fn_1, coef_2, fn_2))
else:
	longest = len(max(le.classes_, key=len))
	for i, class_label in enumerate(le.classes_):
		top = np.argsort(clf.coef_[i])[-n:]
		print('{0: <{1}}'.format(class_label, longest)," ".join(feature_names[j] for j in top[::-1]))