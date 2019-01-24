# hybridclassification

This is a collection of Python scripts that facilitate experimentation and application of machine learning to text classification. The basic logic is that (possibly manually) classified units of text are used to train a classifier that can then be applied to another set of units to classify them. Classes are taken from the training material.

There are two sets of scripts:
* The files in the "bayes" folder use simple and rather fast techniques to derive a classifier: multinominal bayes or support vector machine, both using the implementations from scikit-learn.
* The files in the "keras" folder use a combination of deep learning techniques using the Keras framework, which wraps around lower level ML frameworks such as TensorFlow.

## requirements

The scripts require a Python 3 installation and a number of libraries that can be easily installed via pip:
' $ pip install pandas
	$ pip install numpy
	$ pip install scikit-learn

To use the deep learning scripts you also need to install:
	$ pip install tensorflow
	$ pip install keras
	$ pip install keras-text





Two sets of examples for how to train ML classifiers on the basis of labeled CSV files.