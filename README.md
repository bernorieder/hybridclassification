# hybridclassification

This is a collection of Python scripts that facilitate experimentation and application of machine learning to text classification. The basic logic is that (possibly manually) classified units of text are used to train a classifier that can then be applied to another set of units to classify them. Class labels are taken from the training material.

There are two sets of scripts:
* The files in the 'bayes' folder use simple and rather fast techniques to derive a classifier: multinominal bayes or support vector machine, both using the implementations from scikit-learn.
* The files in the 'keras' folder use a combination of deep learning techniques using the Keras framework, which wraps around lower level ML frameworks such as TensorFlow.

To use these scripts, you need two files:
1. a CSV to use for training that has a column with text units and a column with class labels;
2. a CSV to classify that has a column with text units; 

## requirements

These scripts require a working installation of Python 3 and a number of libraries that can be easily installed via pip:
```
$ pip install pandas
$ pip install numpy
$ pip install scikit-learn
```

To use the deep learning scripts you also need to install:
```
$ pip install tensorflow
$ pip install keras
$ pip install keras-text
```

## use
For the 'bayes' path:
1. to test your labeling and the classifier, modify the variables in the header of 'sklearn\_test.py' and run the script;
2.  check the accuracy score and the most important features to get an idea of the quality of the classification;
3. to apply the classifier to new data, modify the variables in the header of 'sklearn\_apply.py' and run the script;
4. a new file is created with a new column called 'inferred\_label';