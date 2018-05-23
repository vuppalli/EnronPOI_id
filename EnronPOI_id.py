#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Add features that are being used
features_list = ['poi','salary', 'bonus', 'long_term_incentive']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Cross Validation
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.4, random_state=50)

### Classifier
from sklearn import tree
#clf = tree.DecisionTreeClassifier()
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

### evaluating classifier
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print "accuracy: ", acc

from sklearn.metrics import precision_score
pscore = precision_score(labels_test, pred)
print "precision: ", pscore

from sklearn.metrics import recall_score
rscore = recall_score(labels_test, pred)
print "recall: ", rscore

### Dump your classifier, dataset, and features_list so anyone can check your results

dump_classifier_and_data(clf, my_dataset, features_list)

