#
# adaboost_test.py
# 
# date last modified: 12 nov 2017
# modified last by: jerry
# 
#

import math
import operator
import numpy as np
import matplotlib.pyplot as plt
import re
from random import randint
from sklearn.neighbors import KNeighborsClassifier
#from matplotlib.colors import ListedColormap
from perceptron import PerceptronClassifier
from sklearn.linear_model import perceptron
from adaboost import AdaBoost

# once again, change this to switch datasets 
FILENAME = "dataset/default.csv" 
# probability of an example being in the training set 
PROBABILITY_TRAINING_SET = 0.7

# learning rate for perceptron 
ETA = 0.1
# desired threshold for error rate; 0.2 --> 20% 
THRESHOLD = 0.2
# maximum number of epochs for training
UPPER_BOUND = 100 
# verbose flag 
IS_VERBOSE = True 
# number of classifiers to induce in Adaboost
NUM_OF_CLASSIFIERS = 10

def split_dataset(examples, prob_training):
	"""
	receives list of examples  
	returns a tuple consisting of a training set and testing set 
	"""
	training_set = []
	testing_set = []

	# generate a random number [1,100]. if it is greater than 
	# prob_training then add the example to the 
	# testing set; otherwise add it to the training set 
	percent_training = prob_training * 100; 
	for example in examples:
		result = randint(1, 100) 
		# if the result is a number less than percent_training, 
		# add to training set; else add it to the testing set 
		if (result < percent_training):
			training_set.append(example)
		else:
			testing_set.append(example)

	return (training_set, testing_set)

def load_dataset(filename):
	"""
	given a filename that points to a file containing the data-set, 
	load it into memory and return an array containing this data-set
	"""
	dataset = []
	# open the data-set file
	file = open(filename, "r")
	# we want to load this data-set into a 2D array 
	# where each row is an example and each column is 
	# an attribute. 
	for line in file: 
		example = line.strip().split(",") # a row in the data-set 
		dataset.append(list(map(float, example[1:]))) # append it to the 2D array

	return dataset 

def split_attribute_and_label(dataset):
	"""
	split attribute vectors from their class-labels 
	"""

	# add 0.1 because values are processed as floats and we may have 0.999...
	class_labels = [int(row[-1] + 0.1) for row in dataset]
	attributes = [row[0:-1] for row in dataset]

	return (attributes, class_labels)

def calculate_error(class_labels, hypothesis_list):
	"""
	calculates simple error rate on a dataset
	:param class_labels: list of given class-labels 
	:param hypothesis_list: list of classifier predictions for examples
	"""
	num_errors = 0
	for i in range(len(class_labels)):
		if class_labels[i] != hypothesis_list[i]:
			num_errors += 1

	return (num_errors / len(class_labels))

# preprocessing: load in the dataset and split into a training and testing set 
dataset = load_dataset(FILENAME) 
(training_set,testing_set) = split_dataset(dataset, PROBABILITY_TRAINING_SET)

if IS_VERBOSE:
	print("training set size: %s testing set size: %s num instances: %s" % 
		(len(training_set), len(testing_set), len(dataset)))

# because datasets sometimes place the class attribute at the end or even 
# at the beginning or the middle, we'll separate the attribute vector from
# the class-label. also note that this is the way scikit-learn does it. 
# train_x: the attribute vector; train_y: the class_label  
(train_x, train_y) = split_attribute_and_label(training_set)
(test_x, test_y) = split_attribute_and_label(testing_set)

# create the perceptron classifier 
#linear_classifier = PerceptronClassifier(ETA, THRESHOLD, UPPER_BOUND, IS_VERBOSE)
# train the classifier 
#linear_classifier.fit(train_x, train_y)
#print("Training error rate %s" % linear_classifier.training_error_rate)
#print(linear_classifier.weights)

# test the trained classifier on the testing set 
#result_list = linear_classifier.predict(test_x)
#print("Testing error rate %s" % calculate_error(test_y, result_list))

# Create the perceptron object (net)
#net = perceptron.Perceptron(max_iter=UPPER_BOUND, verbose=0, random_state=None, fit_intercept=True, eta0=ETA)

# Train the perceptron object (net)
#net.fit(train_x,train_y)

#pred = net.predict(train_x)
#print("scikit-learn perceptron training error rate %s" % calculate_error(train_y, pred))

#pred_t = net.predict(test_x)
#print("scikit-learn perceptron testing error rate %s" % calculate_error(test_y, pred_t))

ada_obj = AdaBoost(NUM_OF_CLASSIFIERS, ETA, UPPER_BOUND, IS_VERBOSE)
ada_obj.fit(train_x, train_y)

hypothesis_list = ada_obj.predict(test_x)
mistakes = ada_obj.xor_tuples(test_y, hypothesis_list)
error_rate = ada_obj.classifier_error_rate(mistakes)

if IS_VERBOSE:
	print("Testing set error rate after training %f" % (error_rate))
