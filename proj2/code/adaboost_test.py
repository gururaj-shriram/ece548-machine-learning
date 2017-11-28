#
# adaboost_test.py
# 
# date last modified: 28 nov 2017
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
from sklearn import tree 
from perceptron import PerceptronClassifier
from sklearn.linear_model import perceptron
from sklearn.ensemble import AdaBoostClassifier
from adaboost import AdaBoost
from adaboost import classifier_error_rate

# once again, change this to switch datasets; 
# don't forget to toggle load_dataset() as well 

#FILENAME = "dataset/default.csv" 
#FILENAME = "dataset/ionosphere.dat" 
FILENAME = "dataset/musk.dat"  
#FILENAME = "dataset/heart.dat"  
#FILENAME = "dataset/spambase.dat" 
#FILENAME = "dataset/animals.dat" 
#FILENAME = "dataset/ecoli.dat"
#FILENAME = "dataset/fertility.dat"

# probability of an example being in the training set 
PROBABILITY_TRAINING_SET = 0.65

# learning rate for perceptron 
ETA = 0.05
# learning rate for perceptron when adjusting weights of classifiers 
ETA_WEIGHTS = 0.0001
# desired threshold for error rate; 0.2 --> 20% 
THRESHOLD = 0.05
# maximum number of epochs for training
UPPER_BOUND = 500
# verbose flag 
IS_VERBOSE = True 
# number of classifiers to induce in Adaboost
NUM_OF_CLASSIFIERS = 20

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
		dataset.append(list(map(float, example[:]))) # append it to the 2D array

	return dataset 


def load_dataset_ionosphere(filename):
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
		if example[-1] == 'g':
			example[-1] = 1
		else:
			example[-1] = 0
		dataset.append(list(map(float, example[:]))) # append it to the 2D array

	return dataset 

def load_dataset_musk(filename):
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
		dataset.append(list(map(float, example[2:]))) # append it to the 2D array

	return dataset 

def load_dataset_ecoli(filename):
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
		example = line.strip().split("  ") # a row in the data-set 
		dataset.append(list(map(float, example[1:]))) # append it to the 2D array

	return dataset 

def load_dataset_heart(filename):
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
		example = line.strip().split(" ") # a row in the data-set 
		if example[-1] == '2':
			example[-1] = 1
		else:
			example[-1] = 0
		dataset.append(list(map(float, example[:]))) # append it to the 2D array

	return dataset 

def split_attribute_and_label(dataset):
	"""
	split attribute vectors from their class-labels 
	"""

	# add 0.1 because values are processed as floats and we may have 0.999...
	class_labels = [round(row[-1]) for row in dataset]
	attributes = [row[:-1] for row in dataset]
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

class ErrorWrapper:
	def __init__(self, num_classifiers, train_error, test_error, scikit_error):
		self.num_classifiers = num_classifiers
		self.train_error = train_error
		self.test_error = test_error
		self.scikit_error = scikit_error

	def __str__(self):
		return "# of Classifiers {0}, Train Error: {1}, Test Error: {2}, Scikit Error: {3}".format(
			self.num_classifiers, self.train_error, self.test_error, self.scikit_error)

def perceptron_avg_run(avg_num_of_run, training_set, testing_set):
	(train_x, train_y) = split_attribute_and_label(training_set)
	(test_x, test_y) = split_attribute_and_label(testing_set)

	perceptron_error = []
	
	for i in range(avg_num_of_run):

		p = perceptron.Perceptron(max_iter=UPPER_BOUND, verbose=0, random_state=None, 
								fit_intercept=True, eta0=ETA)
		p.fit(train_x, train_y)
		result_list = p.predict(test_x)
		perceptron_error.append(calculate_error(test_y, result_list))

	return sum(perceptron_error) / len(perceptron_error)


def decision_tree_avg_run(avg_num_of_run, training_set, testing_set):
	(train_x, train_y) = split_attribute_and_label(training_set)
	(test_x, test_y) = split_attribute_and_label(testing_set)

	# run decision tree classifier avg_num_of_run times
	decision_tree_error = []
	for i in range(avg_num_of_run):
		clf = tree.DecisionTreeClassifier()
		clf = clf.fit(train_x, train_y)
		decision_tree_result_list = clf.predict(test_x)
		decision_tree_error.append(calculate_error(test_y, decision_tree_result_list))

	return sum(decision_tree_error) / len(decision_tree_error)

def adaboost_avg_run(max_classes, avg_num_of_run, training_set, testing_set):
	testing_error_list = []
	all_error_list = []

	# because datasets sometimes place the class attribute at the end or even 
	# at the beginning or the middle, we'll separate the attribute vector from
	# the class-label. also note that this is the way scikit-learn does it. 
	# train_x: the attribute vector; train_y: the class_label  
	(train_x, train_y) = split_attribute_and_label(training_set)
	(test_x, test_y) = split_attribute_and_label(testing_set)
	# print(len(train_x))	
	train_subset_num = int(len(train_y) * 0.2) 

	for cl in range(1, max_classes+1, 3):
		train_error = []
		testing_error = []
		scikit_error = []
		for i in range(avg_num_of_run):
			
			ada_obj = AdaBoost(cl, train_subset_num, THRESHOLD, ETA, UPPER_BOUND, ETA_WEIGHTS, False)
			ada_obj.fit(train_x, train_y)

			hypothesis_list = ada_obj.predict(train_x)
			mistakes = ada_obj.xor_tuples(train_y, hypothesis_list)
			error_rate_train = classifier_error_rate(mistakes)

			hypothesis_list = ada_obj.predict(test_x)
			mistakes = ada_obj.xor_tuples(test_y, hypothesis_list)
			error_rate_test = classifier_error_rate(mistakes)
			train_error.append(error_rate_train)
			testing_error.append(error_rate_test)

			pada = perceptron.Perceptron(max_iter=UPPER_BOUND, verbose=0, random_state=None, 
							fit_intercept=True, eta0=ETA)

			bdt = AdaBoostClassifier(p,algorithm="SAMME",n_estimators=cl)
			bdt.fit(train_x, train_y)
			result_list = bdt.predict(test_x)
			scikit_error.append(calculate_error(test_y, result_list))

		errors = ErrorWrapper(
			cl, 
			sum(train_error)/len(train_error), 
			sum(testing_error)/len(testing_error), 
			sum(scikit_error)/len(scikit_error))

		all_error_list.append(errors)
		print("Train avg for %s   %s"%(cl, errors.train_error))
		print("Testing avg for %s   %s"%(cl, errors.test_error))
		testing_error_list.append((sum(testing_error)/len(testing_error)) * 100)
		print("Scikit adaboost avg for %s   %s"%(cl, errors.scikit_error))

	#return testing_error_list
	return all_error_list

def plot_errors(error_list):
	num_classifiers_list = []
	train_error_list = []
	test_error_list = []
	scikit_error_list = []

	tmp = FILENAME.split('/')
	title = tmp[1].split('.')[0].title()
	
	for error in error_list:
		num_classifiers_list.append(error.num_classifiers)
		train_error_list.append(error.train_error)
		test_error_list.append(error.test_error)
		scikit_error_list.append(error.scikit_error)

	plt.plot(num_classifiers_list, train_error_list, 'r-')
	plt.plot(num_classifiers_list, test_error_list, 'g-')
	plt.plot(num_classifiers_list, scikit_error_list, 'b-')
	plt.legend(['Training Set Error', 'Testing Set Error', 'Scikit Error'], loc = 'upper left')
	plt.xlabel('Number of Classifiers')
	plt.ylabel('Error Rate')
	plt.title('Adaboost Error Rates on the {0} Dataset'.format(title))
	plt.savefig('{0}.png'.format(title))
	plt.gcf().clear()

def plot_testing_set_errors(error_list, decision_tree_avg_error, perceptron_avg_error):
	num_classifiers_list = []
	test_error_list = []

	tmp = FILENAME.split('/')
	title = tmp[1].split('.')[0].title()
	
	for error in error_list:
		num_classifiers_list.append(error.num_classifiers)
		test_error_list.append(error.test_error)

	plt.plot(num_classifiers_list, test_error_list, 'r-')
	plt.axhline(y=decision_tree_avg_error, color='g')
	plt.axhline(y=perceptron_avg_error, color='b')

	plt.legend(['Testing Set Error', 'Average Error using a Decision Tree', 'Average Error using a Single Perceptron Classifier'], loc = 'upper right')
	plt.xlabel('Number of Classifiers')
	plt.ylabel('Error Rate')
	plt.title('Error Rates with Different Classifiers on the {0} Dataset'.format(title))
	plt.savefig('{0}_different_classifiers.png'.format(title))
	plt.gcf().clear()

# preprocessing: load in the dataset and split into a training and testing set 
#dataset = load_dataset(FILENAME) 
#dataset =load_dataset_ionosphere(FILENAME)
dataset =load_dataset_musk(FILENAME)
#dataset =load_dataset_heart(FILENAME)
#dataset =load_dataset_ecoli(FILENAME)

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


# # create the perceptron classifier 
#linear_classifier = PerceptronClassifier(ETA, THRESHOLD, UPPER_BOUND, False)
# # train the classifier 
#linear_classifier.fit(train_x, train_y)
#print("Training error rate %s" % linear_classifier.training_error_rate)
# #print(linear_classifier.weights)

# # test the trained classifier on the testing set 
#result_list = linear_classifier.predict(test_x)
# print("=========")
#print("our perceptron error rate on test: %s" % calculate_error(test_y, result_list))
# print("=========")

# test their perceptron and adaboost for comparison 
p = perceptron.Perceptron(max_iter=UPPER_BOUND, verbose=0, random_state=None, 
							fit_intercept=True, eta0=ETA)
p.fit(train_x, train_y)
result_list = p.predict(test_x)
print("their perceptron error on test: %s" % calculate_error(test_y, result_list))

pada = perceptron.Perceptron(max_iter=UPPER_BOUND, verbose=0, random_state=None, 
							fit_intercept=True, eta0=ETA)


bdt = AdaBoostClassifier(p,algorithm="SAMME",n_estimators=NUM_OF_CLASSIFIERS)
bdt.fit(train_x, train_y)
result_list = bdt.predict(test_x)
print("their adaboost error on test: %s" % calculate_error(test_y, result_list))

# test a decision tree 
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x, train_y)
decision_tree_result_list = clf.predict(test_x)
print("=========")
print("single tree error rate on testing set: %s" % calculate_error(test_y, decision_tree_result_list))
print("=========")

# # Create the perceptron object (net)
# #net = perceptron.Perceptron(max_iter=UPPER_BOUND, verbose=0, random_state=None, fit_intercept=True, eta0=ETA)

# # Train the perceptron object (net)
# #net.fit(train_x,train_y)

# #pred = net.predict(train_x)
# #print("scikit-learn perceptron training error rate %s" % calculate_error(train_y, pred))

# #pred_t = net.predict(test_x)
# #print("scikit-learn perceptron testing error rate %s" % calculate_error(test_y, pred_t))

# # need to find good number for training subset size
# train_subset_num = int(len(train_y)*.5) #int(len(train_y)*10/NUM_OF_CLASSIFIERS)
# print("num examples in training subset : " + str(train_subset_num))

# ada_obj = AdaBoost(NUM_OF_CLASSIFIERS, train_subset_num, THRESHOLD, ETA, UPPER_BOUND, ETA_WEIGHTS, IS_VERBOSE)
# ada_obj.fit(train_x, train_y)
# print(ada_obj.classifiers_weights)

# hypothesis_list = ada_obj.predict(train_x)
# mistakes = ada_obj.xor_tuples(train_y, hypothesis_list)
# error_rate = ada_obj.classifier_error_rate(mistakes)

# print('training error rate %f'%error_rate)

# hypothesis_list = ada_obj.predict(test_x)
# mistakes = ada_obj.xor_tuples(test_y, hypothesis_list)
# error_rate = ada_obj.classifier_error_rate(mistakes)

# print('testing error rate %f'%error_rate)

# #(average_train, average_test) = average_for_runs(10, train_subset_num, NUM_OF_CLASSIFIERS)

# #print('average train error rate %f'%average_train)
# #print('average test error rate %f'%average_test)

# #average_error_rate = average_for_runs(10, train_subset_num, 10)
# #print("average error rate : %f" % average_error_rate) 

# split dataset only once
(training_set,testing_set) = split_dataset(dataset, PROBABILITY_TRAINING_SET)
# error_list = adaboost_avg_run(40, 5, training_set, testing_set)
error_list = adaboost_avg_run(25, 5, training_set, testing_set)
decision_tree_avg_error = decision_tree_avg_run(5, training_set, testing_set)
perceptron_avg_error = perceptron_avg_run(5, training_set, testing_set)
plot_errors(error_list)
plot_testing_set_errors(error_list, decision_tree_avg_error, perceptron_avg_error)
