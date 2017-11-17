#
# adaboost.py
# 
# date last modified: 16 nov 2017
# modified last by: guru
# 
#

import numpy as np
from perceptron import PerceptronClassifier
from sklearn.linear_model import perceptron
from random import random

# probability that an example is our training subset
PROBABILITY_TRAINING_SUBSET = 0.6

class AdaBoost:
	def __init__(self,num_of_classifiers, eta, upper_bound, verbose, use_scikit_learn = False):
		self.num_of_classifiers = num_of_classifiers
		self.eta = eta
		self.upper_bound = upper_bound
		self.verbose = verbose
		self.scikit_learn = use_scikit_learn
		self.threshold = 0.2

	def fit(self, train_x, train_y):
		self.train_x = train_x
		self.train_y = train_y
		self.classifiers_list = []
		if self.scikit_learn == False:
			self.__our_perceptron()
		#else:
		#	self.__scikit_perceptron()

	def __our_perceptron(self):
		# every ex has this initial probability
		prob_list = [1.0/len(self.train_x)]*len(self.train_x)

		for i in range(self.num_of_classifiers):
			# this is the ith classifier
			(T_i_x, T_i_y) = self.__training_set_Ti_with_probablity(prob_list, PROBABILITY_TRAINING_SUBSET)
			linear_classifier = PerceptronClassifier(self.eta, self.threshold, self.upper_bound, self.verbose)
			# train the classifier 
			linear_classifier.fit(T_i_x, T_i_y)

			# test the trained classifier on the testing set 
			result_list = linear_classifier.predict(self.train_x)
			self.classifiers_list.append(linear_classifier)

			# this step will give the error vector for the training set,
			# where 1 = incorrect classification and 0 = correct classification
			mistakes = self.xor_tuples(self.train_y,result_list)

			# get error rate for current classifier
			classifier_error = self.__classifier_error_rate(mistakes)
			print("Classifier #%d error rate = %f" % (i+1, classifier_error))

			# update probablity for the next i+1 classifier
			prob_list = self.__update_probabity(prob_list, mistakes)

	def __training_set_Ti_with_probablity(self, prob_list, prob_in_subset):
		T_i_x = []
		T_i_y = []

		n_training = len(prob_list)
		n_subset = int(n_training * prob_in_subset)
		
		# Generates a list of indexes (from 0,...,n_training) 
		# of size n_subset using the prob_list
		index_list = np.random.choice(n_training, n_subset, replace=True, p=prob_list)

		for i in index_list:
			T_i_x.append(self.train_x[i][:])
			T_i_y.append(self.train_y[i])
	
		return (T_i_x,T_i_y)

	def __update_probabity(self, prob_list, mistakes):
		epsilon_i = 0
		for j in range(len(self.train_x)):
			epsilon_i += mistakes[j] * prob_list[j]
		
		beta_i = epsilon_i/(1 - epsilon_i)
		
		for j in range(len(self.train_x)):
			prob_list[j] = beta_i * prob_list[j]

		normal_total = sum(prob_list)
		for j in range(len(self.train_x)):
			prob_list[j]/= normal_total 
		return prob_list

	def xor_tuples(self, class_labels, hypothesis_list):
		abzip = zip(class_labels, hypothesis_list)

		xor_val = []
		for v in abzip:
			if v[0] == v[1]:
				xor_val.append(0)
			else:
				xor_val.append(1)
		return xor_val

	def __classifier_error_rate(self, mistakes):
		error_count = 0
		for val in mistakes:
			error_count = error_count + 1 if val == 1 else error_count

		return error_count / len(mistakes)
