#
# adaboost.py
# 
# date last modified: 16 nov 2017
# modified last by: jerry
# 
#

from perceptron import PerceptronClassifier
from sklearn.linear_model import perceptron
from random import random

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
		# we get an extremely small probability ???? 
		prob_list = [1.0/len(self.train_x)]*len(self.train_x)
		for i in range(self.num_of_classifiers):
			# this is the ith classifier
			(T_i_x, T_i_y) = self.__training_set_Ti_with_probablity(prob_list)
			print("ti length : " + str(len(T_i_x)))
			linear_classifier = PerceptronClassifier(self.eta, self.threshold, self.upper_bound, self.verbose)
			# train the classifier 
			linear_classifier.fit(T_i_x, T_i_y)
			# test the trained classifier on the testing set 
			result_list = linear_classifier.predict(self.train_x)
			self.classifiers_list.append(linear_classifier)

			# this step will update the probability for the next i+1 classifiner
			mistakes = self.xor_tuples(self.train_y,result_list)
			# update probablity
			prob_list = self.__update_probabity(prob_list, mistakes)


	def __training_set_Ti_with_probablity(self, prob_list):
		T_i_x = []
		T_i_y = []
		for i in range(len(self.train_x)):
			if prob_list[i] >= random():
				T_i_x.append(self.train_x[i][:])
				T_i_y.append(self.train_y[i])
		return (T_i_x,T_i_y)


	def __update_probabity(self, prob_list, mistakes):
		epsilon_i = 0
		for j in range(len(self.train_x)):
			epsilon_i += wrong_samples[j] * prob_list[j]
		beta_i = epsilon_i/(1 - epsilon_i)
		for j in range(len(self.train_x)):
			# 0 means correctly classified
			if wrong_samples[j] == 0:
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
