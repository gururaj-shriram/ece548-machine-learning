#
# adaboost.py
#
# date last modified: 26 nov 2017
# modified last by: jerry
#
#

import numpy as np
from collections import Counter
from perceptron import PerceptronClassifier
from sklearn.linear_model import perceptron
from random import random, randint

# toggle verbose mode for perceptron
PERCEPTRON_IS_VERBOSE = False 

class AdaBoost:

	def __init__(self, num_classifiers, train_num, threshold, eta, upper_bound, eta_weights, 
		verbose, use_scikit_learn=False):
		"""
		num_of_classifiers: number of base-learners 
		train_num: number of examples to be in each training subset
		threshold: error rate threshold for perceptron
		eta: learning rate for perceptron learners 
		upper_bound: maximum number of iterations before terminating; for perceptron 
		eta_weights: learning rate for perceptron when modifying weight of classifiers 
		verbose: verbose mode 
		scikit_learn: whether or not to use scikit's perceptron (not implemented yet) 
		"""
		self.num_of_classifiers = num_classifiers
		self.train_num = train_num
		self.threshold = threshold #0.2
		self.eta = eta
		self.upper_bound = upper_bound
		self.weight_learning_rate = eta_weights #0.001
		self.verbose = verbose
		self.scikit_learn = use_scikit_learn

	def fit(self, train_x, train_y):
		self.train_x = train_x[:]
		self.train_y = train_y
		self.classifiers_list = []
		# give all classifiers an initial weight of 1 
		self.classifiers_weights = [1] * self.num_of_classifiers
		if self.scikit_learn == False:
			self.__our_perceptron()
			self.__calculate_adaboost_weights()

	def predict(self, testing_set):
		hypothesis_list = []
		(result_y, result_list) = self.__get_evidence_for_assembly(testing_set)
		#for i in range(len(testing_set)):
			#(result_y, result_list) = self.__get_evidence_for_assembly(testing_set[i])
		#hypothesis_list.append(result_y)
		#print(self.classifiers_weights)
		#return hypothesis_list
		return result_list

	def __our_perceptron(self):

		# Table 9.3 Adaboost algorithm 
		# Let m be the number of examples in training_set and let i = 1  
		# 1. For each ex in training_set, set p1(ex) = 1 / m 
		# 2. Create subset Ti consisting of train_num examples randomly selected 
		#    according to the given probabilities. From this Ti, induce Ci. 
		#    note: the book says m examples?
		#    
		# 3. Evaluate Ci on each ex in training_set 
		#    Let e_i(ex) = 1 if Ci misclassified; else, 0 
		#    (i)  Calculate epsilon_i = SUM(j=1, m) pi(ex_j) * ei(ex_j) 
		#    (ii) Calculate beta_i = epsilon_i / (1 - epsilon_i) 
		# 4. Modify probabilities of correctly classified examples 
		#    by p_i+1(ex) = p_i(ex) * beta_i 
		# 5. Normalize the probabilities such that all SUM(j=1, m) p_i+1(ex_j) = 1
		# 6. termination: Do until we have reached num_classifiers 
		#    note: or, should we go until a desired error rate is met?     

		# STEP 1: every example has this initial probability p1 = 1 / m  
		prob_list = [1.0 / len(self.train_x)] * len(self.train_x)

		for i in range(self.num_of_classifiers):
			# STEP 2: create the i-th training subset 
			(T_i_x, T_i_y) = self.__training_set_Ti_with_probablity(prob_list)
			# induce the i-th classifier Ci from this training subset
			linear_classifier = PerceptronClassifier(
				self.eta, self.threshold, self.upper_bound, PERCEPTRON_IS_VERBOSE) 
			linear_classifier.fit(T_i_x, T_i_y)

			# STEP 3: evaluate Ci on each example in the training set 
			hypothesis_list = linear_classifier.predict(self.train_x)

			# Let e_i(ex) = 1 if Ci misclassified; else, 0 
			# this step will give the error vector for the training set,
			# where 1 = incorrect classification and 0 = correct classification
			mistakes = self.xor_tuples(self.train_y, hypothesis_list)

			# get error rate for Ci
			training_error = self.classifier_error_rate(mistakes)
			linear_classifier.training_error_rate = training_error
			# add this Ci to the list of classifiers 
			self.classifiers_list.append(linear_classifier)

			if self.verbose:
				print("C%d has training error rate : %f" %
					  (i + 1, training_error))

			# update probablity for the next i+1 classifier
			prob_list = self.__update_distribution(prob_list, mistakes)

	def __wheel_of_fortune(self, prob_list):
		train_subset = set()
		fortune_value = [prob_list[0]]
		for i in range(1, len(prob_list)):
			fortune_value.append(fortune_value[i-1]+prob_list[i])
		
		while len(train_subset) < self.train_num:
			cur_fortune = random()
			s = 0
			e = len(fortune_value)-1
			mid = 0
			while(s <= e):
				mid = int((s + e)/2)
				if fortune_value[mid] == cur_fortune:
					break 
				elif fortune_value[mid] > cur_fortune:
					e = mid - 1
				else:
					s = mid + 1
			train_subset.add(mid)	
		return train_subset


	def __training_set_Ti_with_probablity(self, prob_list):
		"""
		create the training subset Ti given a probability distribution 
		"""
		T_i_x = [] # attribute vector
		T_i_y = [] # class label (in parallel with attr vector) 

		n_training = len(self.train_x)

		# this np.random.choice() is an implementation of the 
		# wheel of fortune; see todo.txt

		# Generates a list of indexes (from 0,...,n_training)
		# of size train_num (user-constant) using the prob_list
		index_list = np.random.choice(
			n_training, self.train_num, replace=False, p=prob_list)

		#index_list = self.__wheel_of_fortune(prob_list)
		for i in index_list:
			T_i_x.append(self.train_x[i][:])
			T_i_y.append(self.train_y[i])

		return (T_i_x, T_i_y)

	def __get_evidence_for_assembly(self, example):
		"""
		performs *weighted majority voting* where we sum up all weights 
		in support of the positive and negative classes

		the class with the overwhelming amount of evidence is the class the 
		"assembly" decides to label the example with 
		"""
		hypothesis_list = [] # which class does Ci think this example belongs to?
		postitive_weight_sum = 0
		negative_weight_sum = 0

		#example_list = []
		#example_list.append(example)
		#print(example)
		# query the assembly about this example  
		for i in range(len(self.classifiers_list)):
			hypothesis = self.classifiers_list[i].predict(example)
			# hypothesis will be a list of size 1; that way, it is compatible
			# with the perceptron implementation
			for x in range(len(hypothesis)):
				if hypothesis[x] == 1:
					postitive_weight_sum += self.classifiers_weights[i]
				else:
					negative_weight_sum += self.classifiers_weights[i]
				hypothesis_list.append(hypothesis[x])

		if postitive_weight_sum > negative_weight_sum:
			return (1, hypothesis_list)
		else:
			return (0, hypothesis_list)

	#def __calculate_adaboost_weights_using_error(self):
	#	for k in range(len(self.classifiers_list)):
	#		self.classifiers_weights[k] = 1 - self.classifiers_list[k].training_error_rate

	def __calculate_adaboost_weights(self):
		"""
		final classification decision is reached by a weighted majority voting 
		where each classifier has been assigned a certain weight; each classifier
		has a different strength defined by its weight wi 

		we use perceptron learning to modify and update these weights 
		"""
		for i in range(len(self.train_x)):
			example = self.train_x[i][:]
			example_list = []
			example_list.append(example)
			(result_y, classifier_results) = self.__get_evidence_for_assembly(example_list)
			if(self.train_y[i] != result_y):
				# Each time the assembly misclassifies an example, increase or 
				# decrease the weights of the individual classifiers according to 
				# the relation between the assembly hypothesis and the 
				# training example's true class 
				for k in range(len(self.classifiers_weights)):
					# note: i am a little unsure about the weight-updating formula...
					#delta_weight = self.weight_learning_rate * abs(classifier_results[k] - result_y)
					delta_weight = self.weight_learning_rate * (self.train_y[i] - classifier_results[k])
					self.classifiers_weights[k] += delta_weight

	def __get_result_list(self, testing_set):
		voted_result_list = []
		all_results = []

		# predict values for each example in the training set with each
		# classifier
		for classifier in self.classifiers_list:
			all_results.append(classifier.predict(testing_set))

		# for each attribute value, choose the value with the highest frequency
		# of votes from the classifiers
		for i in range(len(all_results[0])):
			i_attribute_values = []
			for j in range(len(all_results)):
				i_attribute_values.append(all_results[j][i])

			c = Counter(i_attribute_values).most_common()

			max_results = []
			for value, count in c:
				if count == c[0][1]:
					max_results.append(value)
				else:
					break

			voted_result_list.append(
				max_results[randint(0, len(max_results) - 1)])

		return voted_result_list

	def __update_distribution(self, prob_list, mistakes):
		"""
		update the probability distribution prob_list given a list of mistakes 
		""" 

		# STEP 3 continued
		# (i) Calculate epsilon_i = SUM(j=1, m) pi(ex_j) * ei(ex_j) 
		epsilon_i = 0
		for j in range(len(self.train_x)):
			epsilon_i += mistakes[j] * prob_list[j]
		# (ii) Calculate beta_i = epsilon_i / (1 - epsilon_i) 
		beta_i = epsilon_i / (1 - epsilon_i)

		# STEP 4 modify probabilities of correctly classified examples 
		# it is possible that beta_i can be 0...
		if beta_i == 0:
			beta_i += 0.01
		# using p_i+1(ex) = p_i(ex) * beta_i 
		for j in range(len(self.train_x)):
			# do all probabilities get modified by beta_i?? 
			if mistakes[j] == 0: 
				prob_list[j] = prob_list[j] * beta_i 

		# STEP 5 normalize all probabilities
		normal_total = sum(prob_list)
		for j in range(len(self.train_x)):
			prob_list[j] /= normal_total

		return prob_list

	def xor_tuples(self, class_labels, hypothesis_list):
		xor_val = []
		for i in range(len(class_labels)):
			if class_labels[i] == hypothesis_list[i]:
				xor_val.append(0)
			else:
				xor_val.append(1)
		return xor_val

	def classifier_error_rate(self, mistakes):
		error_count = 0
		for val in mistakes:
			error_count = error_count + 1 if val == 1 else error_count

		return error_count / len(mistakes)
