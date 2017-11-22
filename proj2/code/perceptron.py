#
# perceptron.py
# 
# date last modified: 22 nov 2017
# modified last by: jerry
# 
# implementation of the linear classifier, perceptron
#

from random import random
from optparse import OptionParser

class PerceptronClassifier:

	def __init__(self, eta, threshold, upper_bound, verbose = False,equal_weight_value = 0):
		"""
		constructor 
		"""
		# member variables for this class 
		self.learning_rate = eta
		self.threshold = threshold
		self.upper_bound = upper_bound
		self.verbose = verbose
		self.equal_weight_value = equal_weight_value

		self.weights = []
		self.training_error_rate = 0
		self.training_labels = []
		self.normal_train_data = []

	def __normalize(self, data_set):
		"""
		normalize the dataset with values [0,1]
		"""
		try:
			normal = []
			min_tuple = data_set[0][:]
			max_tuple = data_set[0][:]
			# find the min and max tuple
			for example in data_set:
				for i in range(len(example)):
					if example[i] < min_tuple[i]:
						# print("i ex min", i, example[i], min_tuple[i])
						min_tuple[i] = example[i]
					if example[i] > max_tuple[i]:
						max_tuple[i] = example[i]

			for example in data_set:
				tmp = []
				#print(example)
				for i in range(len(example)):
					#print(i)
					if max_tuple[i] != min_tuple[i]:
						tmp.append((example[i] - min_tuple[i])/ (max_tuple[i] - min_tuple[i]))
					else:
						tmp.append(max_tuple[i])
				#tmp.append(example[-1])
				normal.append(tmp)  # append the modified tuple to normalized array
		except ZeroDivisionError:
			print(i)
			print(max_tuple)
			print(min_tuple)

		return normal


	def __calculate_evidence(self, weights, example):
		"""
		calculates the evidence of sum(wi*xi), which is all the weights
		multiplied by the example attributes
		"""

		# sum up the wi*xi, we'll assume w0 is at end of the vector
		evidence = 0
		for i in range(len(weights) - 1):
			evidence += weights[i] * example[i]
		evidence += weights[-1]  # -1 is the w0 x0 = 1

		return evidence

	def __calculate_training_error(self):
		"""
		calculates the simple error rate on the training set 
		"""
		num_errors = 0
		for i in range(len(self.normal_train_data)):
			example = self.normal_train_data[i][:]
			hypothesis = 1 if self.__calculate_evidence(self.weights, example) > 0 else 0
			if self.training_labels[i] != hypothesis:
				num_errors += 1.0

		return (num_errors / len(self.training_labels))


	def fit(self, attribute_vectors, training_labels):
		"""
		'fits' the classifier given a training set and a set of class labels 
		for that training set 
		:param attribute_vectors: list of attribute vectors 
		:param training_labels: list of class labels associated with examples
		"""

		# first normalize the attribute vectors 
		self.normal_train_data = self.__normalize(attribute_vectors)
		self.training_labels = training_labels[:]

		# keep track of the best linear equation, in case threshold is 
		# never reached 
		best_weights = []
		best_error_rate  = 1

		#
		# adapted perceptron-learning algorithm from the textbook... 
		#
		# TABLE 4.1 (pg. 71)
		# 1. Initialize all weights, wi, to small random numbers. 
		#    Choose an appropriate learning rate, eta, between 0 and 1. 
		# 2. For each training example, whose class is c(x) 
		#    (i) Let h(x) = 1 if evidence > 0 and h(x) = 0 otherwise 
		#    (ii) Update each weight using the formula: 
		#         wi = wi + eta * [c(x) - h(x)] * xi 
		# 3. If c(x) = h(x) for all training examples, stop; otherwise 
		#    return to 2. 
		# 

		# equation has form: w0 + w1 * x1 + ... + wn * xn = 0 
		# STEP 1: initialize the weights for all wi, to small random numbers
		for i in range(len(self.normal_train_data[0]) + 1):
			if self.equal_weight_value == 0:
				self.weights.append(random()/3)  # i'th weight
			else:
				self.weights.append(self.equal_weight_value)

		# now put the w0 weight at the end of the equation (list) 
		# usually its the first weight but to make our implementation
		# easier, we put it at the end
		#self.weights.append(random()/3)


		epoch = 1
		error_rate = 100
		# STEP 2: for each training example... 
		while True:
			# calculate the evidence
			for k in range(len(self.normal_train_data)):
				example = self.normal_train_data[k][:]
				# (i) determine the hypothesis, h
				hypothesis = 1 if self.__calculate_evidence(self.weights, example) > 0 else 0
				# (ii) update each weight according to the formula
				#  delta_weight = eta * [c(x) - h(x)]
				delta_weight = self.learning_rate * (self.training_labels[k] - hypothesis)
				for i in range(len(self.weights) - 1):
					# wi = wi + delta_weight * xi 
					self.weights[i] += delta_weight * example[i]
				# remember to modify the bias, w0 
				self.weights[-1] += delta_weight

			# STEP 3: does c(x) = h(x) for all training examples? or does it 
			# meet a threshold? 
			error_rate = self.__calculate_training_error()
			if epoch % 5 == 0 and self.verbose:
				print("--> err4epoch: {:f} e: {:d}".format(error_rate, epoch))

			# meets the threshold error rate; training ends 
			if error_rate < self.threshold:
				self.training_error_rate = error_rate
				break
			# otherwise if this equation is the best seen, save it 
			elif error_rate < best_error_rate:
				best_error_rate = error_rate
				best_weights = self.weights[:]
			
			# c(x) != h(x) for all examples and did not meet the threshold
			# return to STEP 2 if upper-bound limit has not exceeded.  
			epoch += 1

			# now check if we have crossed the upper bound
			if epoch >= self.upper_bound :
				self.weights = best_weights[:]
				self.training_error_rate = best_error_rate
				break

	def predict(self, testing_set):
		"""
		given a trained linear classifier, predict the classes 
		of the testing set 

		:return: list of hypotheses for each example in the testing set
		"""
		result_list = []
		for k in range(len(testing_set)):
			example = testing_set[k][:]
			# determine the hypothesis, h
			hypothesis = 1 if self.__calculate_evidence(self.weights, example) > 0 else 0
			result_list.append(int(hypothesis))
		return result_list
 

