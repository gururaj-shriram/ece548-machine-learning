#
# nearest-neighbor.py  
# 
# date last modified: 8 oct 2017
# modified last by: jerry
#

from random import randint
import math
import operator
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

FILENAME = "iris-modified.csv"
PROBABILITY_TRAINING_SET = 0.7
K = 3


def split_dataset(examples, prob_training):
	"""
	receives list of examples  
	returns a tuple consisting of a training set and testing set 
	"""
	training_set = []
	testing_set = []

	# generate a random number [1,100]. if it is greater than 0.7 
	# then add the example to the testing set; otherwise add 
	# it to the training set 
	percent_training = prob_training * 100; 
	for example in examples:
		result = randint(1, 100) 
		# if the result is a number less than 70, add to training set
		# else add it to the testing set 
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
	# an attribute. for iris, total size is # of lines * 5 
	for line in file: 
		example = line.strip().split(",") # a row in the data-set 
		dataset.append(example) # append it to the 2D array

	# need to remove newline character from row 
	#for x in range(len(dataset)):
	#	row = dataset[x][-2] 
	#	dataset[x][-2] = row[0:-1]

	return dataset 

def sort_distances(distances):
	"""
	given a list of distances that are a two-tuple of distance and
	class-label, sort by increasing distance such that the first
	element has the lowest distance
	"""
	sorted_distances = sorted(distances, key=lambda by_distance:by_distance[0])

	return sorted_distances

def calculate_distance(training_set, testing_record):
	"""
	given a training set and a testing "point" called testing_record,
	calculate the distance from the test "point" to all records in
	the training set 
	"""
	distances = [] 
	for example in training_set:
		# we want the attribute tuple, but not the class label 
		distance = 0
		for x in range(len(example) - 1):
			# SUM of sqrt((x2 - x1)^2)  
			distance += pow(float(example[x]) - float(testing_record[x]), 2) 
		# two-tuple: (distance, class-label) 
		distances.append((math.sqrt(distance), example[-1]))

	return distances

def get_nearest_neighbors(training_set, testing_record, k):
	"""
	given a training set and a testing "point" called testing_record,
	discover the nearest neighbor(s) 
	"""
	distances = []
	sorted_distances = []
	nearest_neighbors = []
	distances = calculate_distance(training_set, testing_record)
	# from Table 3.2 
	# 1) Among the training examples, identify the k nearest neighbors of x 
	sorted_distances = sort_distances(distances)
	#print(sorted_distances)
	for i in range(k):
		nearest_neighbors.append(sorted_distances[i])
	return nearest_neighbors


def get_accuracy(test_set, hypothesis_set):
	"""
	returns a performance measure for our k-NN 
	in particular an error rate and an accuracy rate (1 - error rate)
	"""
	Test_label = [row[-1] for row in test_set]
	#print(Test_label == hypothesis_list)
	zipped = list(zip(hypothesis_set, Test_label))
	total_miss = 0
	for i in range(len(test_set)):
		#print(zipped[i])
		if zipped[i][0] != zipped[i][1]:
			total_miss += 1
	error_rate = total_miss * 100.0/len(test_set)
	accuracy = 100 - error_rate
	return (accuracy,error_rate)

def get_accuracy_scikit(Test_label, result_label):
	"""
	returns a performance measure for scikit-learn k-NN 
	"""
	zipped = list(zip(result_label, Test_label))
	total_miss = 0
	for i in range(len(Test_label)):
		#print(zipped[i])
		if zipped[i][0] != zipped[i][1]:
			total_miss += 1
	error_rate = total_miss * 100.0/len(Test_label)
	accuracy = 100 - error_rate
	return (accuracy,error_rate)

def seperate_attribute_and_label(data):
	"""
	helper function to pre-process row-data for scikit-learn 
	attribute vectors and class-label must be separated 
	"""
	class_label = [row[-1] for row in data]
	attributes = [row[0:-1] for row in data]
	return (attributes,class_label)

def classify(nearest_neighbors):
	"""
	receives a list of nearest neighbors and performs a majority vote 
	"""
	# frequency map is a dictionary from class label 
	# to frequency, i.e., number of occurrences 
	frequency_map = {}
	for i in range(len(nearest_neighbors)):
		if (nearest_neighbors[i][-1] in frequency_map):
			# class-label has already been seen
			frequency_map[nearest_neighbors[i][-1]] += 1
		else:
			# class-label has not yet been seen 
			frequency_map[nearest_neighbors[i][-1]] = 1
	# from Table 3.2 
	# 2) Let ci be the class most frequently found among these k nearest neighbors.
	# 3) Label x with ci.
	# sort in reverse order so that the example with highest vote appears first 
	frequency_list = sorted(frequency_map.items(), key=operator.itemgetter(1), reverse=True)
	# this class is appended to the hypothesis list 
	return frequency_list[0][0]


def nearest_neighbors_implementation(training_set,testing_set):
	"""
	our nearest neighbor classifier 
	"""

	# list that stores the classifier's class label for each example 
	# for instance, element 0 stores the majority vote for the first example
	hypothesis_list = []
	for i in range(len(testing_set)):
		nearest_neighbors = get_nearest_neighbors(training_set, testing_set[i], K)
		hypothesis_list.append(classify(nearest_neighbors))
	(accuracy, error) = get_accuracy(testing_set,hypothesis_list)
	#print("ours-Accuracy: %s%% and error: %s%% " % (accuracy, error))
	return (accuracy,error)

def nearest_neighbors_scikit(training_set, testing_set):
	"""
	scikit-learn's nearest neighbor classifier 
	we run the classifier using theirs in order to give validity to our results 
	"""
	(train_X, train_y) = seperate_attribute_and_label(training_set)
	neigh = KNeighborsClassifier(n_neighbors=K)
	neigh.fit(train_X, train_y)
	(test_X, test_y) = seperate_attribute_and_label(testing_set)
	test_result_y = neigh.predict(test_X)
	(accuracy, error) = get_accuracy_scikit(test_y,test_result_y)
	#print("scikit-Accuracy: %s%% and error: %s%% " % (accuracy, error))
	return (accuracy,error)

def add_a_new_dimension(dataset):
	"""
	adds a new dimension to the specified dataset 
	values are added based on the normal distribution 
	"""
	random_dim1 = np.absolute(np.random.normal(3.5, 1, len(dataset)))
	i = 0
	for row in dataset:
		row.insert(0,int((random_dim1[i] * 100) + 0.5) / 100.0)
		i += 1
	return dataset

def average_for_runs(dataset, num_runs):
	"""
	run our k-NN classifier and scikit learn's k-NN for a specified number 
	of times. return the average error rate over the total number of runs
	"""
	our_total_acc = 0
	scikit_total_acc = 0

	for i in range(num_runs):
		training_set = []
		testing_set = []
		(training_set, testing_set) = split_dataset(dataset, PROBABILITY_TRAINING_SET)
		#print("training : " + str(len(training_set)) + " testing : " + str(len(testing_set)))
		(acc_ours,error_ours) = nearest_neighbors_implementation(training_set,testing_set)

		(acc_scikit, error_scikit) = nearest_neighbors_scikit(training_set,testing_set)
		#print("\n")
		#run_list.append([acc_ours,acc_scikit])	
		our_total_acc += acc_ours
		scikit_total_acc += acc_scikit
	
	return(our_total_acc/100.0,scikit_total_acc/100.0)

def generate_data_with_irrelevent_attributes(dataset,num_groups = 0):
	"""
	modifies the input dataset by adding irrelevant attributes
	based on the number of groups received as a parameter 
	e.g: num_groups = 2 --> add 2 dimensions of irrelevant attributes
	if no value is passed in, then no irrelevant attributes are added 
	"""
	#dataset = load_dataset(filename) # "iris-modified.csv"
	for i in range(num_groups):
		dataset = add_a_new_dimension(dataset)
		dataset = add_a_new_dimension(dataset)
	return dataset


orginial_dataset = load_dataset("iris-modified.csv") # "iris-modified.csv"
for i in range(40):
	print("iteration : " + str(i))
	dataset = generate_data_with_irrelevent_attributes(orginial_dataset,i)
	print([i, average_for_runs(dataset, 100)])
# TODO plot a graph using matplotlib 

#(training_set, testing_set) = split_dataset(dataset, PROBABILITY_TRAINING_SET)
#print("training : " + str(len(training_set)) + " testing : " + str(len(testing_set)))
#knn_scikit_learn(training_set,testing_set)
