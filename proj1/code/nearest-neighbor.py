#
# nearest-neighbor.py  
# 
# date last modified: 13 oct 2017
# modified last by: jerry
#

from random import randint
import math
import operator
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# change these to determine the dataset to be run 
FILENAME = "iris-modified.csv"
#FILENAME = "animals.data" 

# probability of an example being in the training set 
PROBABILITY_TRAINING_SET = 0.7
# normal distribution will center around this mean as a starting value
RANDOM_DATA_MEAN = 10
# normal distribution will center around this STD as a starting value
RANDOM_DATA_STD = 3
# maximum number of irrelevant groups to add; 5 groups --> 10 irrelevant attributes
NUM_GROUPS = 5
# number of times to run the classifier for a given number of groups 
NUM_RUNS = 50 
# max number of nearest neighbors to test 
MAX_K = 9


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

def get_nearest_neighbors(training_set, testing_record, K):
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
	for i in range(K):
		nearest_neighbors.append(sorted_distances[i])
	return nearest_neighbors


def get_accuracy(testing_set, hypothesis_set):
	"""
	returns a performance measure for our k-NN
	in particular an error rate and an accuracy rate (1 - error rate)

	@param testing_set: the testing set consisting of an example's correct class label
	@param hypothesis_set: consists of classifier's hypothesis of the example's class 
	"""
	total_miss = 0

	# get the correct class labels from the testing set 
	class_label = [row[-1] for row in testing_set]
	zipped = list(zip(hypothesis_set, class_label))
	# directly compare the class label to the hypothesis  
	for i in range(len(testing_set)):
		if zipped[i][0] != zipped[i][1]:
			total_miss += 1
	error_rate = total_miss * 100.0/len(testing_set)
	accuracy = 100 - error_rate
	return (accuracy,error_rate)

def get_accuracy_scikit(test_label, result_label):
	"""
	returns a performance measure for scikit-learn k-NN 
	"""
	zipped = list(zip(result_label, test_label))
	total_miss = 0
	for i in range(len(test_label)):
		if zipped[i][0] != zipped[i][1]:
			total_miss += 1
	error_rate = total_miss * 100.0/len(test_label)
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
	NOTE: no logic has been defined for splitting ties 
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
	# returns the first element of the sorted list 
	return frequency_list[0][0]


def nearest_neighbors_implementation(training_set,testing_set,K):
	"""
	our nearest neighbor classifier 
	"""

	# list that stores the classifier's class label for each example 
	# for instance, element 0 stores the majority vote for the first example
	hypothesis_list = []
	for i in range(len(testing_set)):
		nearest_neighbors = get_nearest_neighbors(training_set, testing_set[i], K)
		# classifier's hypothesis of the examples 
		hypothesis_list.append(classify(nearest_neighbors))
	(accuracy, error) = get_accuracy(testing_set,hypothesis_list)
	#print("ours-Accuracy: %s%% and error: %s%% " % (accuracy, error))
	return (accuracy,error)

def nearest_neighbors_scikit(training_set, testing_set, K):
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

	return (accuracy,error)

def add_a_new_dimension(dataset, random_itr):
	"""
	adds a new dimension to the specified dataset 
	values are added based on the normal distribution 
	"""
	# generate random values based on the normal distribution, using
	# a user-defined mean value that is scaled by the number of irrelevant
	# attribute groups. the scaling is done so that the irr attribute values 
	# grow as more groups are added
	random_dimension = np.absolute(np.random.normal(RANDOM_DATA_MEAN 
		* (random_itr+1), RANDOM_DATA_STD+random_itr, len(dataset)))
	i = 0
	# append an element from the distribution to the first element of the example
	# this is how we add a dimension  
	for row in dataset:
		row.insert(0,int((random_dimension[i] * 100) + 0.5) / 100.0)
		i += 1
	return dataset

def average_for_runs(dataset, num_runs, K):
	"""
	run our k-NN classifier and scikit learn's k-NN for a specified number 
	of times. return the average error rate over the total number of runs
	"""
	our_total_acc = 0
	scikit_total_acc = 0

	# here we perform random subsampling on the dataset. 
	# for the user-specified number of runs (say 100), continue the division
	# of the dataset into a training set and testing set. collect the accuracy
	# rate and divide by 100 to compute an overall average 
	for i in range(num_runs):
		training_set = []
		testing_set = []
		# randomly splits the dataset into a training and testing set 
		(training_set, testing_set) = split_dataset(dataset, PROBABILITY_TRAINING_SET)
		# run our nearest-neighbor classifier 
		(acc_ours, error_ours) = nearest_neighbors_implementation(training_set,testing_set,K)
		# run scikit learn nearest-neighbor classifier 
		(acc_scikit, error_scikit) = nearest_neighbors_scikit(training_set,testing_set,K)
		# total the accuracy rates (note: the error rate is not used) 
		our_total_acc += acc_ours
		scikit_total_acc += acc_scikit
	
	# finally divide by the number of runs to get the overall average 
	return(our_total_acc/float(num_runs), scikit_total_acc/float(num_runs))

def generate_data_with_irrelevent_attributes(dataset,num_groups = 0):
	"""
	modifies the input dataset by adding irrelevant attributes
	based on the number of groups received as a parameter 
	e.g: num_groups = 2 --> add 2 dimensions of irrelevant attributes
	if no value is passed in, then no irrelevant attributes are added 
	"""
	#dataset = load_dataset(filename) # "iris-modified.csv"
	for i in range(num_groups):
		dataset = add_a_new_dimension(dataset,num_groups)
		dataset = add_a_new_dimension(dataset,num_groups)
	return dataset

def run(filename, groups, max_k, num_runs):
	"""
	for an input list of K neighbors (e.g: 3, 5, and 7 neighbors), 
	run the nearest neighbors classifiers over a specified number of runs, 
	calculate the average accuracy, and append the results to two files:
	one for our nearest neighbor and another for sci-kit learn nearest neighbor.
	then repeat the above until the groups irrelevant attributes has been
	reached. 

	@param filename: points to file containing the dataset  
	@param groups: number of desired irrelevant attribute groups (added in pairs of 2)
	@param k_list: list of K nearest neighbors to test algorithm on  
	@param num_runs: number of times to run the classifier for a given group 

	"""
	# output files for plotting data; first create the header 
	f1=open("our_accuracy.txt","w+")
	for i in range(groups):
		f1.write(str(i*2)+", ")
	f1.write("\n")
	
	f2=open("scikit_accuracy.txt","w+")
	for i in range(groups):
		f2.write(str(i*2)+",")
	f2.write("\n")
	
	# start at 1-NN 
	current_k = 1 
	while current_k <= max_k:
		print("running k = " + str(current_k))
		our_accuracy = []
		scikit_accuracy = []
		for i in range(groups):
			#print("iteration : " + str(i))
			# each time we run a group, need to load a fresh dataset 
			org_data = load_dataset(filename) 
			dataset = generate_data_with_irrelevent_attributes(org_data,i)
			(ours_acc,sci_acc) = average_for_runs(dataset, num_runs, current_k)
			# printing format: 
			# [current iteration, our accuracy, scikit accuracy, # of attributes]
			print([i, ours_acc,sci_acc,len(dataset[0])])
			our_accuracy.append(ours_acc)
			scikit_accuracy.append(sci_acc)
		# write the results to file 
		for i in range(groups):
			f1.write(str(our_accuracy[i])+",")
			f2.write(str(scikit_accuracy[i])+",")	
		f1.write("\n")
		f2.write("\n")
		current_k += 2

	f1.close()
	f2.close()

#orginial_dataset = load_dataset(FILENAME) # "iris-modified.csv"
#for i in range(5):
#	print("iteration : " + str(i))
#	dataset = generate_data_with_irrelevent_attributes(orginial_dataset,i)
#	print([i, average_for_runs(dataset, 100,K)])

#(training_set, testing_set) = split_dataset(dataset, PROBABILITY_TRAINING_SET)

# run it! 
#run(FILENAME, NUM_GROUPS, [3,5,7,9], NUM_RUNS)
run(FILENAME, NUM_GROUPS, MAX_K, NUM_RUNS)


