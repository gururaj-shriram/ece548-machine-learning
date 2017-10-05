#
# nearest-neighbor.py  
# 
# date last modified: 4 oct 2017
# modified last by: jerry
#

from random import randint
import math
import operator

PROBABILITY_TRAINING_SET = 0.7
K = 3

# list that stores the classifier's class label for each example 
# for instance, element 0 stores the majority vote for the first example
hypothesis_list = []

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
		example = line.split(",") # a row in the data-set 
		dataset.append(example) # append it to the 2D array

	# need to remove newline character from row 
	for x in range(len(dataset)):
		row = dataset[x][-1] 
		dataset[x][-1] = row[0:-1]

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
	hypothesis_list.append(frequency_list[0][0])



dataset = load_dataset("iris.data")
(training_set, testing_set) = split_dataset(dataset, PROBABILITY_TRAINING_SET)
print("training : " + str(len(training_set)) + " testing : " + str(len(testing_set)))
#test = calculate_distance(training_set, testing_set[0])
for i in range(len(testing_set)):
	nearest_neighbors = get_nearest_neighbors(training_set, testing_set[i], K)
	classify(nearest_neighbors)

print("=================================================")
print(hypothesis_list)



