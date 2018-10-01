# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:59:55 2018

@author: mathias.engel
"""

#######################################################################################
##################### 1. Import der benötigten Bibs ###################################
import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
#import matplotlib.pyplot
# ensure the plots are inside this notebook, not an external window
#%matplotlib inline

#######################################################################################
##################### 2. Anpassung der Variablen ######################################
# number of input, hidden and output nodes
input_nodes = 4
hidden_nodes = 10
output_nodes = 3

# learning rate
learning_rate = 0.3

# epochs is the number of times the training data set is used for training
epochs = 100

# load the training data CSV file into a list
training_data_file = open("data/iris_dataset/iris_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# load the test data CSV file into a list
test_data_file = open("data/iris_dataset/iris_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#######################################################################################
##################### 2.1 Maximalfeaturewert auslesen #################################
# max_test_train_set um alle Werte in eine Liste zu schreiben
max_test_train_set = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = numpy.asfarray(all_values[1:])
    max_test_train_set.append(inputs)  
    pass

for record in training_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = numpy.asfarray(all_values[1:])
    max_test_train_set.append(inputs)  
    pass

#print(max_test_train_set)

max_test_train_value=numpy.amax(max_test_train_set)

#print(max_test_value)

#######################################################################################
##################### 3. Klasse des Neuronalen Netzes #################################
# neural network class definition
class neuralNetwork:
    
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

#######################################################################################
##################### 4. Erstellen eines Objekts der obigen Klasse ####################
    
# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

#######################################################################################
##################### 5. Das Netz basierend auf den Epochen trainieren ################
 

# train the neural network
for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / max_test_train_value*0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

#######################################################################################
##################### 6. Das Netz auf Basis der Testdaten prüfen ######################

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / max_test_train_value*0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    print(label,correct_label,scorecard)
    pass

#######################################################################################
##################### 7. Ausgabe der Genauigkeit des Netzes (Performance) #############

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)

#######################################################################################
#######################################################################################

##################### Testabfrage für neue eigene Eingangswerte ###########################################
##################### Mutmaßt das Netz die richtige Antwort? ##############################################
# wähle dazu einen passenden Datensatz und trage die entsprechenden Featurewerte hier ein (bsp. Label=2 und Feature=5,2,3.5,1)

#n.query([5,2,3.5,1])