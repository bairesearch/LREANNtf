"""LREANNtf_algorithmLREANN_expNUANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LREANNtf_main.py

# Usage:
see LREANNtf_main.py

# Description:
LREANNtf algorithm LREANN expNUANN - define learning rule experiment artificial neural network with (neuron activation) normalisation update

also

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs
import math

debugSmallBatchSize = False

trainBackprop = True	#default: False (train weights based on normalisation of layer activations) #True: use backpropagation for training (test biased activationFunction only; applyNeuronThresholdBias)
if(trainBackprop):
	onlyTrainFinalLayer = False
else:
	onlyTrainFinalLayer = True
	
applyNeuronThresholdBias = True	#default: True #simulate biological neurons having a positive fire threshold req of +x and each synapse being initialised with a positive weight +w
if(applyNeuronThresholdBias):
	applyNeuronThresholdBiasValue = 1.0
	zeroParametersIfViolateEItypeCondition = True
	verifyParametersDoNotViolateEItypeCondition = True
	constrainBiases = False	#default: False
	if(constrainBiases):
		constrainBiasesLastLayer = False

normaliseFirstLayer = True	#require inputs normalised between -1 and 1 (first hidden layer neurons are entirely excitatory)
equaliseNumberExamplesPerClass = True

noisySampleGeneration = False


W = {}
B = {}

#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0

batchSize = 0
learningRate = 0.0

def defineTrainingParameters(dataset):
	global batchSize
	global learningRate
	
	learningRate = 0.001
	if(debugSmallBatchSize):
		batchSize = 10
	else:
		batchSize = 100
	numEpochs = 10	#100 #10
	trainingSteps = 10000	#1000

	displayStep = 100
			
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	


def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, generateLargeNetwork=False)
	
	return numberOfLayers
	
def defineNeuralNetworkParameters():

	print("numberOfNetworks", numberOfNetworks)
	
	for networkIndex in range(1, numberOfNetworks+1):
		
		for l1 in range(1, numberOfLayers+1):
			if(applyNeuronThresholdBias):
				meanWeight = applyNeuronThresholdBiasValue/n_h[l1-1]
				Wlayer = tf.Variable(tf.random.uniform([n_h[l1-1], n_h[l1]], minval=0, maxval=meanWeight*2))
			else:
				Wlayer = tf.Variable(tf.random.normal([n_h[l1-1], n_h[l1]]))
			W[generateParameterNameNetwork(networkIndex, l1, "W")] = Wlayer
			B[generateParameterNameNetwork(networkIndex, l1, "B")] = tf.Variable(tf.zeros(n_h[l1]))
					
def neuralNetworkPropagation(x, networkIndex=1):
	return neuralNetworkPropagationANN(x, networkIndex)

def neuralNetworkPropagationLayer(x, networkIndex=1, l=None):
	return neuralNetworkPropagationANN(x, networkIndex, l)
		
def neuralNetworkPropagationANN(x, networkIndex=1, l=None):
			
	#print("numberOfLayers", numberOfLayers)

	if(l == None):
		maxLayer = numberOfLayers
	else:
		maxLayer = l
			
	if(applyNeuronThresholdBias):
		x = x + applyNeuronThresholdBiasValue	#ensure neuron input averages to 1	#requires calibration
	#print("x = ", x)
	
	AprevLayer = x

	for l1 in range(1, maxLayer+1):
		Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l1, "W")]), B[generateParameterNameNetwork(networkIndex, l1, "B")])
		
		A = activationFunction(Z)

		#print("l1 = " + str(l1))		
		#print("W = ", W[generateParameterNameNetwork(networkIndex, l1, "W")] )
		#print("Z = ", Z)
		#print("A = ", A)

		if(onlyTrainFinalLayer):
			if(l1 < numberOfLayers):
				A = tf.stop_gradient(A)
							
		AprevLayer = A

	if(maxLayer == numberOfLayers):
		return tf.nn.softmax(Z)
	else:
		return A


def neuralNetworkPropagationLREANN_expNUANNtrain(x, y, networkIndex=1):

	maxLayer = numberOfLayers

	if(applyNeuronThresholdBias):
		x = x + applyNeuronThresholdBiasValue	#ensure neuron input averages to 1	#requires calibration
	#print("x = ", x)

	AprevLayer = x

	for l1 in range(1, maxLayer+1):
		Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l1, "W")]), B[generateParameterNameNetwork(networkIndex, l1, "B")])

		A = activationFunction(Z)
	
		if(l1 < maxLayer):
			Zaverage = tf.reduce_mean(Z, axis=0)
			Zaverage = Zaverage - applyNeuronThresholdBiasValue
			Wmod = -(Zaverage*learningRate)
			Wmod = tf.expand_dims(Wmod, axis=0)
			multiples = tf.constant([n_h[l1-1],1], tf.int32)
			Wmod = tf.tile(Wmod, multiples)
			#print("Wmod.shape = ", Wmod.shape)
			#print("Wmod = ", Wmod)

			Wlayer = W[generateParameterNameNetwork(networkIndex, l1, "W")] 
			Wlayer = Wlayer + Wmod
			W[generateParameterNameNetwork(networkIndex, l1, "W")] = Wlayer
		
		#print("l1 = " + str(l1))
		#print("W = ", W[generateParameterNameNetwork(networkIndex, l1, "W")] )
		#print("Z = ", Z)
		#print("A = ", A)
			
		AprevLayer = A	

def activationFunction(Z):

	if(applyNeuronThresholdBias):
		Z = Z - applyNeuronThresholdBiasValue
	
	A = tf.nn.relu(Z)
	
	if(applyNeuronThresholdBias):
		A = A*2	#renormalisation	#ensure neuron input averages to 1	#requires calibration
		#A = A + applyNeuronThresholdBiasValue
	
	return A

  
