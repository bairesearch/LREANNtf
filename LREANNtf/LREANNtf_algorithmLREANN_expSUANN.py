"""LREANNtf_algorithmLREANN_expSUANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LREANNtf_main.py

# Usage:
see LREANNtf_main.py

# Description:
LREANNtf algorithm LREANN expSUANN - define learning rule experiment artificial neural network with stochastic update

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs
import math
from numpy import random

debugLongTrain = False	#increased number of training epochs (100)
debugPrintLossVerbose = False	#useful for debugging
debugOnlyTrainFinalLayer = False	#for performance comparison tests


stochasticUpdateNeurons = True	#default: True
if(not stochasticUpdateNeurons):
	stochasticUpdateLayers = True
	#stochasticUpdateNetwork = False	#not yet coded
	if(stochasticUpdateLayers):
		trainLayersUseHebbianHeuristic = True
		if(trainLayersUseHebbianHeuristic):
			trainLayersUseHebbianHeuristicMultidirection = True	#try both directions (non/coincident with current activation path)
		#else
			#not currently possible since never any negative weight mod applied

	
biologicalConstraints = True	#batchSize=1, _?

useBatch = True
noisySampleGeneration = False
noisySampleGenerationNumSamples = 0
noiseStandardDeviation = 0

useBinaryWeights = True
if(biologicalConstraints):
	useBinaryWeights = True	#increases stochastically updated training speed, but reduces final accuracy
	if(useBinaryWeights):	
		averageTotalInput = -1
		useBinaryWeightsReduceMemoryWithBool = True	#can use bool instead of float32 to limit memory required, but requires casting to float32 for matrix multiplications
	numberOfSubsetsTrialledPerBaseParameter = 1
	parameterUpdateSubsetSize = 1
	if(not useBinaryWeights):
		useBatch = False
		noisySampleGeneration = True	#possible biological replacement for input data batchSize > 1 (provides better performance than standard input data batchSize == 1, but less performance than input data batchSize > 10+)
		if(noisySampleGeneration):
			noisySampleGenerationNumSamples = 10
			noiseStandardDeviation = 0.03
else:
	updateParameterSubsetSimultaneously = False	#current tests indiciate this is not required/beneficial with significantly high batch size
	if(updateParameterSubsetSimultaneously):
		numberOfSubsetsTrialledPerBaseParameter = 10	#decreases speed, but provides more robust parameter updates
		parameterUpdateSubsetSize = 5	#decreases speed, but provides more robust parameter updates
	else:
		numberOfSubsetsTrialledPerBaseParameter = 1
		parameterUpdateSubsetSize = 1	

normaliseFirstLayer = False
equaliseNumberExamplesPerClass = False
	
	

W = {}
B = {}

Wbackup = {}
Bbackup = {}

NETWORK_PARAM_INDEX_TYPE = 0
NETWORK_PARAM_INDEX_LAYER = 1
NETWORK_PARAM_INDEX_H_CURRENT_LAYER = 2
NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER = 3
NETWORK_PARAM_INDEX_VARIATION_DIRECTION = 4

#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0
datasetNumClasses = 0

learningRate = 0.0
batchSize = 0

#randomNormal = tf.initializers.RandomNormal()

def getNoisySampleGenerationNumSamples():
	return noisySampleGeneration, noisySampleGenerationNumSamples, noiseStandardDeviation
	
def defineTrainingParameters(dataset):

	global learningRate
	global forgetRate
	global batchSize
	
	if(biologicalConstraints):
		if(useBatch):
			learningRate = 0.01
		else:
			learningRate = 0.001
	else:
		learningRate = 0.001
	trainingSteps = 1000
	if(useBatch):
		batchSize = 100
		if(debugLongTrain):
			numEpochs = 100
		else:
			numEpochs = 10
	else:
		batchSize = 1
		numEpochs = 100
	
	displayStep = 100

	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	

def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	global datasetNumClasses
	
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet)
	
	return numberOfLayers


def defineNeuralNetworkParameters():
	
	tf.random.set_seed(5);
	if(useBinaryWeights):
		if(useBinaryWeightsReduceMemoryWithBool):
			dtype=tf.dtypes.bool
		else:
			dtype=tf.dtypes.float32
	else:
		#randomNormal = tf.initializers.RandomNormal()
		dtype=tf.dtypes.float32
	
	for networkIndex in range(1, numberOfNetworks+1):
	
		for l in range(1, numberOfLayers+1):

			if(useBinaryWeights):
				Wint = tf.random.uniform([n_h[l-1], n_h[l]], minval=0, maxval=2, dtype=tf.dtypes.int32)		#The lower bound minval is included in the range, while the upper bound maxval is excluded.
				W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(tf.dtypes.cast(Wint, dtype=dtype))
				#print("W[generateParameterNameNetwork(networkIndex, l, W)] = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
			else:
				W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(tf.random.normal([n_h[l-1], n_h[l]], dtype=dtype))		#tf.Variable(randomNormal([n_h[l-1], n_h[l]]))
			B[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(tf.zeros(n_h[l], dtype=dtype))
			
			Wbackup[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(W[generateParameterNameNetwork(networkIndex, l, "W")])
			Bbackup[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(B[generateParameterNameNetwork(networkIndex, l, "B")])
	
			#print(W[generateParameterNameNetwork(networkIndex, l, "W")])
			#print(Wbackup[generateParameterNameNetwork(networkIndex, l, "W")])
			#exit()
				
def neuralNetworkPropagation(x, networkIndex=1):
	return neuralNetworkPropagationLREANN(x, networkIndex)
	
def neuralNetworkPropagationLREANN(x, networkIndex=1):
	
	global averageTotalInput
		
	AprevLayer = x

	if(useBinaryWeights):	
		if(averageTotalInput == -1):
			averageTotalInput = tf.math.reduce_mean(x)
			print("averageTotalInput = ", averageTotalInput)
			 
	#print("x = ", x)
	
	for l in range(1, numberOfLayers+1):
	
		#print("l = " + str(l))
		
		if(useBinaryWeights):
			if(useBinaryWeightsReduceMemoryWithBool):
				Wfloat = tf.dtypes.cast(W[generateParameterNameNetwork(networkIndex, l, "W")], dtype=tf.float32)
				Bfloat = tf.dtypes.cast(B[generateParameterNameNetwork(networkIndex, l, "B")], dtype=tf.float32)
				Z = tf.add(tf.matmul(AprevLayer, Wfloat), Bfloat)
			else:
				Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
			A = activationFunction(Z, n_h[l-1])
		else:
			Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
			A = activationFunction(Z)
			
		AprevLayer = A
			
	return tf.nn.softmax(Z)
	

def neuralNetworkPropagationLREANN_test(x, y, networkIndex=1):

	pred = neuralNetworkPropagationLREANN(x, networkIndex)
	loss = ANNtf2_operations.calculateLossCrossEntropy(pred, y, datasetNumClasses, costCrossEntropyWithLogits=False)
	acc = ANNtf2_operations.calculateAccuracy(pred, y)
	
	return loss, acc


def neuralNetworkPropagationLREANN_expSUANNtrain(x, y=None, networkIndex=1):

	if(stochasticUpdateNeurons):
		neuralNetworkPropagationLREANN_expSUANNtrain_updateNeurons(x, y, networkIndex)
	elif(stochasticUpdateLayers):
		neuralNetworkPropagationLREANN_expSUANNtrain_updateLayers(x, y, networkIndex)
	#elif(stochasticUpdateNetwork):
	#	neuralNetworkPropagationLREANN_expSUANNtrain_updateNetwork(x, y, networkIndex)
		

def neuralNetworkPropagationLREANN_expSUANNtrain_updateLayers(x, y=None, networkIndex=1):

	#print("batchSize = ", batchSize)
	#print("learningRate = ", learningRate)
	
	#print("x = ", x)
	
	lossBase, accBase = neuralNetworkPropagationLREANN_test(x, y, networkIndex)	#debug only
	#print("lossBase = ", lossBase)
		
	if(trainLayersUseHebbianHeuristic):
		AprevLayer = x
	
	if(useBinaryWeights):
		if(useBinaryWeightsReduceMemoryWithBool):
			dtype=tf.dtypes.bool
		else:
			dtype=tf.dtypes.float32	
	else:
		dtype=tf.dtypes.float32	
	
	for l in range(1, numberOfLayers+1):
	
		#print("\nl = " + str(l))
		
		WbackupLocal = W[generateParameterNameNetwork(networkIndex, l, "W")]
		
		if(trainLayersUseHebbianHeuristic):
			Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])	
			A = activationFunction(Z, n_h[l-1])
			
			AcoincidenceMatrix = tf.matmul(tf.transpose(AprevLayer), A)
			
			AprevLayer = A
			
			if(trainLayersUseHebbianHeuristicMultidirection):
				direction = random.randint(0, 1)
			else:
				direction = 0
			if(direction == 1):	#apply negative
				AcoincidenceMatrix = tf.negative(AcoincidenceMatrix)
			WmodRandom = AcoincidenceMatrix
		else:
			if(useBinaryWeights):
				Wint = tf.random.uniform(WbackupLocal.shape, minval=0, maxval=2, dtype=tf.dtypes.int32)		#The lower bound minval is included in the range, while the upper bound maxval is excluded.
				WmodRandom = tf.Variable(tf.dtypes.cast(Wint, dtype=dtype))
			else:
				WmodRandom = tf.random.normal(WbackupLocal.shape, dtype=dtype)
	
		if(useBinaryWeights):
			Wmod = WmodRandom
			W[generateParameterNameNetwork(networkIndex, l, "W")] = Wmod
		else:
			Wmod = WmodRandom*learningRate
			#print("Wmod = ", Wmod)
			W[generateParameterNameNetwork(networkIndex, l, "W")] = W[generateParameterNameNetwork(networkIndex, l, "W")] + Wmod
				
		loss, acc = neuralNetworkPropagationLREANN_test(x, y, networkIndex)
		#print("loss = ", loss)

		accuracyImprovementDetected = False
		if(loss < lossBase):
			accuracyImprovementDetected = True
			lossBase = loss
			#print("\t(loss < lossBase): loss = ", loss)						

		if not accuracyImprovementDetected:
			#print("!accuracyImprovementDetected")
			#print("WbackupLocal = ", WbackupLocal)
			W[generateParameterNameNetwork(networkIndex, l, "W")] = WbackupLocal		
		#else:
			#print("accuracyImprovementDetected")

	pred = neuralNetworkPropagationLREANN(x, networkIndex)
	
	return pred
					
					
					
def neuralNetworkPropagationLREANN_expSUANNtrain_updateNeurons(x, y=None, networkIndex=1):

	#print("batchSize = ", batchSize)
	#print("learningRate = ", learningRate)
	
	#print("x = ", x)

	if(debugOnlyTrainFinalLayer):
		minLayerToTrain = numberOfLayers
	else:
		minLayerToTrain = 1
		
			
	lossStart, accStart = neuralNetworkPropagationLREANN_test(x, y, networkIndex)	#debug only
	if(debugPrintLossVerbose):
		print("lossStart = ", lossStart)
	
	#ensure that an update is tried at least once for each parameter of the network during each training iteration:
	
	if(useBinaryWeights):
		variationDirections = 1
	else:
		variationDirections = 2
	
	for l in range(minLayerToTrain, numberOfLayers+1):
		
		#print("l = ", l)
		#print("W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
		#print("B = ", B[generateParameterNameNetwork(networkIndex, l, "B")])
			
		for hIndexCurrentLayer in range(0, n_h[l]):
			for hIndexPreviousLayer in range(0, n_h[l-1]+1):
				if(hIndexPreviousLayer == n_h[l-1]):	#ensure that B parameter updates occur/tested less frequently than W parameter updates
					parameterTypeWorB = 0
				else:
					parameterTypeWorB = 1
				for variationDirectionInt in range(variationDirections):
	
					networkParameterIndexBase = (parameterTypeWorB, l, hIndexCurrentLayer, hIndexPreviousLayer, variationDirectionInt)
			
					lossBase, accBase = neuralNetworkPropagationLREANN_test(x, y, networkIndex)
					
					#print("hIndexCurrentLayer = ", hIndexCurrentLayer)
					#print("hIndexPreviousLayer = ", hIndexPreviousLayer)
					#print("parameterTypeWorB = ", parameterTypeWorB)
					#print("variationDirectionInt = ", variationDirectionInt)
					
					for subsetTrialIndex in range(0, numberOfSubsetsTrialledPerBaseParameter):

						accuracyImprovementDetected = False

						currentSubsetOfParameters = []
						currentSubsetOfParameters.append(networkParameterIndexBase)

						for s in range(1, parameterUpdateSubsetSize):
							networkParameterIndex = getRandomNetworkParameter(networkIndex, currentSubsetOfParameters)
							currentSubsetOfParameters.append(networkParameterIndex)

						for s in range(0, parameterUpdateSubsetSize):
							networkParameterIndex = currentSubsetOfParameters[s]
							
							if(not useBinaryWeights):
								if(networkParameterIndex[NETWORK_PARAM_INDEX_VARIATION_DIRECTION] == 1):
									variationDiff = learningRate
								else:
									variationDiff = -learningRate							
														
							if(networkParameterIndex[NETWORK_PARAM_INDEX_TYPE] == 1):
								#Wnp = W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")].numpy()
								#currentVal = Wnp[networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]]
								currentVal = W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")][networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].numpy()
								
								#print("currentVal = ", currentVal)
								#print("W1 = ", W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")])
								if(useBinaryWeights):
									if(useBinaryWeightsReduceMemoryWithBool):
										newVal = not currentVal
									else:
										newVal = float(not bool(currentVal))
										#print("newVal = ", newVal)
								else:
									newVal = currentVal + variationDiff
								W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")][networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].assign(newVal)
						
								#print("W2 = ", W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")])
							else:
								#Bnp = B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")].numpy()
								#currentVal = Bnp[networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]]
								currentVal = B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")][networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].numpy()
								
								if(useBinaryWeights):
									if(useBinaryWeightsReduceMemoryWithBool):
										newVal = not currentVal
									else:
										newVal = float(not bool(currentVal))
								else:
									newVal = currentVal + variationDiff
								B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")][networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].assign(newVal)
			
						loss, acc = neuralNetworkPropagationLREANN_test(x, y, networkIndex)
						#print("loss = ", loss)
						
						if(loss < lossBase):
							accuracyImprovementDetected = True
							lossBase = loss
							#print("\t(loss < lossBase): loss = ", loss)						
						
						if(accuracyImprovementDetected):
							#print("accuracyImprovementDetected")
							Wbackup[generateParameterNameNetwork(networkIndex, l, "W")].assign(W[generateParameterNameNetwork(networkIndex, l, "W")])
							Bbackup[generateParameterNameNetwork(networkIndex, l, "B")].assign(B[generateParameterNameNetwork(networkIndex, l, "B")])								
						else:
							#print("!accuracyImprovementDetected")
							W[generateParameterNameNetwork(networkIndex, l, "W")].assign(Wbackup[generateParameterNameNetwork(networkIndex, l, "W")])
							B[generateParameterNameNetwork(networkIndex, l, "B")].assign(Bbackup[generateParameterNameNetwork(networkIndex, l, "B")])					
		
	pred = neuralNetworkPropagationLREANN(x, networkIndex)
	
	return pred
									

def getRandomNetworkParameter(networkIndex, currentSubsetOfParameters):
	
	foundNewParameter = False
	while not foundNewParameter:
	
		variationDirection = random.randint(2)
		layer = random.randint(1, len(n_h))
		parameterTypeWorBtemp = random.randint(n_h[layer-1]+1)	#ensure that B parameter updates occur/tested less frequently than W parameter updates	#OLD: random.randint(2)	
		if(parameterTypeWorBtemp == n_h[layer-1]):
			parameterTypeWorB = 0
		else:
			parameterTypeWorB = 1
		hIndexCurrentLayer = random.randint(n_h[layer])	#randomNormal(n_h[l])
		hIndexPreviousLayer = random.randint(n_h[layer-1]) #randomNormal(n_h[l-1])
		networkParameterIndex = (parameterTypeWorB, layer, hIndexCurrentLayer, hIndexPreviousLayer, variationDirection)
	
		matches = [item for item in currentSubsetOfParameters if item == networkParameterIndex]
		if len(matches) == 0:
			foundNewParameter = True
			
	return networkParameterIndex



def activationFunction(Z, prevLayerSize=None):
	return reluCustom(Z, prevLayerSize)
			
def reluCustom(Z, prevLayerSize=None):
	
	if(useBinaryWeights):	
		#offset required because negative weights are not used:
		Zoffset = tf.ones(Z.shape)
		Zoffset = tf.multiply(Zoffset, averageTotalInput)
		Zoffset = tf.multiply(Zoffset, prevLayerSize/2)
		#print("Zoffset = ", Zoffset)
		Z = tf.subtract(Z, Zoffset) 
		A = tf.nn.relu(Z)
		#AaboveZero = tf.math.greater(A, 0)
		#AaboveZeroFloat = tf.dtypes.cast(AaboveZero, dtype=tf.dtypes.float32)
		#ZoffsetRestore = AaboveZeroFloat*Zoffset
		#print("ZoffsetRestore = ", ZoffsetRestore)
		#A = tf.add(A, ZoffsetRestore)
	else:
		A = tf.nn.relu(Z)
	
	#print("Z = ", Z)
	#print("A = ", A)
	
	return A

  
