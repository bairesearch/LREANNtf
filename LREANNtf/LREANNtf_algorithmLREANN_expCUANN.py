"""LREANNtf_algorithmLREANN_expCUANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LREANNtf_main.py

# Usage:
see LREANNtf_main.py

# Description:
LREANNtf algorithm LREANN expCUANN - define learning rule experiment artificial neural network with common (activation path) update

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *
import ANNtf2_operations
import ANNtf2_globalDefs
import math
from numpy import random


debugWexplosion = True
debugFastTrain = True

learningRate = 0.0001
enableForgetting = True
if(enableForgetting):
	enableForgettingRestrictToAPrevAndNotAConnections = True	#True	#this ensures that only connections between active lower layer neurons and unactive higher layer exemplar neurons are suppressed
	forgetRate = learningRate	#CHECKTHIS

useBatch = True
if(useBatch):
	batchSize = 10
else:
	batchSize = 1	
	
applyWmaxCap = True
if(applyWmaxCap):
	applyWmaxCapValue = 5.0	#max W = 1
applyAmaxCap = False	
if(applyAmaxCap):
	applyAmaxCapValue = 5.0	#max A = 1
	
onlyTrainNeuronsIfActivationContributionAboveThreshold1 = False
if(onlyTrainNeuronsIfActivationContributionAboveThreshold1):
	onlyTrainNeuronsIfActivationContributionAboveThreshold1Value = 0.5
onlyTrainNeuronsIfActivationContributionAboveThreshold2 = False	#theshold neurons which will be positively biased, and those which will be negatively (above a = 0 as it is currently) 
if(onlyTrainNeuronsIfActivationContributionAboveThreshold2):
	onlyTrainNeuronsIfActivationContributionAboveThreshold2Value = 0.5

biologicalConstraints = False	#batchSize=1, _?

sparsityLevel = 1.0	#probability of initial strong neural connection per neuron in layer

noisySampleGeneration = False
noisySampleGenerationNumSamples = 0
noiseStandardDeviation = 0

if(biologicalConstraints):
	useBinaryWeights = True	#increases stochastically updated training speed, but reduces final accuracy
	if(useBinaryWeights):	
		averageTotalInput = -1
		useBinaryWeightsReduceMemoryWithBool = False	#can use bool instead of float32 to limit memory required, but requires casting to float32 for matrix multiplications
	if(not useBatch):
		noisySampleGeneration = False	#possible biological replacement for input data batchSize > 1 (provides better performance than standard input data batchSize == 1, but less performance than input data batchSize > 10+)
		if(noisySampleGeneration):
			noisySampleGenerationNumSamples = 10
			noiseStandardDeviation = 0.03
else:
	useBinaryWeights = False

normaliseFirstLayer = False
equaliseNumberExamplesPerClass = False	
	

W = {}
B = {}
#Atrace = {}


#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0
datasetNumClasses = 0

#randomNormal = tf.initializers.RandomNormal()
	
def getNoisySampleGenerationNumSamples():
	return noisySampleGeneration, noisySampleGenerationNumSamples, noiseStandardDeviation
	
def defineTrainingParameters(dataset):
	
	if(debugFastTrain):
		trainingSteps = batchSize
	else:
		trainingSteps = 1000
	if(useBatch):
		numEpochs = 10
	else:
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
				W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(tf.random.normal([n_h[l-1], n_h[l]], stddev=sparsityLevel, dtype=dtype))		#tf.Variable(randomNormal([n_h[l-1], n_h[l]]))	
				#note stddev=sparsityLevel: a weakly tailed distribution for sparse activated network (such that the majority of weights are close to zero)
			B[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(tf.zeros(n_h[l], dtype=dtype))
			
			#Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")] = tf.Variable(tf.zeros(n_h[l], dtype=tf.dtypes.float32))

	
def neuralNetworkPropagation(x, networkIndex=1, recordAtrace=False):
	return neuralNetworkPropagationLREANN(x, networkIndex, recordAtrace)

def neuralNetworkPropagationLREANN(x, networkIndex=1, recordAtrace=False):
	
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

		if(applyAmaxCap):
			A = tf.clip_by_value(A, clip_value_min=-applyAmaxCapValue, clip_value_max=applyAmaxCapValue)
					
		#if(recordAtrace):
		#	if(onlyTrainNeuronsIfActivationContributionAboveThreshold1):
		#		#apply threshold to A
		#		AAboveThreshold = tf.math.greater(A, onlyTrainNeuronsIfActivationContributionAboveThreshold1Value)
		#		AAboveThresholdFloat = tf.dtypes.cast(AAboveThreshold, dtype=tf.float32)
		#		ALearn = A*AAboveThresholdFloat
		#	else:
		#		ALearn = A
		#	#print("ALearn.shape = ", ALearn.shape)
		#	Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")] = ALearn
			
		AprevLayer = A
		
	pred = tf.nn.softmax(Z)
	
	#print("neuralNetworkPropagationLREANN pred.shape = ", pred.shape)	

	return pred
	

def neuralNetworkPropagationLREANN_test(x, y, networkIndex=1):

	pred = neuralNetworkPropagationLREANN(x, networkIndex)
	loss = ANNtf2_operations.calculateLossCrossEntropy(pred, y, datasetNumClasses, costCrossEntropyWithLogits=False)
	acc = ANNtf2_operations.calculateAccuracy(pred, y)
	
	return loss, acc
	
def neuralNetworkPropagationLREANN_expCUANNtrain(x, y, networkIndex=1):	#currentClassTarget
	
	#debug:
	#print("batchSize = ", batchSize)
	#print("learningRate = ", learningRate)
	#print("x = ", x)
	
	#important notes;
	#the highest layer output (A) being forward propagated from x is never used during training, as the output neuron actually being trained is defined in y
		
	#predExemplars = neuralNetworkPropagationLREANN(exemplarsX, networkIndex, recordAtrace=True)	#record exemplar activation traces
	
	AprevLayer = x

	for l in range(1, numberOfLayers+1):

		if(debugWexplosion):
			print("l = " + str(l))
			print("W[generateParameterNameNetwork(networkIndex, l, W)] = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
		
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

		if(applyAmaxCap):
			A = tf.clip_by_value(A, clip_value_min=-applyAmaxCapValue, clip_value_max=applyAmaxCapValue)

		if(l == numberOfLayers):
			A = tf.one_hot(y, depth=datasetNumClasses) #replace final layer A with y values during train (discard forward propagation values)
			#print("A.shape = ", A.shape)
	
		if(onlyTrainNeuronsIfActivationContributionAboveThreshold1):
			#apply threshold to AprevLayer
			AaboveThreshold = tf.math.greater(A, onlyTrainNeuronsIfActivationContributionAboveThreshold1Value)
			AaboveThresholdFloat = tf.dtypes.cast(AaboveThreshold, dtype=tf.float32)
			Alearn = A*AaboveThresholdFloat
		else:
			Alearn = A		
		if(onlyTrainNeuronsIfActivationContributionAboveThreshold2):
			#apply threshold to AprevLayer
			AprevLayerAboveThreshold = tf.math.greater(AprevLayer, onlyTrainNeuronsIfActivationContributionAboveThreshold2Value)
			AprevLayerAboveThresholdFloat = tf.dtypes.cast(AprevLayerAboveThreshold, dtype=tf.float32)
			AprevLayerLearn = AprevLayer*AprevLayerAboveThresholdFloat
		else:
			AprevLayerLearn = AprevLayer

		#update weights based on hebbian learning rule
		#strengthen those connections that link the previous layer neuron to the activation trace for the class target (and weaken those that did not)

		AcoincidenceMatrix = tf.matmul(tf.transpose(AprevLayerLearn), A)
		Wmod = AcoincidenceMatrix/batchSize*learningRate
		W[generateParameterNameNetwork(networkIndex, l, "W")] = W[generateParameterNameNetwork(networkIndex, l, "W")] + Wmod	#apply weight update

		if(enableForgetting):	#this isn't necessarily required for highly sparsely activated network + low shot learning
			if(enableForgettingRestrictToAPrevAndNotAConnections):
				AboolNeg = tf.math.equal(Alearn, 0.0)	#Abool = tf.math.greater(Alearn, 0.0), AboolNeg = tf.math.logical_not(Abool)
				#AboolNegInt = tf.dtypes.cast(AboolNeg, tf.int32)
				AboolNegFloat = tf.dtypes.cast(AboolNeg, tf.float32)
				AcoincidenceMatrixForget = tf.matmul(tf.transpose(AprevLayerLearn), AboolNegFloat)
				Wmod2 = AcoincidenceMatrixForget/batchSize*forgetRate	#tf.square(AcoincidenceMatrixForget) - square is required to normalise the forget rate relative to the learn rate [assumes input tensor is < 1]
				#print("Wmod2 = ", Wmod2)
				W[generateParameterNameNetwork(networkIndex, l, "W")] = W[generateParameterNameNetwork(networkIndex, l, "W")] - Wmod2
			else:
				AcoincidenceMatrixIsZero = tf.math.equal(AcoincidenceMatrix, 0)
				#AcoincidenceMatrixIsZeroInt = tf.dtypes.cast(AcoincidenceMatrixIsZero, tf.int32)
				AcoincidenceMatrixIsZeroFloat = tf.dtypes.cast(AcoincidenceMatrixIsZero, dtype=tf.float32)
				Wmod2 = AcoincidenceMatrixIsZeroFloat/batchSize*forgetRate	#tf.square(AcoincidenceMatrixIsZeroFloat) - square is required to normalise the forget rate relative to the learn rate [assumes input tensor is < 1]
				W[generateParameterNameNetwork(networkIndex, l, "W")] = W[generateParameterNameNetwork(networkIndex, l, "W")] - Wmod2

		if(applyWmaxCap):
			W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.clip_by_value(W[generateParameterNameNetwork(networkIndex, l, "W")], clip_value_min=-applyWmaxCapValue, clip_value_max=applyWmaxCapValue)
				
		AprevLayer = A	#Alearn

	#clear Atrace;
	#for l in range(1, numberOfLayers+1):
	#	Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")] = 0	#tf.zeros(n_h[l])
		
	pred = neuralNetworkPropagationLREANN(x, networkIndex)
	
	return pred

	

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

 


def generateTFtrainDataFromNParraysLREANN_expCUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses):

	trainDataList = []
	
	for classTarget in range(datasetNumClasses):
			
		train_xClassFiltered, train_yClassFiltered = filterNParraysByClassTarget(train_x, train_y, classTargetFilterIndex=classTarget)
		trainDataUnbatched = generateTFtrainDataUnbatchedFromNParrays(train_xClassFiltered, train_yClassFiltered)
		trainData = generateTFtrainDataFromTrainDataUnbatched(trainDataUnbatched, shuffleSize, batchSize)
		trainDataList.append(trainData)
		
	return trainDataList
		

