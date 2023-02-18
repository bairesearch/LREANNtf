"""LREANNtf_algorithmLREANN_expMUANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LREANNtf_main.py

# Usage:
see LREANNtf_main.py

# Description:
LREANNtf algorithm LREANN expMUANN - define learning rule experiment artificial neural network with multi propagation (per layer; with synaptic delta calculation) update

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *
import ANNtf2_operations
import ANNtf2_globalDefs
import math
from numpy import random

debugActivationFunctionAllSoftmax = False

trainUsingSingleLayerBackprop = True	#revert to standard ANN algorithm (single layer training)

debugVerifyGradientBackpropStopSub = False


debugWexplosion = False
debugFastTrain = False
if(debugFastTrain):
	learningRate = 0.01
else:
	learningRate = 0.001


useBatch = True
if(useBatch):
	if(debugFastTrain):
		batchSize = 1000
	else:
		batchSize = 10	#100
else:
	batchSize = 1	

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
Atrace = {}

#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0
datasetNumClasses = 0

#randomNormal = tf.initializers.RandomNormal()

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learningRate)
	
def getNoisySampleGenerationNumSamples():
	return noisySampleGeneration, noisySampleGenerationNumSamples, noiseStandardDeviation
	
def defineTrainingParameters(dataset):
	
	if(debugFastTrain):
		trainingSteps = 1000
	else:
		trainingSteps = 10000
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
			
			Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")] = tf.Variable(tf.zeros(n_h[l], dtype=tf.dtypes.float32))

	
def neuralNetworkPropagation(x, networkIndex=1, recordAtrace=False):
	return neuralNetworkPropagationLREANN(x, networkIndex, recordAtrace)
	
def neuralNetworkPropagationLREANN(x, networkIndex=1, recordAtrace=False):
	pred = neuralNetworkPropagationLREANNlayer(x, lTrain=numberOfLayers, networkIndex=networkIndex)
	return pred
	
def neuralNetworkPropagationLREANNlayer(x, lTrain, networkIndex=1, recordAtrace=False):
	
	global averageTotalInput
		
	AprevLayer = x
	#Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")] = x

	if(useBinaryWeights):
		if(averageTotalInput == -1):
			averageTotalInput = tf.math.reduce_mean(x)	#CHECKTHIS: why was disabled?
			print("averageTotalInput = ", averageTotalInput)	 
	#print("x = ", x)
	
	for l in range(1, lTrain+1):
	
		#print("l = " + str(l))
		#print("W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
		
		if(useBinaryWeights):
			if(useBinaryWeightsReduceMemoryWithBool):
				Wfloat = tf.dtypes.cast(W[generateParameterNameNetwork(networkIndex, l, "W")], dtype=tf.float32)
				Bfloat = tf.dtypes.cast(B[generateParameterNameNetwork(networkIndex, l, "B")], dtype=tf.float32)
				#Z = tf.add(tf.matmul(Atrace[generateParameterNameNetwork(networkIndex, l-1, "Atrace")], Wfloat), Bfloat)
				Z = tf.add(tf.matmul(AprevLayer, Wfloat), Bfloat)
			else:
				#Z = tf.add(tf.matmul(Atrace[generateParameterNameNetwork(networkIndex, l-1, "Atrace")], W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
				Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
			Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")] = activationFunction(Z, n_h[l-1])
		else:
			#Z = tf.add(tf.matmul(Atrace[generateParameterNameNetwork(networkIndex, l-1, "Atrace")], W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
			Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
			Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")] = activationFunction(Z)

		if(l < lTrain):
			#A = tf.stop_gradient(A)
			AprevLayer = Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")]
		
	if(debugActivationFunctionAllSoftmax):
		if(l == numberOfLayers):
			pred = tf.nn.softmax(Z)
		else:
			pred = Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")]
	else:
		pred = Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")]
	
	#print("neuralNetworkPropagationLREANN pred.shape = ", pred.shape)	

	return pred


def neuralNetworkPropagationLREANN_test(x, y, networkIndex=1):

	pred = neuralNetworkPropagationLREANN(x, networkIndex)
	costCrossEntropyWithLogits = False	#binary classification
	loss = ANNtf2_operations.calculateLossCrossEntropy(pred, y, datasetNumClasses, costCrossEntropyWithLogits=costCrossEntropyWithLogits)
	acc = ANNtf2_operations.calculateAccuracy(pred, y)
	
	return loss, acc
	

def neuralNetworkPropagationLREANN_expMUANNtrain(x, y, networkIndex=1):

	if(trainUsingSingleLayerBackprop):
		neuralNetworkPropagationLREANN_expMUANNtrainBackprop(x, y, networkIndex)
	else:
		neuralNetworkPropagationLREANN_expMUANNtrainBio(x, y, networkIndex)

def neuralNetworkPropagationLREANN_expMUANNtrainBackprop(x, y, networkIndex=1):

	yLayer = tf.one_hot(y, depth=datasetNumClasses)
	for l in reversed(range(1, numberOfLayers+1)):
		if(debugVerifyGradientBackpropStopSub):
			print("l = ", l)
		#print("l = ", l)
		yLayer = executeOptimisationSub(x, yLayer, l, networkIndex)
		
		
def lossFunction(y_pred, y_true, lTrain):
	if(debugActivationFunctionAllSoftmax):
		loss = calculateLossCrossEntropy(y_pred, y_true, n_h[lTrain], costCrossEntropyWithLogits=False, oneHotEncoded=True)
	else:
		#loss = lossFunctionCustom(y_pred, y_true)
		loss = calculateLossCrossEntropy(y_pred, y_true, n_h[lTrain], costCrossEntropyWithLogits=False, oneHotEncoded=True)
	print("y_pred = ", y_pred)
	print("y_true = ", y_true)
	#print("loss = ", loss)
	return loss
	
def lossFunctionCustom(y_pred, y_true):
	loss = -tf.reduce_mean(tf.subtract(y_true, y_pred))
	return loss
			
def executeOptimisationSub(x, yLayer, lTrain, networkIndex=1):

	with tf.GradientTape() as g:
		pred = neuralNetworkPropagationLREANNlayer(x, lTrain, networkIndex, recordAtrace=True)
		loss = lossFunction(pred, yLayer, lTrain)

	if(debugVerifyGradientBackpropStopSub):
		for l in range(1, numberOfLayers+1):
			print("executeOptimisationSub before: l = ", l, ", W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
					
	Wlist = []
	Blist = []
	Wlist.append(W[generateParameterNameNetwork(networkIndex, lTrain, "W")])
	Blist.append(B[generateParameterNameNetwork(networkIndex, lTrain, "B")])
	if(lTrain > 1):
		Aprev = Atrace[generateParameterNameNetwork(networkIndex, lTrain-1, "Atrace")]
		#print("Aprev = ", Aprev)
		
		Alist = []
		Alist.append(Atrace[generateParameterNameNetwork(networkIndex, lTrain-1, "Atrace")])
		trainableVariablesAll = Wlist + Blist + Alist
	else:
		trainableVariablesAll = Wlist + Blist

	trainableVariables = Wlist + Blist	
	gradientsAll = g.gradient(loss, trainableVariablesAll)
	#print("gradientsAll = ", gradientsAll)
	gradients = [gradientsAll[0], gradientsAll[1]]
	optimizer.apply_gradients(zip(gradients, trainableVariables))

	if(debugVerifyGradientBackpropStopSub):
		for l in range(1, numberOfLayers+1):
			print("executeOptimisationSub after: l = ", l, ", W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
	
	if(lTrain > 1):
		gradientsAprev = gradientsAll[2]
		#print("gradientsA = ", gradientsA)
		AprevIdeal = tf.add(Atrace[generateParameterNameNetwork(networkIndex, lTrain-1, "Atrace")], gradientsAprev)
		#print("AprevIdeal = ", AprevIdeal)
	else:
		AprevIdeal = None
		
	#print("AprevIdeal = ", AprevIdeal)
	
	return AprevIdeal
		
def neuralNetworkPropagationLREANN_expMUANNtrainBio(x, y, networkIndex=1):
	
	#TODO: needs to pass back through relu
	
	#print("neuralNetworkPropagationLREANN_expMUANNtrain")
	
	neuralNetworkPropagationLREANN(x, networkIndex=networkIndex, recordAtrace=True)

	#perform highest layer weight updates
	pred, ATop = neuralNetworkPropagationLREANNlayer(x, numberOfLayers, networkIndex)
	AexpectedTop = tf.one_hot(y, depth=datasetNumClasses)	#see ANNtf2_operations.calculateLossCrossEntropy;
	ADeltaTop = calculateAdelta(AexpectedTop, ATop)
	ADeltaTop = tf.reduce_mean(ADeltaTop, axis=0)	#average over entire batch
	
	#print("ATop.shape = ", ATop.shape)
	#print("AexpectedTop.shape = ", AexpectedTop.shape)
	WTop = W[generateParameterNameNetwork(networkIndex, numberOfLayers, "W")]
	W[generateParameterNameNetwork(networkIndex, numberOfLayers, "W")] = calculateUpdatedWeights(ADeltaTop, WTop)	#updates dentrite synapse (ie postsynaptic) terminals of top layer
			
	for l in reversed(range(1, numberOfLayers)):
			
		#[calculate higher layer A delta,] apply potentiation (eg STP/LTP) to post synaptic terminals of each neuron based on its (neurotransmitter equilibrium) delta, then adjust its dendritic input weights [protensity of firing]

		pred, AhigherNew = neuralNetworkPropagationLREANNlayer(x, l+1, networkIndex, recordAtrace=False)
		AtraceHigher = Atrace[generateParameterNameNetwork(networkIndex, l+1, "Atrace")]
		AhigherDelta = calculateAdelta(AhigherNew, AtraceHigher)
		AhigherDelta = tf.reduce_mean(AhigherDelta, axis=0)	#average over entire batch
		AhigherDelta = tf.expand_dims(AhigherDelta, axis=1)		#prepare for matrix multiplication
		Whigher = W[generateParameterNameNetwork(networkIndex, l+1, "W")]
		Adelta = tf.matmul(Whigher, AhigherDelta)
		Wcurrent = W[generateParameterNameNetwork(networkIndex, l, "W")]
		W[generateParameterNameNetwork(networkIndex, l, "W")] = calculateUpdatedWeights(Adelta, Wcurrent)	#updates dentrite synapse (ie postsynaptic) terminals of layer
		
def calculateUpdatedWeights(Adelta, Worig):

	#overview: chemical syapses are used to infer neurotransmitter disequilibrium in synaptic cleft
	#if there is a disequilbrium between presynaptic neurotransmitter emission/uptake synaptic and post synaptic neurotransmitter uptake, there will be either an excess or scarcity of neurotransmitter in the synaptic cleft
	#this signals to the lower level neuron (combined neurotransmitter / presynaptic neurotransmission) that dendritic modulation is required
		#possibly by modifying the probability of presynaptic action potential, and therefore the frequency of backpropagated signals to its own post synaptic terminals
	#Adelta = synaptic neurotransmitter disequlibrium detected (excess or lack)
	
	Adelta = tf.squeeze(Adelta)	
	#Adelta = tf.expand_dims(Adelta, axis=0)	
	#Adelta = tf.transpose(Adelta)
	Adelta = Adelta*learningRate
	#print("Adelta = ", Adelta)
	Wnew = tf.add(Worig, Adelta)	#CHECKTHIS: broadcast required
	
	return Wnew

def calculateAdelta(AhigherNew, AtraceHigher):
	
	AhigherDelta = AhigherNew - AtraceHigher
	return AhigherDelta

	

def activationFunction(Z, prevLayerSize=None):
	if(debugActivationFunctionAllSoftmax):
		A = tf.nn.softmax(Z)
	else:
		A = reluCustom(Z, prevLayerSize)
	return A
	
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

 

		
