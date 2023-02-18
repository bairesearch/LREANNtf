"""LREANNtf_algorithmLREANN_expXUANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LREANNtf_main.py

# Usage:
see LREANNtf_main.py

# Description:
LREANNtf algorithm LREANN expXUANN - define learning rule experiment artificial neural network with - contrastive (pos/neg sample diff) update

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *
import ANNtf2_operations
import ANNtf2_globalDefs
import math
from numpy import random

debugOnlyTrainFinalLayer = False	#emulate learning performance of single layer perceptron

debugWexplosion = False
debugFastTrain = False
if(debugFastTrain):
	learningRate = 0.001
else:
	learningRate = 0.0001	#0.00001
debugVerifyGradientBackpropStopSub = False
debugVerifyGradientBackpropStopFinalLayer = False


objectiveTargetMinimiseDiffBetweenPositiveSamples = True

useBatch = True
if(useBatch):
	if(debugFastTrain):
		batchSize = 100
	else:
		batchSize = 10
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
#Atrace = [{}]
numberOfTraces = 3	#current, positive sample, negative sample
traceIndexCurrent = 0
traceIndexPositiveSample = 1
traceIndexNegativeSample = 2


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
		trainingSteps = 1000 #batchSize
	else:
		trainingSteps = 10000 #100000
	if(useBatch):
		numEpochs = 50	#10
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
			
			#for t in range(numberOfTraces):
			#	Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")][t] = tf.Variable(tf.zeros(n_h[l], dtype=tf.dtypes.float32))

def neuralNetworkPropagation(x, networkIndex=1, enableFinalLayerWeightUpdatesOnly=False):
	return neuralNetworkPropagationLREANN(x, networkIndex, enableFinalLayerWeightUpdatesOnly)

def neuralNetworkPropagationLREANNfinal(x, networkIndex=1):
	return neuralNetworkPropagationLREANN(x, networkIndex, enableFinalLayerWeightUpdatesOnly=True)

def neuralNetworkPropagationLREANN(x, networkIndex=1, enableFinalLayerWeightUpdatesOnly=False):
			
	global averageTotalInput
	
	AprevLayer = x

	if(useBinaryWeights):
		if(averageTotalInput == -1):
			averageTotalInput = tf.math.reduce_mean(x)	#CHECKTHIS: why was disabled?
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

		if(enableFinalLayerWeightUpdatesOnly):
			if(l < numberOfLayers):
				A = tf.stop_gradient(A)
		
		AprevLayer = A
		
	pred = tf.nn.softmax(Z)
	
	#print("neuralNetworkPropagationLREANN pred.shape = ", pred.shape)	

	return pred

def neuralNetworkPropagationLREANNsub(x, samplePositiveX, sampleNegativeX, lTrain, networkIndex=1):

	global averageTotalInput
	
	AprevLayer = [x, samplePositiveX, sampleNegativeX]

	if(useBinaryWeights):
		if(averageTotalInput == -1):
			averageTotalInput = tf.math.reduce_mean(x)	#CHECKTHIS: why was disabled? 
			print("averageTotalInput = ", averageTotalInput)
	
	#print("x.shape = ", x.shape)
	#print("samplePositiveX.shape = ", samplePositiveX.shape)
	#print("sampleNegativeX.shape = ", sampleNegativeX.shape)
	
	for l in range(1, lTrain+1):

		#print("l = " + str(l))
		
		#propagate each activation trace independently (x, samplePositiveX, sampleNegativeX);
		for t in range(numberOfTraces):
	
			#print("t = " + str(t))

			if(useBinaryWeights):
				if(useBinaryWeightsReduceMemoryWithBool):
					Wfloat = tf.dtypes.cast(W[generateParameterNameNetwork(networkIndex, l, "W")], dtype=tf.float32)
					Bfloat = tf.dtypes.cast(B[generateParameterNameNetwork(networkIndex, l, "B")], dtype=tf.float32)
					Z = tf.add(tf.matmul(AprevLayer[t], Wfloat), Bfloat)
				else:
					Z = tf.add(tf.matmul(AprevLayer[t], W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
				A = activationFunction(Z, n_h[l-1])
			else:
				Z = tf.add(tf.matmul(AprevLayer[t], W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
				A = activationFunction(Z)
	
			if(l < lTrain):
				A = tf.stop_gradient(A)
					
			AprevLayer[t] = A
							
		if(l == lTrain):
			#add additional intermediary artificial output layer (for contrastive boolean objective function)
			
			#perform activation contrast
			#print("batchSize = ", batchSize)
			#print("AprevLayer[traceIndexCurrent].shape = ", AprevLayer[traceIndexCurrent].shape)
			positiveDiff = tf.abs(tf.subtract(AprevLayer[traceIndexCurrent], AprevLayer[traceIndexPositiveSample]))
			negativeDiff = tf.abs(tf.subtract(AprevLayer[traceIndexCurrent], AprevLayer[traceIndexNegativeSample]))
			#print("negativeDiff.shape = ", negativeDiff.shape)
			positiveDiffavg = tf.math.reduce_mean(positiveDiff, axis=1)
			negativeDiffavg = tf.math.reduce_mean(negativeDiff, axis=1)
			#print("positiveDiffavg.shape = ", positiveDiffavg.shape)
			ZlastLayer = tf.concat([positiveDiffavg, negativeDiffavg], 0)
			#print("ZlastLayer.shape = ", ZlastLayer.shape)
			
			pred = tf.nn.softmax(ZlastLayer)
			#pred = tf.nn.sigmoid(ZlastLayer)	#binary classification	
			
			#print("neuralNetworkPropagationLREANNsub: pred.shape = ", pred.shape)	

	#print("neuralNetworkPropagationLREANN pred.shape = ", pred.shape)
	
	return pred
	
		


def neuralNetworkPropagationLREANN_test(x, y, networkIndex=1):

	pred = neuralNetworkPropagationLREANN(x, networkIndex)
	costCrossEntropyWithLogits = False	#binary classification
	loss = ANNtf2_operations.calculateLossCrossEntropy(pred, y, datasetNumClasses, costCrossEntropyWithLogits=costCrossEntropyWithLogits)
	acc = ANNtf2_operations.calculateAccuracy(pred, y)
	
	return loss, acc
	
	
def neuralNetworkPropagationLREANN_expXUANNtrain(x, y, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, networkIndex=1):
	if not debugOnlyTrainFinalLayer:
		for l in range(1, numberOfLayers):
			if(debugVerifyGradientBackpropStopSub):
				print("l = ", l)
			executeOptimisationSub(x, y, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, l, networkIndex)
	executeOptimisationFinal(x, y, networkIndex)

def executeOptimisationSub(x, y, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, lTrain, networkIndex=1):

	if(objectiveTargetMinimiseDiffBetweenPositiveSamples):
		yIntermediaryArtificialTargetPart1 = tf.zeros(samplePositiveY.shape, dtype=tf.dtypes.float32)
		yIntermediaryArtificialTargetPart2 = tf.ones(sampleNegativeY.shape, dtype=tf.dtypes.float32)
	else:
		yIntermediaryArtificialTargetPart1 = tf.ones(samplePositiveY.shape, dtype=tf.dtypes.float32)
		yIntermediaryArtificialTargetPart2 = tf.zeros(sampleNegativeY.shape, dtype=tf.dtypes.float32)	
	yIntermediaryArtificialTarget = tf.concat([yIntermediaryArtificialTargetPart1, yIntermediaryArtificialTargetPart2], 0)
	yIntermediaryArtificialTargetNumClasses = y.shape[0]*2	#batchSize*2
	
	#debug sanity checks:
	#print("executeOptimisation: yIntermediaryArtificialTarget.shape = ", yIntermediaryArtificialTarget.shape)
	#print("executeOptimisation: batchSize*2 = ", batchSize*2)
	
	with tf.GradientTape() as g:
		pred = neuralNetworkPropagationLREANNsub(x, samplePositiveX, sampleNegativeX, lTrain, networkIndex)
		#print("pred.shape = ", pred.shape)
		#print("yIntermediaryArtificialTarget.shape = ", yIntermediaryArtificialTarget.shape)
		loss = calculateLossCrossEntropy(pred, yIntermediaryArtificialTarget, yIntermediaryArtificialTargetNumClasses, costCrossEntropyWithLogits=False, oneHotEncoded=True)	#single intermediary (per layer) output neuron used for training

	if(debugVerifyGradientBackpropStopSub):
		for l in range(1, numberOfLayers+1):
			print("executeOptimisationSub before: l = ", l, ", W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
		
	Wlist = []
	Blist = []
	#for l in range(1, lTrain+1):
	l = lTrain
	Wlist.append(W[generateParameterNameNetwork(networkIndex, l, "W")])
	Blist.append(B[generateParameterNameNetwork(networkIndex, l, "B")])
	
	trainableVariables = Wlist + Blist
	
	gradients = g.gradient(loss, trainableVariables)
	optimizer.apply_gradients(zip(gradients, trainableVariables))

	if(debugVerifyGradientBackpropStopSub):
		for l in range(1, numberOfLayers+1):
			print("executeOptimisationSub after: l = ", l, ", W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])

	
def executeOptimisationFinal(x, y, networkIndex=1):

	with tf.GradientTape() as g:
		pred = neuralNetworkPropagationLREANNfinal(x, networkIndex)
		loss = calculateLossCrossEntropy(pred, y, datasetNumClasses, costCrossEntropyWithLogits=False)
		
	Wlist = []
	Blist = []
	#for l in range(1, numberOfLayers+1):
	l = numberOfLayers
	
	if(debugVerifyGradientBackpropStopFinalLayer):
		for l in range(1, numberOfLayers+1):
			print("executeOptimisationFinal before: l = ", l, ", W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
		
	Wlist.append(W[generateParameterNameNetwork(networkIndex, l, "W")])
	Blist.append(B[generateParameterNameNetwork(networkIndex, l, "B")])
	
	trainableVariables = Wlist + Blist
	
	gradients = g.gradient(loss, trainableVariables)
	optimizer.apply_gradients(zip(gradients, trainableVariables))

	if(debugVerifyGradientBackpropStopFinalLayer):
		for l in range(1, numberOfLayers+1):
			print("executeOptimisationFinal after: l = ", l, ", W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
			
	

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

 


def generateTFtrainDataFromNParraysLREANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses, generatePositiveSamples):

	trainDataList = []
	
	for classTarget in range(datasetNumClasses):
		
		if(generatePositiveSamples):
			train_xClassFiltered, train_yClassFiltered = filterNParraysByClassTarget(train_x, train_y, classTargetFilterIndex=classTarget)		
		else:
			#generateNegativeSamples
			train_xClassFiltered, train_yClassFiltered = filterNParraysByClassTargetInverse(train_x, train_y, classTargetFilterIndex=classTarget)
		trainDataUnbatched = generateTFtrainDataUnbatchedFromNParrays(train_xClassFiltered, train_yClassFiltered)
		trainData = generateTFtrainDataFromTrainDataUnbatched(trainDataUnbatched, shuffleSize, batchSize)
		trainDataList.append(trainData)
		
	return trainDataList
		
		
