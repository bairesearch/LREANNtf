"""LREANNtf_algorithmLREANN_expHUANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LREANNtf_main.py

# Usage:
see LREANNtf_main.py

# Description:
LREANNtf algorithm LREANN expHUANN - define learning rule experiment artificial neural network with hebbian update

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs
import math

useZAcoincidenceMatrix = True	#else use AAcoincidenceMatrix
 
LREANN_expHUANN2 = True
if(LREANN_expHUANN2):
	immutableConnections = True	#a fraction of connection weights (e.g. 1) is immutable, the others are used in its prediction
	sparseConnections = False	#a fraction of connections are active, the others are disabled
	positiveConnections = True	#all connections are positive
	if(immutableConnections):
		immutableConnectionsFraction = 0.1	#0.1	#only ~1 connection will be immutable
		#FUTURE: guarantee that at least one incoming connection per neuron is immutable during initialisation
		immutableConnectionsInitialise = True	#initialise immutableConnectionsInitialiseWeight (high)
		mutableConnectionsInitialise = True	#apply mutableConnectionsInitialiseWeightMax (low)
		mutableConnectionsWeightMaxRestrict = True	#restrict max weight of mutable connections
		if(immutableConnectionsInitialise):
			immutableConnectionsInitialiseWeight = 0.5	#high
		if(mutableConnectionsInitialise):
			mutableConnectionsInitialiseWeightMax = 0.1	#low
		if(mutableConnectionsWeightMaxRestrict):
			mutableConnectionsWeightMax = immutableConnectionsInitialiseWeight
	if(sparseConnections):
		activeConnectionsFraction = 0.5	#0.1	#only x connections will be active (connected)	#connectivity sparsity
else:
	immutableConnections = False
	sparseConnections = False
	positiveConnections = False

	
debugHebbianForwardPropOnlyTrainFinalSupervisedLayer = False
applyWmaxCap = True	#max W = 1
applyAmaxCap = True	#max A = 1
if(useZAcoincidenceMatrix):
	enableForgetting = False	#ZAcoincidence matrix naturally provides negative Wmod updates
else:
	enableForgetting = True
#debugSparseActivatedNetwork = False	#creates much larger network

if(immutableConnections):
	onlyTrainNeuronsIfLayerActivationIsSparse = False
else:
	onlyTrainNeuronsIfLayerActivationIsSparse = True	#only train upper layer [neuron] if layer activation is sparse - ie if only a single hypothesis is detected as true
	if(onlyTrainNeuronsIfLayerActivationIsSparse):
		onlyTrainNeuronsIfLayerActivationIsSparseRequireUniqueNeuronActivation = True
		if(not onlyTrainNeuronsIfLayerActivationIsSparseRequireUniqueNeuronActivation):
			onlyTrainNeuronsIfLayerActivationIsSparseMinSparsity = 0.9	#ie only 10% of neurons can be activation for training to occur

if(onlyTrainNeuronsIfLayerActivationIsSparse):		
	generateFirstLayerSDR = True	#required	#approximates k winners takes all	
else:
	generateFirstLayerSDR = False	#optional	#approximates k winners takes all	
	
if(generateFirstLayerSDR):
	maximumNetworkHiddenLayerNeuronsAsFractionOfInputNeurons = 10.0	#100.0
else:
	maximumNetworkHiddenLayerNeuronsAsFractionOfInputNeurons = 0.8	#0.8

if(generateFirstLayerSDR):
	ignoreFirstXlayersTraining = True	#this can be used to significantly increase the network activation sparsity	#required for onlyTrainNeuronsIfLayerActivationIsSparse
	if(ignoreFirstXlayersTraining):
		ignoreFirstXlayersTrainingX = 1
else:
	ignoreFirstXlayersTraining = False
		
if(positiveConnections):
	applyNeuronThresholdBias = True
	if(applyNeuronThresholdBias):
		applyNeuronThresholdBiasValue = 0.5
		applyNeuronThresholdBiasDuringTrainOnly = False
else:
	applyNeuronThresholdBias = False	#this can be used to significantly increase the network activation sparsity
	if(applyNeuronThresholdBias):
		applyNeuronThresholdBiasValue = 0.1
		applyNeuronThresholdBiasDuringTrainOnly = True
	
onlyTrainNeuronsIfActivationContributionAboveThreshold = False	#theshold neurons which will be positively biased, and those which will be negatively (above a = 0 as it is currently) 
if(onlyTrainNeuronsIfActivationContributionAboveThreshold):
	onlyTrainNeuronsIfActivationContributionAboveThresholdValue = 0.1
	backpropCustomOnlyUpdateWeightsThatContributedTowardsTarget = True	#as not every neuron which fires contributes to the learning in fully connected network
		#requires trainHebbianBackprop


if(enableForgetting):
	if(immutableConnections):
		enableForgettingRestrictToAPrevAndNotAConnections = False
		enableForgettingRestrictToNotAPrevAndAConnections = True	#CHECKTHIS
	else:
		if(onlyTrainNeuronsIfLayerActivationIsSparse):
			enableForgettingRestrictToAPrevAndNotAConnections = False	#required
			enableForgettingRestrictToNotAPrevAndAConnections = True	#required
		else:
			enableForgettingRestrictToAPrevAndNotAConnections = True	#optional	#True	#this ensures that only connections between active lower layer neurons and unactive higher layer neurons are suppressed
			enableForgettingRestrictToNotAPrevAndAConnections = False 	#required

normaliseFirstLayer = False
equaliseNumberExamplesPerClass = False


W = {}
B = {}
if(sparseConnections):
	Wactive = {}	#tf.dtypes.bool
if(immutableConnections):
	Wmutable = {}	#tf.dtypes.bool
	Wimmutable = {}	#tf.dtypes.bool

#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0

learningRate = 0.0
forgetRate = 0.0
batchSize = 0


def getNoisySampleGenerationNumSamples():
	return False, 0, 0
	
def defineTrainingParameters(dataset):

	global learningRate
	global forgetRate
	global batchSize
	
	learningRate = 0.001
	forgetRate = 0.001
	trainingSteps = 1000
	batchSize = 10		#1	#10	#100	#1000	#temporarily reduce batch size for visual debugging (array length) purposes)
	numEpochs = 100
	
	displayStep = 100
				
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	

def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, generateLargeNetwork=True, generateNetworkStatic=True)

	return numberOfLayers


def defineNeuralNetworkParameters():

	randomNormal = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
	
		for l in range(1, numberOfLayers+1):

			W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(randomNormal([n_h[l-1], n_h[l]]))
			B[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(tf.zeros(n_h[l]))
			
			if(positiveConnections):
				W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.abs(W[generateParameterNameNetwork(networkIndex, l, "W")])
			
			if(sparseConnections):
				WactiveFloat = tf.random.uniform([n_h[l-1], n_h[l]], dtype=tf.dtypes.float32)
				WactiveBool = tf.less(WactiveFloat, activeConnectionsFraction)
				Wactive[generateParameterNameNetwork(networkIndex, l, "Wactive")] = WactiveBool
				WactiveFloat = tf.dtypes.cast(WactiveBool, tf.float32)  
				W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.multiply(W[generateParameterNameNetwork(networkIndex, l, "W")], WactiveFloat)
				#print("WactiveBool = ", WactiveBool)
				#print(W[generateParameterNameNetwork(networkIndex, l, "W")])
			
			if(immutableConnections):
				WmutableFloat = tf.random.uniform([n_h[l-1], n_h[l]], dtype=tf.dtypes.float32)
				WmutableBool = tf.greater(WmutableFloat, immutableConnectionsFraction)
				Wmutable[generateParameterNameNetwork(networkIndex, l, "Wmutable")]	= WmutableBool
				WmutableFloat = tf.dtypes.cast(WmutableBool, tf.float32)  
				WimmutableBool = tf.math.logical_not(WmutableBool) 
				Wimmutable[generateParameterNameNetwork(networkIndex, l, "Wimmutable")]	= WimmutableBool
				WimmutableFloat = tf.dtypes.cast(WimmutableBool, tf.float32) 
				#print("WmutableBool = ", WmutableBool)
				if(immutableConnectionsInitialise):
					Wfloat = W[generateParameterNameNetwork(networkIndex, l, "W")]
					Wfloat = tf.multiply(Wfloat, WmutableFloat)	#zero all immutable weights
					WimmutableComponentsFloat = tf.multiply(WimmutableFloat, immutableConnectionsInitialiseWeight)
					Wfloat = tf.add(Wfloat, WimmutableComponentsFloat)
					W[generateParameterNameNetwork(networkIndex, l, "W")] = Wfloat
					#print("W = ", Wfloat)
				if(mutableConnectionsInitialise):
					Wfloat = W[generateParameterNameNetwork(networkIndex, l, "W")]
					WmutableComponentsFloat = tf.multiply(WmutableFloat, Wfloat)
					WmutableComponentsFloat = tf.minimum(WmutableComponentsFloat, mutableConnectionsInitialiseWeightMax)
					Wfloat = tf.multiply(Wfloat, WimmutableFloat)	#zero all mutable weights
					Wfloat = tf.add(Wfloat, WmutableComponentsFloat)
					W[generateParameterNameNetwork(networkIndex, l, "W")] = Wfloat
					#print("W = ", Wfloat)
					
				#print("W = ", Wfloat)
		

def neuralNetworkPropagation(x, networkIndex=1):
	return neuralNetworkPropagationLREANN(x, networkIndex)
	
def neuralNetworkPropagationLREANN(x, networkIndex=1):
			
	AprevLayer = x
	 
	for l in range(1, numberOfLayers+1):
	
		#print("l = " + str(l))

		Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
		A = activationFunction(Z, train=False)
			
		AprevLayer = A
			
	return tf.nn.softmax(Z)
	
	
def neuralNetworkPropagationLREANN_expHUANNtrain(x, y=None, networkIndex=1, trainHebbianForwardprop=False, trainHebbianBackprop=False, trainHebbianLastLayerSupervision=False):

	#print("batchSize = ", batchSize)
	#print("learningRate = ", learningRate)
	
	AprevLayer = x
	ZprevLayer = x

	Alayers = []
	Zlayers = []	#CHECKTHIS (not implemented)
	if(trainHebbianBackprop):
		Alayers.append(AprevLayer)
		Zlayers.append(ZprevLayer)
	
	#print("x = ", x)
	
	for l in range(1, numberOfLayers+1):
	
		#print("\nl = " + str(l))

		Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])	
		A = activationFunction(Z, train=True)
		
		if(trainHebbianBackprop):
			Alayers.append(A)
			Zlayers.append(Z)
					
		if(applyAmaxCap):
			A = tf.clip_by_value(A, clip_value_min=-1.0, clip_value_max=1.0)

		if(trainHebbianForwardprop):
			trainLayerLREANN_expHUANN(y, networkIndex, l, AprevLayer, ZprevLayer, A, Alayers, trainHebbianBackprop=trainHebbianBackprop, trainHebbianLastLayerSupervision=trainHebbianLastLayerSupervision)

		AprevLayer = A
		ZprevLayer = Z
	
	if(trainHebbianBackprop):
		for l in reversed(range(1, numberOfLayers+1)):
			
			#print("Alayers[l] = ", Alayers[l])
			
			AprevLayer = Alayers[l-1]
			A = Alayers[l]
			
			trainLayerLREANN_expHUANN(y, networkIndex, l, AprevLayer, ZprevLayer, A, Alayers, trainHebbianBackprop=trainHebbianBackprop, trainHebbianLastLayerSupervision=trainHebbianLastLayerSupervision)
							
	return tf.nn.softmax(Z)


def trainLayerLREANN_expHUANN(y, networkIndex, l, AprevLayer, ZprevLayer, A, Alayers, trainHebbianBackprop=False, trainHebbianLastLayerSupervision=False):

		#print("train")
		isLastLayerSupervision = False
		if(trainHebbianLastLayerSupervision):
			if(l == numberOfLayers):
				isLastLayerSupervision = True
				#print("isLastLayerSupervision")

		trainLayer = True
		if(isLastLayerSupervision):
			#perform hebbian learning on last layer based on hypothetical correct one hot class activation (Ahypothetical)
			Alearn = y
		else:
			Alearn = A
			if(debugHebbianForwardPropOnlyTrainFinalSupervisedLayer):
				trainLayer = False
		if(ignoreFirstXlayersTraining):
			if(l <= ignoreFirstXlayersTrainingX):
				trainLayer = False

		if(trainLayer):
			#print("Alearn = ", Alearn)
					
			#update weights based on hebbian learning rule
			#strengthen those connections that caused the current layer neuron to fire (and weaken those that did not)

			AprevLayerLearn = AprevLayer
			ZprevLayerLearn = ZprevLayer
			if(onlyTrainNeuronsIfActivationContributionAboveThreshold):
				#apply threshold to AprevLayer
				AprevLayerAboveThreshold = tf.math.greater(AprevLayer, onlyTrainNeuronsIfActivationContributionAboveThresholdValue)
				AprevLayerAboveThresholdFloat = tf.dtypes.cast(AprevLayerAboveThreshold, dtype=tf.float32)
				AprevLayerLearn = AprevLayer*AprevLayerAboveThresholdFloat
			
			enableLearning = True
			if(onlyTrainNeuronsIfLayerActivationIsSparse):
				enableLearning = False
				#only train upper layer [neuron] if layer activation is sparse - ie if only a single hypothesis is detected as true
				#print(A.shape)
				numberHiddenLayerUnits = A.shape[1]
				AposThresholded = tf.math.greater(A, 0.0)
				numberHiddenLayerUnitsActive = tf.reduce_sum(tf.cast(AposThresholded, tf.float32), axis=1)
				#print("numberHiddenLayerUnitsActive = ", numberHiddenLayerUnitsActive)
				if(onlyTrainNeuronsIfLayerActivationIsSparseRequireUniqueNeuronActivation):
					batchIndexLearn = tf.math.equal(numberHiddenLayerUnitsActive, 1)
				else:
					percentageHiddenLayerUnitsActive = tf.divide(numberHiddenLayerUnitsActive, numberHiddenLayerUnits)
					batchIndexLearn = tf.math.less(percentageHiddenLayerUnitsActive, 1-onlyTrainNeuronsIfLayerActivationIsSparseMinSparsity)
				batchIndexLearnFloat = tf.cast(batchIndexLearn, tf.float32)
				batchIndexLearnFloat = tf.expand_dims(batchIndexLearnFloat, 1)
				#print("batchIndexLearn = ", batchIndexLearn)
				#print("Alearn.shape = ", Alearn.shape)
				#print("batchIndexLearnFloat.shape = ", batchIndexLearnFloat.shape)
				Alearn = tf.math.multiply(Alearn, batchIndexLearnFloat)	#only learn connections which result in an activated higher layer neuron
						
			if(useZAcoincidenceMatrix):
				AcoincidenceMatrix = tf.matmul(tf.transpose(ZprevLayerLearn), Alearn)	#ZAcoincidenceMatrix
			else:
				AcoincidenceMatrix = tf.matmul(tf.transpose(AprevLayerLearn), Alearn)
				#Bmod = 0*learningRate	#biases are not currently used
			
			Wmod = AcoincidenceMatrix/batchSize*learningRate

			if(immutableConnections):
				WmutableFloat = tf.dtypes.cast(Wmutable[generateParameterNameNetwork(networkIndex, l, "Wmutable")], tf.float32)  
				Wmod = tf.multiply(Wmod, WmutableFloat)
			if(sparseConnections):
				WactiveFloat = tf.dtypes.cast(Wactive[generateParameterNameNetwork(networkIndex, l, "Wactive")], tf.float32)  
				Wmod = tf.multiply(Wmod, WactiveFloat)

			#print("Wmod = ", Wmod)

			#B[generateParameterNameNetwork(networkIndex, l, "B")] = B[generateParameterNameNetwork(networkIndex, l, "B")] + Bmod
			W[generateParameterNameNetwork(networkIndex, l, "W")] = W[generateParameterNameNetwork(networkIndex, l, "W")] + Wmod

			#print("Alearn = ", Alearn)
			#print("AprevLayerLearn = ", AprevLayerLearn)
			#print("A = ", A)
			#print("Alearn = ", Alearn)
			#print("AcoincidenceMatrix = ", AcoincidenceMatrix)
			#print("Wmod = ", Wmod)
			#print("W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])

			if(enableForgetting):
				if(enableForgettingRestrictToNotAPrevAndAConnections):
					AprevboolNeg = tf.math.equal(AprevLayerLearn, 0.0)	#Abool = tf.math.greater(Alearn, 0.0), AboolNeg = tf.math.logical_not(Abool)
					#print("AprevboolNeg = ",AprevboolNeg)
					#AprevboolNegInt = tf.dtypes.cast(AprevboolNeg, tf.int32)
					AprevboolNegFloat = tf.dtypes.cast(AprevboolNeg, tf.float32)
					AcoincidenceMatrixForget = tf.matmul(tf.transpose(AprevboolNegFloat), Alearn)
					Wmod2 = tf.square(AcoincidenceMatrixForget)/batchSize*forgetRate	#tf.square(AcoincidenceMatrixForget) - square is required to normalise the forget rate relative to the learn rate [assumes input tensor is < 1]
					#print("Wmod2 = ", Wmod2)
				else:
					if(enableForgettingRestrictToAPrevAndNotAConnections):
						AboolNeg = tf.math.equal(Alearn, 0.0)	#Abool = tf.math.greater(Alearn, 0.0), AboolNeg = tf.math.logical_not(Abool)
						#print("Abool = ",Abool)
						#AboolNegInt = tf.dtypes.cast(AboolNeg, tf.int32)
						AboolNegFloat = tf.dtypes.cast(AboolNeg, tf.float32)
						AcoincidenceMatrixForget = tf.matmul(tf.transpose(AprevLayerLearn), AboolNegFloat)
						Wmod2 = tf.square(AcoincidenceMatrixForget)/batchSize*forgetRate	#tf.square(AcoincidenceMatrixForget) - square is required to normalise the forget rate relative to the learn rate [assumes input tensor is < 1]
						#print("Wmod2 = ", Wmod2)
					else:
						AcoincidenceMatrixIsZero = tf.math.equal(AcoincidenceMatrix, 0)
						#AcoincidenceMatrixIsZeroInt = tf.dtypes.cast(AcoincidenceMatrixIsZero, tf.int32)
						AcoincidenceMatrixIsZeroFloat = tf.dtypes.cast(AcoincidenceMatrixIsZero, dtype=tf.float32)
						Wmod2 = tf.square(AcoincidenceMatrixIsZeroFloat)/batchSize*forgetRate	#tf.square(AcoincidenceMatrixIsZeroFloat) - square is required to normalise the forget rate relative to the learn rate [assumes input tensor is < 1]
						#print("Wmod2 = ", Wmod2)

				if(immutableConnections):
					WmutableFloat = tf.dtypes.cast(Wmutable[generateParameterNameNetwork(networkIndex, l, "Wmutable")], tf.float32)  
					Wmod2 = tf.multiply(Wmod2, WmutableFloat)
				if(sparseConnections):
					WactiveFloat = tf.dtypes.cast(Wactive[generateParameterNameNetwork(networkIndex, l, "Wactive")], tf.float32)  
					Wmod2 = tf.multiply(Wmod2, WactiveFloat)

				#print("Wmod2 = ", Wmod2)

				W[generateParameterNameNetwork(networkIndex, l, "W")] = W[generateParameterNameNetwork(networkIndex, l, "W")] - Wmod2

			if(positiveConnections):
				Wfloat = W[generateParameterNameNetwork(networkIndex, l, "W")]
				Wfloat = tf.clip_by_value(Wfloat, 0.0, applyNeuronThresholdBiasValue*2.0)
				W[generateParameterNameNetwork(networkIndex, l, "W")] = Wfloat
			if(mutableConnectionsWeightMaxRestrict):
				WmutableFloat = tf.dtypes.cast(Wmutable[generateParameterNameNetwork(networkIndex, l, "Wmutable")], tf.float32)  
				WimmutableFloat = tf.dtypes.cast(Wimmutable[generateParameterNameNetwork(networkIndex, l, "Wimmutable")], tf.float32)  
				Wfloat = W[generateParameterNameNetwork(networkIndex, l, "W")]
				WmutableComponentsFloat = tf.multiply(WmutableFloat, Wfloat)
				WmutableComponentsFloat = tf.minimum(WmutableComponentsFloat, mutableConnectionsWeightMax)
				Wfloat = tf.multiply(Wfloat, WimmutableFloat)	#zero all mutable weights
				Wfloat = tf.add(Wfloat, WmutableComponentsFloat)
				W[generateParameterNameNetwork(networkIndex, l, "W")] = Wfloat
							
			#print("W after learning = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
				
			if(applyWmaxCap):
				#print("W before cap = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
				W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.clip_by_value(W[generateParameterNameNetwork(networkIndex, l, "W")], clip_value_min=-1.0, clip_value_max=1.0)
				#print("W after cap = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
				
			if(trainHebbianBackprop):
				if(backpropCustomOnlyUpdateWeightsThatContributedTowardsTarget):
					Alayers[l-1] = AprevLayerLearn	#deactivate AprevLayer during backprop based on threshold (to prevent non contributing activation paths to be learnt)
		
		

def activationFunction(Z, train):

	if(applyNeuronThresholdBias):
		applyBias = False
		if(applyNeuronThresholdBiasDuringTrainOnly):
			if(train):
				applyBias = True
		else:
			applyBias = True
		
		#Z = tf.clip_by_value(Z, min=applyNeuronThresholdBiasValue)	#clamp
		Z = Z - applyNeuronThresholdBiasValue
	
	A = tf.nn.relu(Z)
	
	return A

  
