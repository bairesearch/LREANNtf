"""LREANNtf_algorithmLREANN_expRUANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LREANNtf_main.py

# Usage:
see LREANNtf_main.py

# Description:
LREANNtf algorithm LREANN expRUANN - define learning rule experiment artificial neural network with relaxation update

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *
import ANNtf2_operations
import ANNtf2_globalDefs
import math
from numpy import random

#
# RUANN biological implementation requirements:
#
# backpropagation approximation notes:
# error_L = (y_L - A_L) [sign reversal]
# error_l = (W_l+1 * error_l+1) . activationFunctionPrime(z_l) {~A_l}
# dC/dB = error_l
# dC/dW = A_l-1 * error_l
# Bnew = B+dC/dB [sign reversal]
# Wnew = W+dC/dW [sign reversal]
#
# backpropagation error is stored in temporary firing rate modification [increase/decrease] of neurons (ie Atrace->Aideal)
# Aerror_l update is applied based on signal pass through (W), higher level temporary firing rate adjustment, and current firing rate. error_l = (W_l+1 * error_l+1) * A_l
# W_l update is applied based on firing rate of lower layer and higher level temporary firing rate adjustment. dC/dW = A_l-1 * error_l
#
# RUANN approximates backpropagation for constrained/biological assumptions
# Error calculations are achieved by repropagating signal through neuron and measuring either a) temporary modulation in output (Aideal) relative to original (Atrace), or b) output of a specific error storage neurotransmitter receptor
# can rely on sustained burst/spike of ideal A values to perform weight updates
#
# Outstanding Biological Requirement: Need to identify a method to pass (single layer) error signal back through neuron from tip of axon to base of dendrite (internal/external signal?)
#	the original RUANN (learningAlgorithm == "backpropApproximation3/backpropApproximation4") attempts to achieve this by sending a trial +/- signal from the lower layer l neuron k and slowly ramping it up/down (increasing/decreasing its effective error) until the above layer l+1 neurons reach their ideal values/errors  
#

debugOnlyTrainFinalLayer = False	#debug weight update method only (not Aideal calculation method)	#requires recalculateAtraceUnoptimisedBio==False
debugVerboseOutput = False
debugVerboseOutputTrain = False

averageAerrorAcrossBatch = False	#RUANN was originally implemented to calculate independent idealA for each batch index (rather than averaged across batch)

errorImplementationAlgorithm = "storeErrorAsModulationOfSignalPropagationNeurotransmitterReceptor"	#original	#a) modulates primary propagation neurotransmitter receptor (+/-) to store l error, and for the calculation of l-1 error
#errorImplementationAlgorithm = "storeErrorAsModulationOfUniqueNeurotransmitterReceptor"	#b) designates a specific neurotransmitter receptor to store l error, and for the calculation of l-1 error

#learning algorithm variants in order of emulation similarity to formal backpropagation:
#learningAlgorithm = "backpropApproximation1"	#strict backpropagation (optional: use A ideal instead of A error, use activationFunctionTypeFinalLayer sigmoid rather than softmax)
#learningAlgorithm = "backpropApproximation2"	#incomplete	#modifies the A ideal (trials +ve and -ve adjustments; adjusting firing strength), and performs weight updates based on this modified A value
#learningAlgorithm = "backpropApproximation3"	#incomplete	#modifies the A ideal (trials +ve and -ve adjustments; adjusting firing strength), and performs weight updates based on this modified A value
#learningAlgorithm = "backpropApproximation4"	#incomplete	#modifies the A ideal (trials +ve and -ve adjustments; adjusting firing strength), and performs weight updates based on this modified A value	#original proposal	#emulates backpropagation using a variety of shortcuts (with optional thresholding), but does not emulate backPropagation completely - error_l (Aideal_l) calculations are missing *error_l+1 (multiply by the strength of the higher layer error)
#learningAlgorithm = "backpropApproximation5"	#incomplete	#modifies the A ideal (trials +ve and -ve adjustments; adjusting firing strength), and performs weight updates based on this modified A value	#simplifies RUANN algorithm to only consider +/- performance (not numerical/weighted performance) #probably only feasible with useBinaryWeights #note if useBinaryWeights then could more easily biologically predict the effect of adjusting Aideal of lower layer neuron k on performance of upper layer (perhaps without even trialling the adjustment)
#learningAlgorithm = "backpropApproximation6"	#incomplete	#calculates current layer neuron k A error based on final layer error of propagating signal
learningAlgorithm = "backpropApproximation7"	#calculates current layer A error/ideal based on above level WdeltaStore

#errorStorageAlgorithm = "useAerror"	#l+1 error is stored as a linear modulation of post synaptic receptor
#errorStorageAlgorithm = "useAideal" 	#original	#l+1 error is stored as a hypothetical difference between Atrace and Aideal [ratio]

if(learningAlgorithm == "backpropApproximation1"):
	#strict backpropagation is used for testing only	#no known biological implementation
	#requires recalculateAtraceUnoptimisedBio==False
	errorStorageAlgorithm = "useAideal"		#"useAerror"	#optional
elif(learningAlgorithm == "backpropApproximation2"):
	#requires recalculateAtraceUnoptimisedBio==False
	errorStorageAlgorithm = "useAideal"		#"useAerror"	#optional
elif(learningAlgorithm == "backpropApproximation3"):
	errorStorageAlgorithm = "useAideal"
elif(learningAlgorithm == "backpropApproximation4"):
	errorStorageAlgorithm = "useAideal"
	useWeightUpdateDirectionHeuristicBasedOnExcitatoryInhibitorySynapseType = True	#enables rapid weight updates, else use stochastic (test both +/-) weight upates
	useMultiplicationRatherThanAdditionOfDeltaValues = True	#CHECKTHIS #this ensures that Aideal/weight updates are normalised across their local layer (to minimise the probability an alternate class data propagation will be interferred with by the update)
elif(learningAlgorithm == "backpropApproximation5"):
	errorStorageAlgorithm = "useAideal"
	useWeightUpdateDirectionHeuristicBasedOnExcitatoryInhibitorySynapseType = True
	useMultiplicationRatherThanAdditionOfDeltaValues = False
elif(learningAlgorithm == "backpropApproximation6"):
	errorStorageAlgorithm = "useAerror"
elif(learningAlgorithm == "backpropApproximation7"):
	errorStorageAlgorithm = "useAideal"	#"useAerror"	#optional
	#averageAerrorAcrossBatch = True		#require inverse matrix multiplication operations (their batchSize dimension must equal 1)

activationFunctionType = "sigmoid"	#default
#activationFunctionType = "softmax"	#trial only
#activationFunctionType = "relu"	#not currently supported; a) cannot converge with relu function at final layer, b) requires loss function

applyFinalLayerLossFunction = False		#if False: normalise the error calculation across all layers, taking y_target as Aideal of top layer
if(learningAlgorithm == "backpropApproximation1"):
	activationFunctionTypeFinalLayer = "sigmoid"	#"softmax"	#optional
	applyFinalLayerLossFunction = False
elif(learningAlgorithm == "backpropApproximation2"):
	activationFunctionTypeFinalLayer = "sigmoid"	#"softmax"	#optional
	applyFinalLayerLossFunction = False
else:
	activationFunctionTypeFinalLayer = "sigmoid"	#default	#doesn't currently converge with final layer loss function calculated based on sigmoid
	applyFinalLayerLossFunction = False	
	
errorFunctionTypeDelta = True
errorFunctionTypeDeltaFinalLayer = True		#sigmoid/softmax has already been calculated [Aideal has been created for final layer] so can simply extrace delta error here 	#OLD: use sigmoid/softmax loss for final layer - consider using more simply delta loss here

updateOrder = "updateWeightsAfterAidealCalculations"	#method 1
#updateOrder = "updateWeightsDuringAidealCalculations"	#method 2
#updateOrder = "updateWeightsBeforeAidealCalculations"	#method 3
if(learningAlgorithm == "backpropApproximation6"):
	updateOrder = "updateWeightsDuringAidealCalculations"
if(learningAlgorithm == "backpropApproximation7"):
	updateOrder = "updateWeightsBeforeAidealCalculations"
	
	
#takeAprevLayerFromTraceDuringWeightUpdates = True	#mandatory for computational purposes (normalise across batches)
	#this parameter value should not be critical to RUANN algorithm (it is currently set based on availability of Aideal of lower layer - ie if it has been precalculated)
	#difference between Aideal and Atrace of lower layer should be so small takeAprevLayerFromTraceDuringWeightUpdates shouldn't matter

recalculateAtraceUnoptimisedBio = False


if(not applyFinalLayerLossFunction):
	topLayerIdealAstrict = True #top level learning target (idealA) == y, else learning target (idealA) == A + deltaA
	topLayerIdealAproximity = 0.01	#maximum learning rate (effective learning rate will be less than this)

	
if(learningAlgorithm == "backpropApproximation4"):

	useMultiplicationRatherThanAdditionOfDeltaValuesAideal = False
	useMultiplicationRatherThanAdditionOfDeltaValuesW = False
	
	if(useMultiplicationRatherThanAdditionOfDeltaValues):
		useMultiplicationRatherThanAdditionOfDeltaValuesAideal = True
		useMultiplicationRatherThanAdditionOfDeltaValuesW = True
	else:
		useMultiplicationRatherThanAdditionOfDeltaValuesAideal = False
		useMultiplicationRatherThanAdditionOfDeltaValuesW = False
	learningRateMinFraction = 0.1	#minimum learning rate can be set to always be above 0 (learningRateMinFraction = fraction of learning rate)
	
	applyMinimiumAdeltaContributionThreshold = False 	#only adjust Aideal_k of l based on Aideal of l+1 if it significantly improves Aideal of l+1, where k is neuron index of l
	if(applyMinimiumAdeltaContributionThreshold):
		minimiumAdeltaContributionThreshold = 0.1	#fraction relative to original performance difference
		#minimiumAdeltaContributionThreshold = 1.0	#this contribution threshold is normalised wrt number of neurons (k) on l+1. default=1.0: if a Aideal_k adjustment on l contributes less than what on average an Aideal_k adjustment must necessarily contribute to achieve Aideal on l+1, then do not adjust Aideal_k (leave same as A_k)

	applySubLayerIdealAmultiplierRequirement = True
	if(applySubLayerIdealAmultiplierRequirement):
		subLayerIdealAmultiplierRequirement = 1.5 #idealA of each neuron k on l will only be adjusted if its modification achieves at least xM performance increase for Aideal on l+1
		applySubLayerIdealAmultiplierCorrection = True	#optional: adjust learning neuron learning based on performance multiplier
	else:
		applySubLayerIdealAmultiplierCorrection = False

if(learningAlgorithm == "backpropApproximation5"):
	subLayerIdealAlearningRateBase = 0.001	#small number used to ensure (reduce probablity) that update does not affect nonlinearity of signal upwards
else:
	subLayerIdealAlearningRateBase = 0.01	#each neuron k on l will be adjusted only by this amount (modified by its multiplication effect on Aideal of l+1)

if(learningAlgorithm == "backpropApproximation6"):
	useMultiplicationRatherThanAdditionOfDeltaValuesAideal = False
	
debugWexplosion = False
debugFastTrain = False
if(debugFastTrain):
	learningRate = 0.01
else:
	learningRate = 0.001


useBatch = True

if(learningAlgorithm == "backpropApproximation7"):
	useBatch = False	#require inverse matrix multiplication operations (their batchSize dimension must equal 1)	#or use averageAerrorAcrossBatch instead
	
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

Wbackup = {}
Bbackup = {}

NETWORK_PARAM_INDEX_TYPE = 0
NETWORK_PARAM_INDEX_LAYER = 1
NETWORK_PARAM_INDEX_H_CURRENT_LAYER = 2
NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER = 3
NETWORK_PARAM_INDEX_VARIATION_DIRECTION = 4

if(not recalculateAtraceUnoptimisedBio):
	Atrace = {}
	Ztrace = {}
	
if(errorStorageAlgorithm == "useAideal"):
	Aideal = {}
elif(errorStorageAlgorithm == "useAerror"):
	Aerror = {}

if(learningAlgorithm == "backpropApproximation7"):
	WdeltaStore = {}

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
		numEpochs = 100	#10
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
	
			Wbackup[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(W[generateParameterNameNetwork(networkIndex, l, "W")])
			Bbackup[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(B[generateParameterNameNetwork(networkIndex, l, "B")])
		
			#Aerror = Aideal - Atrace
			if(errorStorageAlgorithm == "useAideal"):
				if(averageAerrorAcrossBatch):
					Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")] = tf.Variable(tf.zeros(n_h[l], dtype=tf.dtypes.float32))
				else:
					Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.dtypes.float32))			
			elif(errorStorageAlgorithm == "useAerror"):
				if(averageAerrorAcrossBatch):
					Aerror[generateParameterNameNetwork(networkIndex, l, "Aerror")] = tf.Variable(tf.zeros(n_h[l], dtype=tf.dtypes.float32))			
				else:
					Aerror[generateParameterNameNetwork(networkIndex, l, "Aerror")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.dtypes.float32))
					
			if(not recalculateAtraceUnoptimisedBio):
				Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.dtypes.float32))
				Ztrace[generateParameterNameNetwork(networkIndex, l, "Ztrace")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.dtypes.float32))
			
			if(learningAlgorithm == "backpropApproximation7"):
				WdeltaStore[generateParameterNameNetwork(networkIndex, l, "WdeltaStore")] = tf.Variable(tf.zeros([n_h[l-1], n_h[l]], dtype=tf.dtypes.float32))
			
	

def neuralNetworkPropagation(x, networkIndex=1, recordAtrace=False):
	return neuralNetworkPropagationLREANN(x, networkIndex, recordAtrace)

def neuralNetworkPropagationLREANN(x, networkIndex=1, recordAtrace=False):
	pred, A, Z = neuralNetworkPropagationLREANNlayer(x, lTrainMax=numberOfLayers, networkIndex=networkIndex)
	return pred

def neuralNetworkPropagationLREANNlayer(x, lTrainMax, networkIndex=1, recordAtrace=False):

	global averageTotalInput
	
	if(useBinaryWeights):
		if(averageTotalInput == -1):
			averageTotalInput = tf.math.reduce_mean(x)	#CHECKTHIS: why was disabled? 
			print("averageTotalInput = ", averageTotalInput)
			
	AprevLayer = x
	return neuralNetworkPropagationLREANNlayer(AprevLayer, lTrainMax, lTrainMin=1, networkIndex=networkIndex, recordAtrace=recordAtrace)
		
def neuralNetworkPropagationLREANNlayer(AprevLayer, lTrainMax, lTrainMin=1, networkIndex=1, recordAtrace=False):
				
	if(recordAtrace):
		Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")] = AprevLayer
	
	for l in range(lTrainMin, lTrainMax+1):	#NB lTrainMax=numberOfLayers = len(n_h)-1
	
		if(debugVerboseOutput):
			print("l = " + str(l))
			print("W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
		
		A, Z = neuralNetworkPropagationLREANNlayerL(AprevLayer, l, networkIndex)

		if(recordAtrace):
			Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")] = A
			Ztrace[generateParameterNameNetwork(networkIndex, l, "Ztrace")] = Z
		
		AprevLayer = A
		
	pred = tf.nn.softmax(Z)
		
	return pred, A, Z

def neuralNetworkPropagationLREANNlayerL(AprevLayer, l, networkIndex=1):

	if(useBinaryWeights):
		if(useBinaryWeightsReduceMemoryWithBool):
			Wfloat = tf.dtypes.cast(W[generateParameterNameNetwork(networkIndex, l, "W")], dtype=tf.float32)
			Bfloat = tf.dtypes.cast(B[generateParameterNameNetwork(networkIndex, l, "B")], dtype=tf.float32)
			Z = tf.add(tf.matmul(AprevLayer, Wfloat), Bfloat)
		else:
			Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
		A = activationFunction(Z, n_h[l-1])
	else:
		#print("l = ", l)
		#print("W.shape = ", W[generateParameterNameNetwork(networkIndex, l, "W")].shape)
		#print("B.shape = ", B[generateParameterNameNetwork(networkIndex, l, "B")].shape)
		Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
		A = activationFunction(Z)
	
	return A, Z
			
def neuralNetworkPropagationLREANNlayerLK(AprevLayer, k, l, networkIndex=1):

	AprevLayerK = AprevLayer[:, k]
	WlayerK = W[generateParameterNameNetwork(networkIndex, l, "W")][k,:]
	AprevLayerK = tf.expand_dims(AprevLayerK, axis=1)
	WlayerK = tf.expand_dims(WlayerK, axis=0)
		
	if(useBinaryWeights):
		if(useBinaryWeightsReduceMemoryWithBool):
			Wfloat = tf.dtypes.cast(WlayerK, dtype=tf.float32)
			Bfloat = tf.dtypes.cast(B[generateParameterNameNetwork(networkIndex, l, "B")], dtype=tf.float32)
			Z = tf.add(tf.matmul(AprevLayerK, Wfloat), Bfloat)
		else:
			Z = tf.add(tf.matmul(AprevLayerK, WlayerK), B[generateParameterNameNetwork(networkIndex, l, "B")])
		A = activationFunction(Z, n_h[l-1])
	else:
		Z = tf.add(tf.matmul(AprevLayerK, WlayerK), B[generateParameterNameNetwork(networkIndex, l, "B")])
		A = activationFunction(Z)
	
	return A, Z
	
			
def neuralNetworkPropagationLREANN_test(x, y, networkIndex=1):

	pred = neuralNetworkPropagationLREANN(x, networkIndex)
	loss = ANNtf2_operations.calculateLossCrossEntropy(pred, y, datasetNumClasses, costCrossEntropyWithLogits=False)
	acc = ANNtf2_operations.calculateAccuracy(pred, y)
	
	return loss, acc
	

def neuralNetworkPropagationLREANN_expRUANNtrain(x, y, networkIndex=1):
	
	#print("numberOfLayers = ", numberOfLayers)
	
	if(debugOnlyTrainFinalLayer):
		minLayerToTrain = numberOfLayers
	else:
		if(learningAlgorithm == "backpropApproximation6"):
			minLayerToTrain = 0
		else:
			minLayerToTrain = 1	#do not calculate Aideal for input layer as this is always set to x
	
	#1. initial propagation;
	y_true = tf.one_hot(y, depth=datasetNumClasses)
	pred, A, Z = neuralNetworkPropagationLREANNlayer(x, numberOfLayers, networkIndex, recordAtrace=(not recalculateAtraceUnoptimisedBio))
	
	#2./3. calculate Aideal / W updates;
	
	calculateAndSetAerrorTopLayerWrapper(A, pred, y_true, networkIndex)			
	calculateAndSetAerrorBottomLayer(x, minLayerToTrain, networkIndex)		
	
	if(updateOrder == "updateWeightsAfterAidealCalculations"):
		for l in reversed(range(minLayerToTrain, numberOfLayers)):
			if(debugVerboseOutputTrain):
				print("calculateAerror: l = ", l)
			calculateAndSetAerror(l, networkIndex)	
		for l in range(minLayerToTrain, numberOfLayers+1):	#optimisation note: this can be done in parallel (weights can be updated for each layer simultaneously)
			#print("updateWeightsBasedOnAerror: l = ", l)
			updateWeightsBasedOnAerror(l, x, y, networkIndex)
	elif(updateOrder == "updateWeightsDuringAidealCalculations"):
		for l in reversed(range(minLayerToTrain, numberOfLayers+1)):
			if(debugVerboseOutputTrain):
				print("calculateAerror: l = ", l)
			if(l != minLayerToTrain):
				calculateAndSetAerror(l-1, networkIndex, y)	
			updateWeightsBasedOnAerror(l, x, y, networkIndex)
	elif(updateOrder == "updateWeightsBeforeAidealCalculations"):
		for l in reversed(range(minLayerToTrain, numberOfLayers+1)):
			if(debugVerboseOutputTrain):
				print("calculateAerror: l = ", l)
			updateWeightsBasedOnAerror(l, x, y, networkIndex)
			if(l != minLayerToTrain):
				calculateAndSetAerror(l-1, networkIndex)


def calculateAndSetAerrorTopLayerWrapper(A, pred, y_true, networkIndex=1):
	AerrorVec, y_pred = calculateAerrorTopLayerWrapper(A, pred, y_true, networkIndex)
	setAerror(AerrorVec, y_pred, numberOfLayers, networkIndex)

def calculateAerrorTopLayerWrapper(A, pred, y_true, networkIndex=1):

	if(activationFunctionTypeFinalLayer == "sigmoid"):
		y_pred = A	#A is after sigmoid
	elif(activationFunctionTypeFinalLayer == "softmax"):	
		y_pred = pred	#pred is after softmax

	return calculateAerrorTopLayer(y_pred, y_true, networkIndex)			


def calculateAerrorTopLayer(y_pred, y_true, networkIndex=1):
	
	if(applyFinalLayerLossFunction):
		if(activationFunctionTypeFinalLayer == "softmax"):
			loss = ANNtf2_operations.calculateLossCrossEntropy(y_pred, y_true, datasetNumClasses, costCrossEntropyWithLogits=False, oneHotEncoded=True, reduceMean=False)
		elif(activationFunctionTypeFinalLayer == "sigmoid"):		
			loss = ANNtf2_operations.calculateLossCrossEntropy(y_pred, y_true, datasetNumClasses=None, costCrossEntropyWithLogits=True, reduceMean=False)	#loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
		else:
			print("activationFunctionTypeFinalLayer not currently supported by RUANN = ", activationFunctionTypeFinalLayer)
			exit()
		AerrorAbs = loss
		#print("loss = ", loss)
	
		#calculate signed error:
		AidealDelta = calculateADelta(y_true, y_pred)
		AidealDeltaSign = tf.sign(AidealDelta)
		AerrorVec = tf.multiply(AerrorAbs, AidealDeltaSign)	#updateWeightsBasedOnAidealHeuristic requires directional error
	else:
		if(topLayerIdealAstrict):
			AerrorVec = tf.subtract(y_true, y_pred)	
		else:
			#calculate Aideal of final layer based on y	
			AdeltaMax = tf.subtract(y_true, y_pred)	
			AerrorVec = calculateDeltaTF(AdeltaMax, topLayerIdealAproximity, True, applyMinimia=False)

	if(averageAerrorAcrossBatch):
		AerrorVec = tf.reduce_mean(AerrorVec, axis=0)      #average across batch 
		y_pred = tf.reduce_mean(y_pred, axis=0)      #average across batch 
		
	return AerrorVec, y_pred

				
def calculateAndSetAerrorBottomLayer(x, minLayerToTrain, networkIndex=1):
	
	if(averageAerrorAcrossBatch):
		xAveraged = tf.reduce_mean(x, axis=0)      #average across batch
	else:
		xAveraged = x
		
	setAerrorGivenAideal(xAveraged, xAveraged, 0, networkIndex)	#set Aideal of input layer to x
	
	if(debugOnlyTrainFinalLayer):
		for l in range(1, minLayerToTrain):
			setAerrorGivenAideal(getAtraceComparison(l, networkIndex), getAtraceComparison(l, networkIndex), l, networkIndex)	#set Aideal of input layer to Atrace



def calculateAndSetAerror(l, networkIndex=1, y=None):

	#stochastically identify Aideal of l (lower) based on Aideal of l+1
		#this is biologically achieved by temporarily/independently adjusting the firing rate (~bias) of each neuron (index k) on l, and seeing if this better achieves Aideal of l+1
		#feedback (positive/negtive trial) is given from higher level l+1 to l_k in the form of "simple" [boolean] ~local chemical signal

	A = Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")]	#get original A value of lower layer
	if(learningAlgorithm == "backpropApproximation1"):
		setAerrorBackpropStrict(A, l, networkIndex)
	elif(learningAlgorithm == "backpropApproximation7"):
		setAerrorFromOutgoingWeightAdjustments(A, l, networkIndex)
	else:
		for k in range(n_h[l]):
			if(debugVerboseOutputTrain):
				print("\tcalculateAideal: k = ", k)
			#try both positive and negative adjustments of A_l_k;
			if(learningAlgorithm == "backpropApproximation2"):
				setAerrorBackpropSemi(A, k, l, networkIndex)
			elif(learningAlgorithm == "backpropApproximation6"):
				setAerrorBackpropFullNetworkCalculation(A, k, l, y, networkIndex)
			else:
				trialAidealMod(True, A, k, l, networkIndex)
				trialAidealMod(False, A, k, l, networkIndex)

def setAerrorBackpropStrict(A, l, networkIndex):
	AerrorVec = calculateAerrorBackpropStrict(A, l, networkIndex)
	setAerror(AerrorVec, getAtraceComparison(l, networkIndex), l, networkIndex)
	
def calculateAerrorBackpropStrict(A, l, networkIndex):

	Z = Ztrace[generateParameterNameNetwork(networkIndex, l, "Ztrace")]	#get original Z value of current layer

	AerrorAbove =  getAerror(l+1, networkIndex)
	WAbove = W[generateParameterNameNetwork(networkIndex, l+1, "W")]
	
	if(averageAerrorAcrossBatch):
		AerrorAbove = tf.expand_dims(AerrorAbove, axis=0)

	AerrorVec = tf.matmul(AerrorAbove, tf.transpose(WAbove))	#(W_l+1 * error_l+1)	#multiply by the strength of the signal weight passthrough	#multiply by the strength of the higher layer error	
	zPrime = activationFunctionPrime(Z)
	AerrorVec = tf.multiply(AerrorVec, zPrime) 		#. zPrime_l	#multiply by the strength of the current layer zPrime
	#print("Z = ", Z)
	#print("zPrime = ", zPrime)
	#print("AerrorVec = ", AerrorVec)
	
	if(averageAerrorAcrossBatch):
		AerrorVec = tf.squeeze(AerrorVec)
		A = tf.reduce_mean(A, axis=0)   #average across batch
		
	return AerrorVec

def setAerrorFromOutgoingWeightAdjustments(A, l, networkIndex):
	AerrorVec = calculateAerrorFromOutgoingWeightAdjustments(A, l, networkIndex)
	setAerror(AerrorVec, getAtraceComparison(l, networkIndex), l, networkIndex)

def calculateAerrorFromOutgoingWeightAdjustments(A, l, networkIndex):

	# backpropagation approximation notes:
	# error_L = (y_L - A_L) [sign reversal]
	# error_l = (W_l+1 * error_l+1) . activationFunctionPrime(z_l) {~A_l}
	# dC/dB = error_l
	# dC/dW = A_l-1 * error_l
	# Bnew = B+dC/dB [sign reversal]
	# Wnew = W+dC/dW [sign reversal]

	Z = Ztrace[generateParameterNameNetwork(networkIndex, l, "Ztrace")]	#get original Z value of current layer

	WdeltaAbove = WdeltaStore[generateParameterNameNetwork(networkIndex, l+1, "WdeltaStore")]
	WAbove = W[generateParameterNameNetwork(networkIndex, l+1, "W")]
	
	#extract AerrorAbove from weights;
	
	#inverse matrix multiplication with batchSize = 1;
	A = getAcomparison(A)
	A = tf.squeeze(A)	#required when averageAerrorAcrossBatch=False but batchSize=1
	WdeltaAboveRow1 = WdeltaAbove[0, :]
	Arow1 = A[0]
	AerrorAbove = tf.divide(WdeltaAboveRow1, Arow1)
	AerrorAbove = tf.expand_dims(AerrorAbove, 0)
	#print("WdeltaAbove = ", WdeltaAbove)
	#print("A = ", A)		
	#print("WdeltaAboveRow1 = ", WdeltaAboveRow1)
	#print("Arow1 = ", Arow1)
	#print("AerrorAbove = ", AerrorAbove)
	
	#inverse matrix multiplication not possible with batchSize > 1;
	#Ainverse = tf.math.inverse(A)
	#AerrorAbove = tf.matmul(Ainverse, WdeltaAbove)		#NO: dC/dW = A_l-1 * error_l, therefore error_l = inverse(A) * dC/dW
		#if A is a vector then inversion possible
	#print("Ainverse.shape = ", Ainverse.shape)
	#print("WdeltaAbove = ", WdeltaAbove)
	#print("AerrorAbove = ", AerrorAbove)

	#strict backprop for debug only;
	#AerrorAbove = getAerror(l+1, networkIndex)
	#if(averageAerrorAcrossBatch):
	#	AerrorAbove = tf.expand_dims(AerrorAbove, axis=0)
	#print("AerrorAbove = ", AerrorAbove)
	
	AerrorVec = tf.matmul(AerrorAbove, tf.transpose(WAbove))	#(W_l+1 * error_l+1)	#multiply by the strength of the signal weight passthrough	#multiply by the strength of the higher layer error	
	zPrime = activationFunctionPrime(Z)
	if(averageAerrorAcrossBatch):
		zPrime = tf.reduce_mean(zPrime, axis=0)   #average across all  batches
	AerrorVec = tf.multiply(AerrorVec, zPrime) 		#. zPrime_l	#multiply by the strength of the current layer zPrime


	if(averageAerrorAcrossBatch):
		AerrorVec = tf.squeeze(AerrorVec)

	return AerrorVec
	
def setAerrorBackpropSemi(A, k, l, networkIndex):

	#error_l = (W_l+1 * error_l+1) * A_l = (A_l*W_l+1) * Aideal_l+1 - (A_l*W_l+1) * Atrace_l+1 

	AlayerAboveWithError = trialAerrorMod(True, A, k, l, networkIndex)
	AlayerAboveWithoutError = trialAerrorMod(False, A, k, l, networkIndex)
	AerrorLayer = calculateErrorAtrial(AlayerAboveWithoutError, AlayerAboveWithError, networkIndex, averageType="none")

	#only set Aerror of neuron k
	if(averageAerrorAcrossBatch):
		AerrorLayer = tf.reduce_mean(AerrorLayer, axis=0)   #average across all k neurons on l+1
	else:
		AerrorLayer = tf.reduce_mean(AerrorLayer, axis=1)   #average across all k neurons on l+1
		#AerrorLayer = tf.expand_dims(AerrorLayer, axis=0)
	
	setAerrorK(AerrorLayer, k, l, networkIndex)
	
	
def setAerrorBackpropFullNetworkCalculation(A, k, l, y, networkIndex):

	AerrorLayerFinalOrig = Aerror[generateParameterNameNetwork(networkIndex, numberOfLayers, "Aerror")]		#use Aerror from final layer

	y_true = tf.one_hot(y, depth=datasetNumClasses)
	
	#code from trialAidealMod:
	direction = True
	if(direction):
		trialAmodValue = subLayerIdealAlearningRateBase
	else:
		trialAmodValue = -subLayerIdealAlearningRateBase
	columnsIdx = tf.constant([k])
	AK = tf.gather(A, columnsIdx, axis=1)	#Atrial[:,k]	
	AtrialK = AK
	AtrialKdelta = calculateDeltaTF(AtrialK, trialAmodValue, useMultiplicationRatherThanAdditionOfDeltaValuesAideal)	#this integrates the fact in backpropagation Aerror should be linearly dependent on A  #* A_l	#multiply by the strength of the current layer signal
	AtrialK = tf.add(AtrialK, AtrialKdelta)
	Atrial = A
	Atrial = ANNtf2_operations.modifyTensorRowColumn(Atrial, False, k, AtrialK, isVector=True)	#Atrial[:,k] = (trialAmodValue)

	pred, Afinal, Zfinal = neuralNetworkPropagationLREANNlayer(Atrial, numberOfLayers, l, networkIndex, recordAtrace=(not recalculateAtraceUnoptimisedBio))
	AerrorLayerFinalMod, y_pred = calculateAerrorTopLayerWrapper(Afinal, pred, y_true, networkIndex)	
	
	AerrorLayer = tf.subtract(AerrorLayerFinalOrig, AerrorLayerFinalMod)
	
	#only set Aerror of neuron k
	if(averageAerrorAcrossBatch):
		AerrorLayer = tf.reduce_mean(AerrorLayer, axis=0)   #average across all k neurons on l+1
	else:
		AerrorLayer = tf.reduce_mean(AerrorLayer, axis=1)   #average across all k neurons on l+1
		#AerrorLayer = tf.expand_dims(AerrorLayer, axis=0)
	
	setAerrorK(AerrorLayer, k, l, networkIndex)
	
	

def trialAerrorMod(applyAboveLayerError, A, k, l, networkIndex):

	AtrialAbove, ZtrialAbove = neuralNetworkPropagationLREANNlayerLK(A, k, l+1, networkIndex)

	if(averageAerrorAcrossBatch):
		ZtrialAbove = tf.reduce_mean(ZtrialAbove, axis=0)   #average across batch
	else:
		None
			
	if(applyAboveLayerError):
		AerrorAbove = Aerror[generateParameterNameNetwork(networkIndex, l+1, "Aerror")]
		
		#unadjusted: #AerrorLayer = tf.add(ZtrialAbove, AerrorAbove) #AerrorLayer = tf.multiply(ZtrialAbove, AerrorAbove)
		#perform adjustment to ensure calculateErrorAtrial(AlayerAboveWithoutError, AlayerAboveWithError) will produce precise error value:
		AerrorAbove = tf.add(tf.sign(AerrorAbove), AerrorAbove)
		AerrorLayer = tf.multiply(ZtrialAbove, AerrorAbove)
	else:
		AerrorLayer = ZtrialAbove
		
	return AerrorLayer

def trialAidealMod(direction, A, k, l, networkIndex):

	if(direction):
		trialAmodValue = subLayerIdealAlearningRateBase
	else:
		trialAmodValue = -subLayerIdealAlearningRateBase
	
	columnsIdx = tf.constant([k])
	AK = tf.gather(A, columnsIdx, axis=1)	#Atrial[:,k]	
	AtrialK = AK
	if(learningAlgorithm == "backpropApproximation5"):
		AtrialKdelta = trialAmodValue
	elif(learningAlgorithm == "backpropApproximation4"):
		AtrialKdelta = calculateDeltaTF(AtrialK, trialAmodValue, useMultiplicationRatherThanAdditionOfDeltaValuesAideal)	#this integrates the fact in backpropagation Aerror should be linearly dependent on A  #* A_l	#multiply by the strength of the current layer signal
	elif(learningAlgorithm == "backpropApproximation3"):
		AtrialKdelta = trialAmodValue
	AtrialK = tf.add(AtrialK, AtrialKdelta)
	
	Atrial = A
	Atrial = ANNtf2_operations.modifyTensorRowColumn(Atrial, False, k, AtrialK, isVector=True)	#Atrial[:,k] = (trialAmodValue)

	AtrialAbove, ZtrialAbove = neuralNetworkPropagationLREANNlayerL(Atrial, l+1, networkIndex)
	
	if(learningAlgorithm == "backpropApproximation4"):
		AtrialKdelta = tf.squeeze(AtrialKdelta)
			
	if(averageAerrorAcrossBatch):
		AtrialAbove = getAcomparison(AtrialAbove)	#average across batch
		#AtrialKdelta is already averaged across all batches
		AtrialK = getAcomparison(AtrialK)	#average across batch
		if(learningAlgorithm == "backpropApproximation4"):
			AtrialKdelta = getAcomparison(AtrialKdelta)
		
	#print("AtrialKdelta.shape = ", AtrialKdelta.shape)
	#print("AtrialAbove.shape = ", AtrialAbove.shape)
		
	successfulTrial, performanceMultiplier = testAtrialPerformance(AtrialKdelta, AtrialAbove, l, networkIndex)
	
	#required for AtrialK/modifyTensorRowColumn preparation:
	if(averageAerrorAcrossBatch):
		successfulTrial = tf.expand_dims(successfulTrial, axis=0)
		performanceMultiplier = tf.expand_dims(performanceMultiplier, axis=0)
		if(learningAlgorithm == "backpropApproximation4"):
			AtrialKdelta = tf.expand_dims(AtrialKdelta, axis=0)
	else:
		successfulTrial = tf.expand_dims(successfulTrial, axis=1)
		performanceMultiplier = tf.expand_dims(performanceMultiplier, axis=1)
		if(learningAlgorithm == "backpropApproximation4"):
			AtrialKdelta = tf.expand_dims(AtrialKdelta, axis=1)

	successfulTrialFloat = tf.dtypes.cast(successfulTrial, dtype=tf.float32)
		
	#print("successfulTrial.shape = ", successfulTrial.shape)
	#print("performanceMultiplier.shape = ", performanceMultiplier.shape)
	#print("AtrialKdelta.shape = ", AtrialKdelta.shape)
	#print("AtrialK.shape = ", AtrialK.shape)
	
	if(learningAlgorithm == "backpropApproximation4"):
		if(applySubLayerIdealAmultiplierCorrection):
			AtrialKdelta = tf.multiply(AtrialKdelta, performanceMultiplier)	#W_l+1	#multiply by the strength of the signal weight passthrough
	elif(learningAlgorithm == "backpropApproximation3"):
		# error_l = (W_l+1 * error_l+1) * A_l
		AtrialKdelta = tf.multiply(AtrialKdelta, performanceMultiplier)	#W_l+1	#multiply by the strength of the signal weight passthrough
		AtrialKdelta = tf.multiply(AtrialKdelta, AtrialK) #* A_l	#multiply by the strength of the current layer signal
			
	AtrialKdeltaSuccessful = tf.multiply(AtrialKdelta, successfulTrialFloat)
	AtrialKSuccessful = tf.add(AtrialK, AtrialKdeltaSuccessful)

	#print("AtrialKSuccessful.shape = ", AtrialKSuccessful.shape)

	if(debugVerboseOutputTrain):
		print("AtrialKSuccessful", AtrialKSuccessful)
		
	AidealLayerNew = ANNtf2_operations.modifyTensorRowColumn(Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")], False, k, AtrialKSuccessful, isVector=True)
	
	setAerrorGivenAideal(AidealLayerNew, None, l, networkIndex)
	

def testAtrialPerformance(AtrialKdelta, AtrialAbove, l, networkIndex):
		
	performanceMultiplier = None	#performanceMultiplier: this calculates the ratio of the performance gain relative to the lower layer adjustment
	
	#print("l = ", l)
	#print("AtrialAbove.shape = ", AtrialAbove.shape)
	
	successfulTrial, trialPerformanceGain = testAtrialPerformanceAbove(AtrialAbove, l+1, networkIndex)

	if(learningAlgorithm == "backpropApproximation4"):
		performanceMultiplier = tf.divide(trialPerformanceGain, tf.abs(AtrialKdelta))	#added tf.abs to ensure sign of performanceMultiplier is maintained	#OLD: fix this; sometimes divides by zero	
			
		if(applySubLayerIdealAmultiplierRequirement):
			performanceMultiplierSuccessful = tf.greater(performanceMultiplier, subLayerIdealAmultiplierRequirement)
			successfulTrial = tf.logical_and(successfulTrial, performanceMultiplierSuccessful)
	elif(learningAlgorithm == "backpropApproximation3"):
		performanceMultiplier = tf.divide(trialPerformanceGain, tf.abs(AtrialKdelta))
		
	return successfulTrial, performanceMultiplier
	

def testAtrialPerformanceAbove(AtrialAbove, l, networkIndex):
	
	#print("l = ", l )
	#print("1 Aideal.shape = ", Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")].shape)

	AidealDeltaOrig = calculateAidealDelta(getAtraceComparison(l, networkIndex), l, networkIndex)
	AidealDeltaTrial = calculateAidealDelta(AtrialAbove, l, networkIndex)
	
	if(averageAerrorAcrossBatch):
		AidealDeltaOrigAvg = tf.reduce_mean(AidealDeltaOrig, axis=0)   #average across all k neurons on l
		AidealDeltaTrialAvg = tf.reduce_mean(AidealDeltaTrial, axis=0) #average across all k neurons on l	
	else:
		AidealDeltaOrigAvg = tf.reduce_mean(AidealDeltaOrig, axis=1)   #average across all k neurons on l
		AidealDeltaTrialAvg = tf.reduce_mean(AidealDeltaTrial, axis=1) #average across all k neurons on l
	AidealDeltaOrigAvgAbs = tf.abs(AidealDeltaOrigAvg)
	AidealDeltaTrialAvgAbs = tf.abs(AidealDeltaTrialAvg)
	successfulTrial = tf.less(AidealDeltaTrialAvgAbs, AidealDeltaOrigAvgAbs)
	
	if(learningAlgorithm == "backpropApproximation5"):
		trialPerformanceGain = None
	elif(learningAlgorithm == "backpropApproximation4"):
		#apply thresholding
		if(applyMinimiumAdeltaContributionThreshold):
			AidealDeltaOrigAvgThreshold = tf.multiply(tf.sign(AidealDeltaOrigAvg), tf.subtract(tf.abs(AidealDeltaOrigAvg), minimiumAdeltaContributionThreshold))
				#OLD: AidealDeltaOrigAvgThreshold = tf.multiply(tf.sign(AidealDeltaOrigAvg), tf.subtract(tf.abs(AidealDeltaOrigAvg), minimiumAdeltaContributionThreshold)) #tf.maximum(tf.subtract(tf.abs(AidealDeltaOrigAvg), minimiumAdeltaContributionThreshold), 0.0)
			successfulTrial = tf.math.logical_not(tf.math.logical_xor(tf.less(AidealDeltaTrialAvg, AidealDeltaOrigAvgThreshold), tf.equal(tf.sign(AidealDeltaTrialAvg), 1)))    #tf.multiply(tf.less(AidealDeltaTrialAvg, AidealDeltaOrigAvg), tf.sign(AidealDeltaTrialAvg))	
		else:
			AidealDeltaOrigAvgAbs = tf.abs(AidealDeltaOrigAvg)
			AidealDeltaTrialAvgAbs = tf.abs(AidealDeltaTrialAvg)
			successfulTrial = tf.less(AidealDeltaTrialAvgAbs, AidealDeltaOrigAvgAbs)			
		trialPerformanceGain = tf.multiply(tf.subtract(AidealDeltaOrigAvg, AidealDeltaTrialAvg), tf.sign(AidealDeltaTrialAvg))	#orig trialPerformanceGain calculation method 	#W_l+1	#multiply by the strength of the signal weight passthrough
		#Algorithm limitation - Missing:  * error_l+1	#multiply by the strength of the higher layer error
	elif(learningAlgorithm == "backpropApproximation3"):
		trialPerformanceGain = tf.multiply(tf.subtract(AidealDeltaOrig, AidealDeltaTrial), tf.sign(AidealDeltaTrial))	#W_l+1	#multiply by the strength of the signal weight passthrough
		trialPerformanceGain = tf.multiply(trialPerformanceGain, tf.abs(AidealDeltaOrig))	# * error_l+1	#multiply by the strength of the higher layer error	#apply abs correction to AidealDeltaOrig?
		if(averageAerrorAcrossBatch):
			trialPerformanceGain = tf.reduce_mean(trialPerformanceGain, axis=0) #average across all k neurons on l
		else:
			trialPerformanceGain = tf.reduce_mean(trialPerformanceGain, axis=1) #average across all k neurons on l
			
	return successfulTrial, trialPerformanceGain

		
		
def updateWeightsBasedOnAerror(l, x, y, networkIndex):
	
	
	if(learningAlgorithm == "backpropApproximation1" 
	or learningAlgorithm == "backpropApproximation2"
	or learningAlgorithm == "backpropApproximation7"):
		Wlayer = W[generateParameterNameNetwork(networkIndex, l, "W")]
		Blayer = B[generateParameterNameNetwork(networkIndex, l, "B")]
		
		AtraceBelow = Atrace[generateParameterNameNetwork(networkIndex, l-1, "Atrace")]
		AerrorLayer = getAerror(l, networkIndex)
			#Aerror[generateParameterNameNetwork(networkIndex, l, "Aerror")]	

		#OLD:
		#AtraceBelow = getAtraceComparison(l-1, networkIndex)	
		#if(not averageAerrorAcrossBatch):
		#	AtraceBelow = tf.reduce_mean(AtraceBelow, axis=0)      #average across batch 
		#	AerrorLayer = tf.reduce_mean(AerrorLayer, axis=0)      #average across batch 
		#AtraceBelow = tf.expand_dims(AtraceBelow, axis=1)	#required for matmul preparation
		#AerrorLayer = tf.expand_dims(AerrorLayer, axis=0)	#required for matmul preparation

		if(averageAerrorAcrossBatch):
			AtraceBelow = tf.reduce_mean(AtraceBelow, axis=0)      #average across batch
			AtraceBelow = tf.expand_dims(AtraceBelow, axis=0)	#required for matmul preparation
			AerrorLayer = tf.expand_dims(AerrorLayer, axis=0)

		#print("AtraceBelow.shape = ", AtraceBelow.shape)
		#print("AerrorLayer.shape = ", AerrorLayer.shape)
		
		Wdelta = tf.matmul(tf.transpose(AtraceBelow), AerrorLayer)	# dC/dW = A_l-1 * error_l
		Bdelta = tf.reduce_mean(AerrorLayer, axis=0) 	# dC/dB = error_l
		
		if(learningAlgorithm == "backpropApproximation7"):
			WdeltaStore[generateParameterNameNetwork(networkIndex, l, "WdeltaStore")] = Wdelta
			
		
		Wlayer = tf.add(Wlayer, tf.multiply(Wdelta, learningRate))
		Blayer = tf.add(Blayer, tf.multiply(Bdelta, learningRate))
		
		W[generateParameterNameNetwork(networkIndex, l, "W")] = Wlayer
		B[generateParameterNameNetwork(networkIndex, l, "B")] = Blayer
	elif(learningAlgorithm == "backpropApproximation3"):
		print("updateWeightsBasedOnAerror warning: learningAlgorithm == backpropApproximation3 has not been coded")
	elif(learningAlgorithm == "backpropApproximation6"):
		print("updateWeightsBasedOnAerror warning: learningAlgorithm == backpropApproximation6 has not been coded")
	else:
		#if(takeAprevLayerFromTraceDuringWeightUpdates):
		if(recalculateAtraceUnoptimisedBio):
			_ , AprevLayer, _ = neuralNetworkPropagationLREANNlayer(x, l-1, networkIndex, recordAtrace=False)
		else:
			AprevLayer = Atrace[generateParameterNameNetwork(networkIndex, l-1, "Atrace")] 
	
		if(recalculateAtraceUnoptimisedBio):
			AtrialBase, ZtrialBase = neuralNetworkPropagationLREANNlayerL(AprevLayer, l, networkIndex)
		else:
			AtrialBase = Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")]

		AidealLayer = Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")]

		if(useWeightUpdateDirectionHeuristicBasedOnExcitatoryInhibitorySynapseType):
			errorVec = calculateErrorAtrialVectorDirectional(getAcomparison(AtrialBase), AidealLayer, networkIndex)	#errorVec will be averaged across all batches 
			updateWeightsBasedOnAidealHeuristic(l, networkIndex, errorVec)
		else:
			lossBase = calculateErrorAtrial(AtrialBase, AidealLayer, networkIndex)		#lossBase will be averaged across all batches, across k neurons on l
			updateWeightsBasedOnAidealStochastic(l, AprevLayer, AidealLayer, networkIndex, lossBase, x)

def updateWeightsBasedOnAidealHeuristic(l, networkIndex, errorVec):

	Wlayer = W[generateParameterNameNetwork(networkIndex, l, "W")]

	AidealDeltaTensorSizeW = tf.expand_dims(errorVec, axis=0)
	multiples = tf.constant([n_h[l-1],1], tf.int32)
	AidealDeltaOrigTensorSizeW = tf.tile(AidealDeltaTensorSizeW, multiples)

	AidealDeltaTensorSizeWSign = tf.sign(AidealDeltaTensorSizeW)
	
	learningRateW = learningRate	#useMultiplicationRatherThanAdditionOfDeltaValuesW: note effective weight learning rate is currently ~topLayerIdealAproximity*subLayerIdealAlearningRateBase*learningRate
	if(learningAlgorithm == "backpropApproximation5"):
		Wdelta = tf.multiply(AidealDeltaTensorSizeWSign, learningRateW)
	elif(learningAlgorithm == "backpropApproximation4"):
		if(useMultiplicationRatherThanAdditionOfDeltaValuesW):
			Wdelta = calculateDeltaTF(AidealDeltaTensorSizeW, learningRateW, useMultiplicationRatherThanAdditionOfDeltaValuesW)
		else:
			Wdelta = tf.multiply(AidealDeltaTensorSizeWSign, learningRateW)
	WlayerNew = tf.add(Wlayer, Wdelta)
	
	W[generateParameterNameNetwork(networkIndex, l, "W")] = WlayerNew

def updateWeightsBasedOnAidealStochastic(l, AprevLayer, AidealLayer, networkIndex, lossBase, x):

	#stochastic algorithm extracted from neuralNetworkPropagationLREANN_expSUANNtrain_updateNeurons()g

	if(useBinaryWeights):
		variationDirections = 1
	else:
		variationDirections = 2
	
	for hIndexCurrentLayer in range(0, n_h[l]):	#k of l
		for hIndexPreviousLayer in range(0, n_h[l-1]+1):	#k of l-1
			if(hIndexPreviousLayer == n_h[l-1]):	#ensure that B parameter updates occur/tested less frequently than W parameter updates
				parameterTypeWorB = 0
			else:
				parameterTypeWorB = 1
			for variationDirectionInt in range(variationDirections):

				networkParameterIndexBase = (parameterTypeWorB, l, hIndexCurrentLayer, hIndexPreviousLayer, variationDirectionInt)
				networkParameterIndex = networkParameterIndexBase
	
				accuracyImprovementDetected = False
					
				if(not useBinaryWeights):
					if(networkParameterIndex[NETWORK_PARAM_INDEX_VARIATION_DIRECTION] == 1):
						variationDiff = learningRate
					else:
						variationDiff = -learningRate		
				
				if(networkParameterIndex[NETWORK_PARAM_INDEX_TYPE] == 1):
					currentVal = W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")][networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].numpy()

					if(useBinaryWeights):
						if(useBinaryWeightsReduceMemoryWithBool):
							newVal = not currentVal
						else:
							newVal = float(not bool(currentVal))
					else:
						WtrialDelta = calculateDeltaNP(currentVal, variationDiff, useMultiplicationRatherThanAdditionOfDeltaValuesW)
						newVal = currentVal + WtrialDelta
						
					W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")][networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].assign(newVal)
				else:
					currentVal = B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")][networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].numpy()

					if(useBinaryWeights):
						if(useBinaryWeightsReduceMemoryWithBool):
							newVal = not currentVal
						else:
							newVal = float(not bool(currentVal))
					else:
						BtrialDelta = calculateDeltaNP(currentVal, variationDiff, useMultiplicationRatherThanAdditionOfDeltaValuesW)
						newVal = currentVal + BtrialDelta
					B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")][networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].assign(newVal)
			
				Atrial, Ztrial = neuralNetworkPropagationLREANNlayerL(AprevLayer, l, networkIndex)
				error = calculateErrorAtrial(Atrial, AidealLayer, networkIndex)	#average across batch, across k neurons on l
				
				if(error < lossBase):
					accuracyImprovementDetected = True
					lossBase = error
					#print("\t(error < lossBase): error = ", error)				
							
				if(accuracyImprovementDetected):
					#print("accuracyImprovementDetected")
					Wbackup[generateParameterNameNetwork(networkIndex, l, "W")].assign(W[generateParameterNameNetwork(networkIndex, l, "W")])
					Bbackup[generateParameterNameNetwork(networkIndex, l, "B")].assign(B[generateParameterNameNetwork(networkIndex, l, "B")])								
				else:
					#print("!accuracyImprovementDetected")
					W[generateParameterNameNetwork(networkIndex, l, "W")].assign(Wbackup[generateParameterNameNetwork(networkIndex, l, "W")])
					B[generateParameterNameNetwork(networkIndex, l, "B")].assign(Bbackup[generateParameterNameNetwork(networkIndex, l, "B")])					



def activationFunction(Z, prevLayerSize=None):
	return activationFunctionCustom(Z, prevLayerSize)
	
def activationFunctionCustom(Z, prevLayerSize=None):

	if(useBinaryWeights):	
		#offset required because negative weights are not used:
		Zoffset = tf.ones(Z.shape)
		Zoffset = tf.multiply(Zoffset, averageTotalInput)
		Zoffset = tf.multiply(Zoffset, prevLayerSize/2)
		Z = tf.subtract(Z, Zoffset) 
	
	if(activationFunctionType == "relu"):
		A = tf.nn.relu(Z)
	elif(activationFunctionType == "sigmoid"):
		A = tf.nn.sigmoid(Z)
	elif(activationFunctionType == "softmax"):
		A = tf.nn.softmax(Z)
		
	return A

 
def calculateDeltaTF(deltaMax, learningRateLocal, useMultiplication, applyMinimia=True):
	if(useMultiplication):
		deltaMaxAbs = tf.abs(deltaMax)
		deltaAbs = tf.multiply(deltaMaxAbs, learningRateLocal)
		if(applyMinimia):
			learningRateLocalMin = learningRateLocal*learningRateMinFraction
			deltaMinAbs = learningRateLocalMin
			deltaAbs = tf.maximum(deltaAbs, deltaMinAbs)
		deltaAbs = tf.minimum(deltaAbs, deltaMaxAbs)
		delta = tf.multiply(deltaAbs, tf.sign(deltaMax))
	else:
		delta = learningRateLocal	#tf.multiply(deltaMax, learningRateLocal)
	return delta

def calculateDeltaNP(deltaMax, learningRateLocal, useMultiplication, applyMinimia=True):
	if(useMultiplication):
		deltaMaxAbs = np.abs(deltaMax)
		deltaAbs = np.multiply(deltaMaxAbs, learningRateLocal)
		if(applyMinimia):
			learningRateLocalMin = learningRateLocal*learningRateMinFraction
			deltaMinAbs = learningRateLocalMin
			deltaAbs = np.maximum(deltaAbs, deltaMinAbs)
		deltaAbs = np.minimum(deltaAbs, deltaMaxAbs)
		delta = np.multiply(deltaAbs, np.sign(deltaMax))
	else:
		delta = learningRateLocal	#np.multiply(deltaMax, learningRateLocal) 
	return delta
		
def calculateErrorAtrialVectorDirectional(Atrial, AidealLayer, networkIndex, averageType="vector"):
	
	#errorVec will be averaged across all batches if not already
	errorVec = calculateErrorAtrial(Atrial, AidealLayer, networkIndex, averageType)

	return errorVec
		
def calculateErrorAtrial(Atrial, AidealLayer, networkIndex, averageType="all"):

	AidealDelta = calculateADelta(AidealLayer, Atrial)
	error = AidealDelta

	if(averageAerrorAcrossBatch):
		if(averageType == "all"):
			error = tf.reduce_mean(error)	#average across k neurons on l
		elif(averageType == "vector"):
			error = error
		elif(averageType == "none"):
			error = error
	else:
		if(averageType == "all"):
			error = tf.reduce_mean(error)	#average across batch, across k neurons on l
		elif(averageType == "vector"):
			error = tf.reduce_mean(error, axis=0)	#average across batch	
		elif(averageType == "none"):
			error = error
									
	return error	
				

def calculateAidealDelta(A, l, networkIndex):
	AidealDelta = calculateADelta(Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")], A)
	return AidealDelta
	
def calculateADelta(Abase, A):
	AidealDelta =  tf.subtract(Abase, A)
	return AidealDelta

def setAerrorK(AerrorK, k, l, networkIndex=1):
	if(averageAerrorAcrossBatch):
		isRow = True
		isVector = False
	else:
		isRow = False
		isVector = False
	if(errorStorageAlgorithm == "useAideal"):
		print("setAerrorK requires (errorStorageAlgorithm == useAerror)")
	elif(errorStorageAlgorithm == "useAerror"):
		AerrorLayer = Aerror[generateParameterNameNetwork(networkIndex, l, "Aerror")]
		#print("AerrorLayer.shape = ", AerrorLayer.shape)
		#print("AerrorK.shape = ", AerrorK.shape)
		#print("AerrorK = ", AerrorK)
		AerrorLayer = ANNtf2_operations.modifyTensorRowColumn(AerrorLayer, isRow, k, AerrorK, isVector)
		Aerror[generateParameterNameNetwork(networkIndex, l, "Aerror")] = AerrorLayer
		#print("AerrorLayer.shape = ", AerrorLayer.shape)
			
def setAerror(AerrorLayer, AtraceLayer, l, networkIndex=1):
	if(errorStorageAlgorithm == "useAideal"):
		#print("setAerror, AtraceLayer.shape = ", AtraceLayer.shape)
		#print("setAerror, AerrorLayer.shape = ", AerrorLayer.shape)
		AtraceLayer = getAcomparison(AtraceLayer)
		Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")] = tf.add(AtraceLayer, AerrorLayer)
	elif(errorStorageAlgorithm == "useAerror"):
		#print("setAerror, AerrorLayer.shape = ", AerrorLayer.shape)
		Aerror[generateParameterNameNetwork(networkIndex, l, "Aerror")] = AerrorLayer

def setAerrorGivenAideal(AidealLayer, AtraceLayer, l, networkIndex=1):
	if(errorStorageAlgorithm == "useAideal"):
		#print("setAerrorGivenAideal, AidealLayer.shape = ", AidealLayer.shape)
		Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")] = AidealLayer
	elif(errorStorageAlgorithm == "useAerror"):
		#print("setAerrorGivenAideal, AidealLayer.shape = ", AidealLayer.shape)
		#print("setAerrorGivenAideal, AtraceLayer.shape = ", AtraceLayer.shape)
		AtraceLayer = getAcomparison(AtraceLayer)
		Aerror[generateParameterNameNetwork(networkIndex, l, "Aerror")] = tf.subtract(AidealLayer, AtraceLayer)

def getAerror(l, networkIndex=1):
	if(errorStorageAlgorithm == "useAideal"):
		AidealLayer = Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")]
		AtraceLayer = Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")]
		AtraceLayer = getAcomparison(AtraceLayer)
		AerrorLayer = tf.subtract(AidealLayer, AtraceLayer)
	elif(errorStorageAlgorithm == "useAerror"):
		AerrorLayer = Aerror[generateParameterNameNetwork(networkIndex, l, "Aerror")]
	return AerrorLayer
		
def getAtraceComparison(l, networkIndex=1):		
	AtraceLayer = Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")]
	AtraceLayer = getAcomparison(AtraceLayer)
	return AtraceLayer
	
def getAcomparison(A):		
	if(averageAerrorAcrossBatch):
		A = tf.reduce_mean(A, axis=0)      #average across batch
	return A

def activationFunctionPrime(z):
	#derivative of the activation function
	if(activationFunctionType == "relu"):
		prime = tf.math.greater(z, 0.0)
		prime = tf.cast(prime, tf.float32)
	elif(activationFunctionType == "sigmoid"):
		prime = tf.multiply(tf.nn.sigmoid(z), (tf.subtract(1.0, tf.nn.sigmoid(z))))
	return prime
	
