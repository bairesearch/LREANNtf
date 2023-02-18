"""LREANNtf_algorithmLREANN_expAUANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see LREANNtf_main.py

# Usage:
see LREANNtf_main.py

# Description:
LREANNtf algorithm LREANN expAUANN - define learning rule experiment artificial neural network with associative (wrt exemplar) update

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *
import ANNtf2_operations
import ANNtf2_globalDefs
import math
from numpy import random

sparsityLevel = 0.1	#probability of initial strong neural connection per neuron in layer	#~poisson distribution

classTargetExemplarsXList = []
classTargetExemplarsYList = []
#OLD tf	classTargetExemplarsLists code:
#classTargetExemplarsXList = None
#classTargetExemplarsYList = None

generateLargeNetwork = True	#required for LREANNtf_algorithmLREANN_expAUANN (see dynamicFinalLayerClassTargetAssignment: exemplar creation currently requires for each data class, at least one successful propagation to output layer in randomly initiated network)
	#consider also setting numberOfLayers=3 (as per original spec)

#exemplarSelectionRequiresCorrectFirstPropagation = True	#default: True (False is not yet implementated, would require algorithm upgrade to determine how to tweak higher levels towards correct target during exemplar selection process)
supportMultipleExemplarsPerClass = True	#default: False #train multiple exemplars oer class
dynamicFinalLayerClassTargetAssignment = True	#default: False #do not assign final layer output class target neurons until encountered first case (exemplar)
if(dynamicFinalLayerClassTargetAssignment):
	dynamicFinalLayerClassTargetAssignmentNumOutputNeuronsAsFractionOfNumClasses = 2
	classTargetExemplarsDynamicOutputNeuronIndexList = []
else:
	dynamicFinalLayerClassTargetAssignmentNumOutputNeuronsAsFractionOfNumClasses = -1
#avoidFinalLayerConnectionUntilTrained
#firstLayerToAssociateNeuronsWithExemplar = 1
#lastLayerToAssociateNeuronsWithExemplar = 1

learningRate = 0.1	#LREANN_expAUANN is designed to use a high learning rate for 1-shot learning (by association with/update of exemplar contents)
enableForgetting = True
if(enableForgetting):
	enableForgettingRestrictToAPrevAndNotAConnections = True	#True	#this ensures that only connections between active lower layer neurons and unactive higher layer exemplar neurons are suppressed
	forgetRate = 0.01	#CHECKTHIS

onlyTrainNeuronsIfActivationContributionAboveThresholdExemplar = True
if(onlyTrainNeuronsIfActivationContributionAboveThresholdExemplar):
	onlyTrainNeuronsIfActivationContributionAboveThresholdExemplarValue = 0.1
onlyTrainNeuronsIfActivationContributionAboveThreshold = True	#theshold neurons which will be positively biased, and those which will be negatively (above a = 0 as it is currently) 
if(onlyTrainNeuronsIfActivationContributionAboveThreshold):
	onlyTrainNeuronsIfActivationContributionAboveThresholdValue = 0.1

biologicalConstraints = False	#batchSize=1, _?

#learning algorithm embedded in forward propagation of new class x experience following forward propagation of existing class x experience
useBatch = True
if(useBatch):
	batchSize = 10
else:
	batchSize = 1	

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

def calculateOutputNeuronsLREANN_expAUANN(datasetNumClasses):
	num_output_neurons = datasetNumClasses
	if(dynamicFinalLayerClassTargetAssignment):
		num_output_neurons = datasetNumClasses * dynamicFinalLayerClassTargetAssignmentNumOutputNeuronsAsFractionOfNumClasses
		#print("num_output_neurons = ", num_output_neurons)
	return num_output_neurons
	
def getNoisySampleGenerationNumSamples():
	return noisySampleGeneration, noisySampleGenerationNumSamples, noiseStandardDeviation
	
def defineTrainingParameters(dataset):

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
	global classTargetExemplarsDynamicOutputNeuronIndexList
	
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, generateLargeNetwork=generateLargeNetwork)
	
	#classTargetExemplarsDynamicOutputNeuronIndexList = [-1] * num_output_neurons

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
		
		if(recordAtrace):
			if(onlyTrainNeuronsIfActivationContributionAboveThresholdExemplar):
				#apply threshold to A
				AAboveThreshold = tf.math.greater(A, onlyTrainNeuronsIfActivationContributionAboveThresholdExemplarValue)
				AAboveThresholdFloat = tf.dtypes.cast(AAboveThreshold, dtype=tf.float32)
				ALearn = A*AAboveThresholdFloat
			else:
				ALearn = A
			#print("ALearn.shape = ", ALearn.shape)
			Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")] = ALearn
			
		AprevLayer = A
		
	pred = tf.nn.softmax(Z)
	
	#print("neuralNetworkPropagationLREANN pred.shape = ", pred.shape)	

	return pred
	

def neuralNetworkPropagationLREANN_test(x, y, networkIndex=1):

	pred = neuralNetworkPropagationLREANN(x, networkIndex)
	loss = ANNtf2_operations.calculateLossCrossEntropy(pred, y, datasetNumClasses, costCrossEntropyWithLogits=False)
	acc = ANNtf2_operations.calculateAccuracy(pred, y)
	
	return loss, acc
	
def generateYActualfromYLREANN_expAUANN(y, num_output_neurons):
	if(dynamicFinalLayerClassTargetAssignment):
		#print("y = ", y)
		#print("classTargetExemplarsDynamicOutputNeuronIndexList = ", classTargetExemplarsDynamicOutputNeuronIndexList)
		exemplarsY = np.zeros(y.shape)
		for e in range(y.shape[0]):
			exemplarsY[e] = classTargetExemplarsDynamicOutputNeuronIndexList[y[e]]
		#print("exemplarsY = ", exemplarsY)
	else:
		#print("generateYActualfromYLREANN_expAUANN error: requires dynamicFinalLayerClassTargetAssignment")
		exemplarsY = y
	return exemplarsY

def generateNumClassesActualLREANN_expAUANN(datasetNumClasses, num_output_neurons):
	if(dynamicFinalLayerClassTargetAssignment):
		return num_output_neurons
	else:
		#print("generateNumClassesActualLREANN_expAUANN error: requires dynamicFinalLayerClassTargetAssignment")
		return datasetNumClasses

def generateTFYActualfromYandExemplarYLREANN_expAUANN(y, exemplarsY):
	if(dynamicFinalLayerClassTargetAssignment):
		return exemplarsY
	else:
		#print("generateTFYActualfromYandExemplarYLREANN_expAUANN error: requires dynamicFinalLayerClassTargetAssignment")
		return y
	
def neuralNetworkPropagationLREANN_expAUANNtrain(x, y, exemplarsX, exemplarsY, currentClassTarget, networkIndex=1):

	#connect/associate x to/with exemplarX
	
	#debug:
	#print("batchSize = ", batchSize)
	#print("learningRate = ", learningRate)
	#print("x = ", x)
	
	#important notes;
	#the i) highest layer output (A) being forward propagated from x and ii) y is never used during training, as the output neuron actually being trained is defined in exemplarsY
	
	#sanity checks:
	numberOfExemplarX = exemplarsX.shape[0]	#default: 1
	numberOfDataX = x.shape[0]	#default: 1
	if(numberOfExemplarX != numberOfDataX):
		print("neuralNetworkPropagationLREANN_expAUANNtrain error: numberOfExemplarX != numberOfDataX")
		exit(0)
	numberOfExemplarY = exemplarsY.shape[0]	#default: 1
	numberOfDataY = y.shape[0]	#default: 1
	if(numberOfExemplarY != numberOfDataY):
		print("neuralNetworkPropagationLREANN_expAUANNtrain error: numberOfExemplarY != numberOfDataY")
		exit(0)
	dataOutputNeuronIndex = currentClassTarget 	#y[0]	#all class targets in data/exemplars tensor are identical
	if(currentClassTarget != y[0]):
		print("neuralNetworkPropagationLREANN_expAUANNtrain error: (currentClassTarget != y[0])")
		print("currentClassTarget = ", currentClassTarget)
		print("y[0] = ", y[0])
		exit(0)
	if(dynamicFinalLayerClassTargetAssignment):
		exemplarOutputNeuronIndex = classTargetExemplarsDynamicOutputNeuronIndexList[currentClassTarget]	#exemplarsY[0]	#all class targets in data/exemplars tensor are identical
		if(classTargetExemplarsDynamicOutputNeuronIndexList[currentClassTarget] != exemplarsY[0]):
			print("neuralNetworkPropagationLREANN_expAUANNtrain error: (classTargetExemplarsDynamicOutputNeuronIndexList[currentClassTarget] != exemplarsY[0])")
			exit(0)
	else:
		exemplarOutputNeuronIndex = dataOutputNeuronIndex
		if(y[0] != exemplarsY[0]):
			print("neuralNetworkPropagationLREANN_expAUANNtrain error: (y[0] != exemplarsY[0])")
			exit(0)		
		
	predExemplars = neuralNetworkPropagationLREANN(exemplarsX, networkIndex, recordAtrace=True)	#record exemplar activation traces
	
	accExemplars = ANNtf2_operations.calculateAccuracy(predExemplars, exemplarsY)
	print("CHECKTHIS: accExemplars (this should always remain close to 100%) = ", accExemplars)
	
	AprevLayer = x

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

		if(onlyTrainNeuronsIfActivationContributionAboveThreshold):
			#apply threshold to AprevLayer
			AprevLayerAboveThreshold = tf.math.greater(AprevLayer, onlyTrainNeuronsIfActivationContributionAboveThresholdValue)
			AprevLayerAboveThresholdFloat = tf.dtypes.cast(AprevLayerAboveThreshold, dtype=tf.float32)
			AprevLayerLearn = AprevLayer*AprevLayerAboveThresholdFloat
		else:
			AprevLayerLearn = AprevLayer
		Alearn = Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")]
		#print("Alearn = ", Alearn)

		#update weights based on hebbian learning rule
		#strengthen those connections that link the previous layer neuron to the exemplar activation trace for the class target (and weaken those that did not)

		#associate all successfully fired neurons [in AprevLayerLearn] with exemplar higher level neurons [Alearn] previously identified during exemplar activation trace
		#CHECKTHIS: note this is currently only a unidirectional association (to exemplar activation tree, and not from exemplar activation tree)

		AcoincidenceMatrix = tf.matmul(tf.transpose(AprevLayerLearn), Alearn)
		Wmod = AcoincidenceMatrix/batchSize*learningRate
		#print("Wmod.shape = ", Wmod.shape)
		#print("W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
		W[generateParameterNameNetwork(networkIndex, l, "W")] = W[generateParameterNameNetwork(networkIndex, l, "W")] + Wmod	#apply weight update

		if(enableForgetting):	#this isn't necessarily required for highly sparsely activated network + low shot learning
			if(enableForgettingRestrictToAPrevAndNotAConnections):
				AboolNeg = tf.math.equal(Alearn, 0.0)	#Abool = tf.math.greater(Alearn, 0.0), AboolNeg = tf.math.logical_not(Abool)
				#print("Abool = ",Abool)
				#AboolNegInt = tf.dtypes.cast(AboolNeg, tf.int32)
				AboolNegFloat = tf.dtypes.cast(AboolNeg, tf.float32)
				AcoincidenceMatrixForget = tf.matmul(tf.transpose(AprevLayerLearn), AboolNegFloat)
				Wmod2 = tf.square(AcoincidenceMatrixForget)/batchSize*forgetRate	#tf.square(AcoincidenceMatrixForget) - square is required to normalise the forget rate relative to the learn rate [assumes input tensor is < 1]
				#print("Wmod2 = ", Wmod2)
				W[generateParameterNameNetwork(networkIndex, l, "W")] = W[generateParameterNameNetwork(networkIndex, l, "W")] - Wmod2
			else:
				AcoincidenceMatrixIsZero = tf.math.equal(AcoincidenceMatrix, 0)
				#AcoincidenceMatrixIsZeroInt = tf.dtypes.cast(AcoincidenceMatrixIsZero, tf.int32)
				AcoincidenceMatrixIsZeroFloat = tf.dtypes.cast(AcoincidenceMatrixIsZero, dtype=tf.float32)
				Wmod2 = tf.square(AcoincidenceMatrixIsZeroFloat)/batchSize*forgetRate	#tf.square(AcoincidenceMatrixIsZeroFloat) - square is required to normalise the forget rate relative to the learn rate [assumes input tensor is < 1]
				#print("Wmod2 = ", Wmod2)
				W[generateParameterNameNetwork(networkIndex, l, "W")] = W[generateParameterNameNetwork(networkIndex, l, "W")] - Wmod2

		AprevLayer = A	#Alearn

	#clear Atrace;
	for l in range(1, numberOfLayers+1):
		Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")] = 0	#tf.zeros(n_h[l])
		
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

 


def generateTFtrainDataFromNParraysLREANN_expAUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses):

	trainDataList = []
	
	for classTarget in range(datasetNumClasses):
			
		train_xClassFiltered, train_yClassFiltered = filterNParraysByClassTarget(train_x, train_y, classTargetFilterIndex=classTarget)
		trainDataUnbatched = generateTFtrainDataUnbatchedFromNParrays(train_xClassFiltered, train_yClassFiltered)
		trainData = generateTFtrainDataFromTrainDataUnbatched(trainDataUnbatched, shuffleSize, batchSize)
		trainDataList.append(trainData)
		
	return trainDataList
		
		
def generateTFexemplarDataFromNParraysLREANN_expAUANN(train_x, train_y, networkIndex, shuffleSize, batchSize, datasetNumClasses, generateClassTargetExemplars):

	global classTargetExemplarsXList
	global classTargetExemplarsYList
	if(dynamicFinalLayerClassTargetAssignment):
		global classTargetExemplarsDynamicOutputNeuronIndexList

	#print("train_x = ", train_x)
	#print("train_y = ", train_y)
	#print("networkIndex = ", networkIndex)
	#print("shuffleSize = ", shuffleSize)
	#print("batchSize = ", batchSize)
	#print("datasetNumClasses = ", datasetNumClasses)
	#print("generateClassTargetExemplars = ", generateClassTargetExemplars)
	
	exemplarDataList = []
	
	if(generateClassTargetExemplars):
		#print("generateClassTargetExemplars")
		classTargetExemplarsDynamicOutputNeuronIndexList = [-1] * datasetNumClasses

		#classTargetExemplarsList global lists are currently required such that generateTFtrainDataFromNParraysLREANN_expAUANN can be executed for multiple epochs (w/wo generateClassTargetExemplars)
		classTargetExemplarsXList = [None] * datasetNumClasses	
		classTargetExemplarsYList = [None] * datasetNumClasses
					
	for classTarget in range(datasetNumClasses):
					
		train_xClassFiltered, train_yClassFiltered = filterNParraysByClassTarget(train_x, train_y, classTargetFilterIndex=classTarget)
		trainDataUnbatched = generateTFtrainDataUnbatchedFromNParrays(train_xClassFiltered, train_yClassFiltered)
		
		if(generateClassTargetExemplars):
		
			#print("\n\tgenerateClassTargetLists: classTarget = ", classTarget)

			train_xLength = train_x.shape[0]
			trainDataAll = trainDataUnbatched.batch(train_xLength)	#train_xLength + 1
			(x, y) = next(iter(trainDataAll))
			
			pred = neuralNetworkPropagationLREANN(x, networkIndex)		
			
			if(dynamicFinalLayerClassTargetAssignment):
				predMaxOutputIndex = tf.argmax(pred, 1).numpy()
				#print("\t\tpredMaxOutputIndex = ", predMaxOutputIndex)
				
				numInitialPredictionsCorrect = 0
				numInitialPredictionsFail = 0
				for experienceIndex in range(batchSize):
					#print("\t\t\texperienceIndex = ", experienceIndex)				
					experiencePredMaxOutput = predMaxOutputIndex[experienceIndex]
					if experiencePredMaxOutput in classTargetExemplarsDynamicOutputNeuronIndexList:
						#print("\t\t\t\texperiencePredMaxOutput = ", experiencePredMaxOutput)				
						if(experiencePredMaxOutput == classTargetExemplarsDynamicOutputNeuronIndexList[classTarget]):
							if(supportMultipleExemplarsPerClass):
								appendToClassTargetExemplarsList(classTargetExemplarsXList, classTargetExemplarsYList, classTarget, train_x[experienceIndex], experiencePredMaxOutput)
							numInitialPredictionsCorrect = numInitialPredictionsCorrect + 1
						else:
							numInitialPredictionsFail = numInitialPredictionsFail + 1
					else:
						#create new exemplar
						#print("\t\t\t\tcreate new exemplar")
						if(classTargetExemplarsDynamicOutputNeuronIndexList[classTarget] == -1):	#verify that all exemplars of classX have same outputIndex
							appendToClassTargetExemplarsList(classTargetExemplarsXList, classTargetExemplarsYList, classTarget, train_x[experienceIndex], experiencePredMaxOutput)
							classTargetExemplarsDynamicOutputNeuronIndexList[classTarget] = experiencePredMaxOutput
					#print("\tgenerateClassTargetLists: dynamicFinalLayerClassTargetAssignment, numInitialPredictionsCorrect = ", numInitialPredictionsCorrect)
					#print("\tgenerateClassTargetLists: dynamicFinalLayerClassTargetAssignment, numInitialPredictionsFail = ", numInitialPredictionsFail)
			else:
				correct_prediction = calculateCorrectPrediction(pred, y)
				correct_predictionNP = correct_prediction.numpy()
				
				xFiltered = train_xClassFiltered[correct_predictionNP]
				yFiltered = train_yClassFiltered[correct_predictionNP]
				
				if(xFiltered.size > 0):
					if(supportMultipleExemplarsPerClass):
						for exemplarIndex in range(xFiltered.shape[0]):
							appendToClassTargetExemplarsList(classTargetExemplarsXList, classTargetExemplarsYList, classTarget, xFiltered[exemplarIndex], yFiltered[exemplarIndex])
					else:
						idx = tf.constant([0])  #take first candidate exemplar in xFiltered
						xFilteredExemplar = xFiltered[0]
						yFilteredExemplar = yFiltered[0]
						appendToClassTargetExemplarsList(classTargetExemplarsXList, classTargetExemplarsYList, classTarget, xFilteredExemplar, yFilteredExemplar)
		
		#print("classTargetExemplarsXList = ", classTargetExemplarsXList)
		#print("classTarget = ", classTarget)
		
		if(classTargetExemplarsXList[classTarget].size == 0):
			if(dynamicFinalLayerClassTargetAssignment):
				print("generateTFtrainDataFromNParraysLREANN_expAUANN error: dynamicFinalLayerClassTargetAssignment: exemplar creation currently requires for each data class, at least one successful propagation to output layer in randomly initiated network")
				print("consider expanding size of network")
			else:
				print("generateTFtrainDataFromNParraysLREANN_expAUANN error: !dynamicFinalLayerClassTargetAssignment: exemplar creation currently requires for each data class, at least one successful propagation to class target in randomly initiated network")
				print("consider enabling dynamicFinalLayerClassTargetAssignment")
			print("classTarget = ", classTarget)
			print("classTargetExemplarsXList[classTarget] = ", classTargetExemplarsXList[classTarget])
			print("classTargetExemplarsYList[classTarget] = ", classTargetExemplarsYList[classTarget])
			exit(0)
			
		exemplarData = generateTFtrainDataFromNParrays(classTargetExemplarsXList[classTarget], classTargetExemplarsYList[classTarget], shuffleSize, batchSize)
		exemplarDataList.append(exemplarData)	
	
	return exemplarDataList


def appendToClassTargetExemplarsList(classTargetExemplarsXList, classTargetExemplarsYList, classTarget, exemplarToAddX, exemplarToAddY):
	
	if(classTargetExemplarsXList[classTarget] is not None):
		classTargetExemplarsXList[classTarget] = np.append(classTargetExemplarsXList[classTarget], np.expand_dims(exemplarToAddX, axis=0), axis=0)
		classTargetExemplarsYList[classTarget] = np.append(classTargetExemplarsYList[classTarget], np.expand_dims(exemplarToAddY, axis=0))	#classTargetExemplarsYList[classTarget] = np.append(classTargetExemplarsYList[classTarget], exemplarToAddY)
	else:
		classTargetExemplarsXList[classTarget] = np.expand_dims(exemplarToAddX, axis=0) 
		classTargetExemplarsYList[classTarget] = np.expand_dims(exemplarToAddY, axis=0) 	#exemplarToAddY

