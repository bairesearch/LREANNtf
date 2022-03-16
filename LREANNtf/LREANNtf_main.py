"""LREANNtf_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
Python 3 and Tensorflow 2.1+ 

conda create -n anntf2 python=3.7
source activate anntf2
conda install -c tensorflow tensorflow=2.3
conda install scikit-learn (ANNtf2_algorithmLIANN_math:SVD/PCA only)

# Usage:
python3 LREANNtf_main.py

# Description:
LREANNtf - train learning rule experiment artificial neural network (LREANN)

"""

# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np

import sys
np.set_printoptions(threshold=sys.maxsize)

#from ANNtf2_operations import *
import ANNtf2_operations
import ANNtf2_globalDefs
from numpy import random
import ANNtf2_loadDataset

#select algorithm:
algorithm = "LREANN"	#learning rule experiment artificial neural network

suppressGradientDoNotExistForVariablesWarnings = True

costCrossEntropyWithLogits = False
if(algorithm == "LREANN"):
	#select algorithmLREANN:
	#algorithmLREANN = "LREANN_expHUANN"	#incomplete+non-convergent
	#algorithmLREANN = "LREANN_expSUANN"	
	#algorithmLREANN = "LREANN_expAUANN"	#incomplete+non-convergent
	#algorithmLREANN = "LREANN_expCUANN"	#incomplete+non-convergent
	#algorithmLREANN = "LREANN_expXUANN"	#incomplete
	#algorithmLREANN = "LREANN_expMUANN"	#incomplete+non-convergent
	algorithmLREANN = "LREANN_expRUANN"
	if(algorithmLREANN == "LREANN_expHUANN"):
		import LREANNtf_algorithmLREANN_expHUANN as LREANNtf_algorithm
	elif(algorithmLREANN == "LREANN_expSUANN"):
		import LREANNtf_algorithmLREANN_expSUANN as LREANNtf_algorithm
	elif(algorithmLREANN == "LREANN_expAUANN"):
		import LREANNtf_algorithmLREANN_expAUANN as LREANNtf_algorithm
	elif(algorithmLREANN == "LREANN_expCUANN"):
		import LREANNtf_algorithmLREANN_expCUANN as LREANNtf_algorithm
	elif(algorithmLREANN == "LREANN_expXUANN"):
		XUANNnegativeSamplesComplement = False	#default: True
		XUANNnegativeSamplesAll = False	#default: False #orig implementation
		XUANNnegativeSamplesRandom = True	#default: False 
		import LREANNtf_algorithmLREANN_expXUANN as LREANNtf_algorithm
	elif(algorithmLREANN == "LREANN_expMUANN"):
		import LREANNtf_algorithmLREANN_expMUANN as LREANNtf_algorithm		
	elif(algorithmLREANN == "LREANN_expRUANN"):
		import LREANNtf_algorithmLREANN_expRUANN as LREANNtf_algorithm
							
#learningRate, trainingSteps, batchSize, displayStep, numEpochs = -1

#performance enhancements for development environment only: 
debugUseSmallPOStagSequenceDataset = True	#def:False	#switch increases performance during development	#eg data-POStagSentence-smallBackup
useSmallSentenceLengths = True	#def:False	#switch increases performance during development	#eg data-simple-POStagSentence-smallBackup
trainMultipleFiles = False	#def:True	#switch increases performance during development	#eg data-POStagSentence
trainMultipleNetworks = False	#trial improve classification accuracy by averaging over multiple independently trained networks (test)

if(trainMultipleFiles):
	randomiseFileIndexParse = True
	fileIndexFirst = 0
	if(useSmallSentenceLengths):
		fileIndexLast = 11
	else:
		fileIndexLast = 1202	#defined by wiki database extraction size
else:
	randomiseFileIndexParse = False
				
#loadDatasetType3 parameters:
#if generatePOSunambiguousInput=True, generate POS unambiguous permutations for every POS ambiguous data example/experience
#if onlyAddPOSunambiguousInputToTrain=True, do not train network with ambiguous POS possibilities
#if generatePOSunambiguousInput=False and onlyAddPOSunambiguousInputToTrain=False, requires simultaneous propagation of different (ambiguous) POS possibilities

numberOfNetworks = 1
trainMultipleNetworks = False

if(algorithm == "LREANN"):
	dataset = "SmallDataset"
	#trainMultipleNetworks not currently supported
	trainHebbianBackprop = False	#default: False

								
if(dataset == "SmallDataset"):
	smallDatasetIndex = 0 #default: 0 (New Thyroid)
	#trainMultipleFiles = False	#required
	smallDatasetDefinitionsHeader = {'index':0, 'name':1, 'fileName':2, 'classColumnFirst':3}	
	smallDatasetDefinitions = [
	(0, "New Thyroid", "new-thyroid.data", True),
	(1, "Swedish Auto Insurance", "UNAVAILABLE.txt", False),	#AutoInsurSweden.txt BAD
	(2, "Wine Quality Dataset", "winequality-whiteFormatted.csv", False),
	(3, "Pima Indians Diabetes Dataset", "pima-indians-diabetes.csv", False),
	(4, "Sonar Dataset", "sonar.all-data", False),
	(5, "Banknote Dataset", "data_banknote_authentication.txt", False),
	(6, "Iris Flowers Dataset", "iris.data", False),
	(7, "Abalone Dataset", "UNAVAILABLE", False),	#abaloneFormatted.data BAD
	(8, "Ionosphere Dataset", "ionosphere.data", False),
	(9, "Wheat Seeds Dataset", "seeds_datasetFormatted.txt", False),
	(10, "Boston House Price Dataset", "UNAVAILABLE", False)	#housingFormatted.data BAD
	]
	dataset2FileName = smallDatasetDefinitions[smallDatasetIndex][smallDatasetDefinitionsHeader['fileName']]
	datasetClassColumnFirst = smallDatasetDefinitions[smallDatasetIndex][smallDatasetDefinitionsHeader['classColumnFirst']]
	print("dataset2FileName = ", dataset2FileName)
	print("datasetClassColumnFirst = ", datasetClassColumnFirst)
			
if(debugUseSmallPOStagSequenceDataset):
	dataset1FileNameXstart = "Xdataset1PartSmall"
	dataset1FileNameYstart = "Ydataset1PartSmall"
	dataset3FileNameXstart = "Xdataset3PartSmall"
	dataset4FileNameStart = "Xdataset4PartSmall"
else:
	dataset1FileNameXstart = "Xdataset1Part"
	dataset1FileNameYstart = "Ydataset1Part"
	dataset3FileNameXstart = "Xdataset3Part"
	dataset4FileNameStart = "Xdataset4Part"
datasetFileNameXend = ".dat"
datasetFileNameYend = ".dat"
datasetFileNameStart = "datasetPart"
datasetFileNameEnd = ".dat"
xmlDatasetFileNameEnd = ".xml"


def generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize, datasetNumClasses=None):
	return ANNtf2_operations.generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)
		
def defineTrainingParameters(dataset, numberOfFeaturesPerWord=None, paddingTagIndex=None):
	return LREANNtf_algorithm.defineTrainingParameters(dataset)

def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths=None, numberOfFeaturesPerWord=None):
	return LREANNtf_algorithm.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks)	

def defineNeuralNetworkParameters():
	return LREANNtf_algorithm.defineNeuralNetworkParameters()

#define default forward prop function for backprop weights optimisation;
def neuralNetworkPropagation(x, networkIndex=1):
	return LREANNtf_algorithm.neuralNetworkPropagation(x, networkIndex)
	
#define default forward prop function for test (identical to below);
def neuralNetworkPropagationTest(test_x, networkIndex=1):
	return LREANNtf_algorithm.neuralNetworkPropagation(test_x, networkIndex)

#if(LREANNtf_algorithm.supportMultipleNetworks):
def neuralNetworkPropagationLayer(x, networkIndex, l):
	return LREANNtf_algorithm.neuralNetworkPropagationLayer(x, networkIndex, l)
def neuralNetworkPropagationAllNetworksFinalLayer(x):
	return LREANNtf_algorithm.neuralNetworkPropagationAllNetworksFinalLayer(x)


#define specific learning algorithms (non-backprop);
#algorithm LREANN:
def executeLearningLREANN(x, y, networkIndex=1):
	if(algorithmLREANN == "LREANN_expHUANN"):
		#learning algorithm embedded in forward propagation
		if(trainHebbianBackprop):
			pred = LREANNtf_algorithm.neuralNetworkPropagationLREANN_expHUANNtrain(x, y, networkIndex, trainHebbianBackprop=True, trainHebbianLastLayerSupervision=True)
		else:
			pred = LREANNtf_algorithm.neuralNetworkPropagationLREANN_expHUANNtrain(x, y, networkIndex, trainHebbianForwardprop=True, trainHebbianLastLayerSupervision=True)
	elif(algorithmLREANN == "LREANN_expSUANN"):
		#learning algorithm embedded in multiple iterations of forward propagation
		pred = LREANNtf_algorithm.neuralNetworkPropagationLREANN_expSUANNtrain(x, y, networkIndex)
	elif(algorithmLREANN == "LREANN_expCUANN"):
		#learning algorithm embedded in forward propagation of new class x experience following forward propagation of existing class x experience
		pred = LREANNtf_algorithm.neuralNetworkPropagationLREANN_expCUANNtrain(x, y, networkIndex)
	elif(algorithmLREANN == "LREANN_expMUANN"):
		#learning algorithm embedded in multiple forward propagation and synaptic delta calculations
		pred = LREANNtf_algorithm.neuralNetworkPropagationLREANN_expMUANNtrain(x, y, networkIndex)
	elif(algorithmLREANN == "LREANN_expRUANN"):
		#learning algorithm: in reverse order, stochastically establishing Aideal of each layer (by temporarily biasing firing rate of neurons) to better achieve Aideal of higher layer (through multiple local/single layer forward propagations), then (simultaneous/parallel layer processing) stochastically adjusting weights to fine tune towards Aideal of their higher layers
		pred = LREANNtf_algorithm.neuralNetworkPropagationLREANN_expRUANNtrain(x, y, networkIndex)
def executeLearningLREANN_expAUANN(x, y, exemplarsX, exemplarsY, currentClassTarget, networkIndex=1):
	#learning algorithm embedded in forward propagation of new class x experience following forward propagation of existing class x experience
	pred = LREANNtf_algorithm.neuralNetworkPropagationLREANN_expAUANNtrain(x, y, exemplarsX, exemplarsY, currentClassTarget, networkIndex)
def executeLearningLREANN_expXUANN(x, y, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, networkIndex):
	#learning algorithm: perform contrast training (diff of interclass experience with current experience, and diff of extraclass experience with current experience) at each layer of network
	pred = LREANNtf_algorithm.neuralNetworkPropagationLREANN_expXUANNtrain(x, y, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, networkIndex)

			

#def executeLearningLREANN(e, batchIndex, x, y, networkIndex):
#	pred = LREANNtf_algorithm.neuralNetworkPropagationLREANNtrainIntro(x, y, networkIndex)
		
#parameter l is only currently used for algorithm AEANN
def executeOptimisation(x, y, datasetNumClasses, numberOfLayers, optimizer, networkIndex=1, l=None):
	with tf.GradientTape() as gt:
		loss, acc = calculatePropagationLoss(x, y, datasetNumClasses, numberOfLayers, costCrossEntropyWithLogits, networkIndex, l)
		
	if(algorithm == "LREANN"):
		print("executeOptimisation error: algorithm LREANN not supported, use executeLearningLREANN() instead")
		exit()
			
	gradients = gt.gradient(loss, trainableVariables)
						
	if(suppressGradientDoNotExistForVariablesWarnings):
		optimizer.apply_gradients([
    		(grad, var) 
    		for (grad, var) in zip(gradients, trainableVariables) 
    		if grad is not None
			])
	else:
		optimizer.apply_gradients(zip(gradients, trainableVariables))
		

def calculatePropagationLoss(x, y, datasetNumClasses, numberOfLayers, costCrossEntropyWithLogits, networkIndex=1, l=None):
	
	acc = 0	#only valid for softmax class targets 
	pred = neuralNetworkPropagation(x, networkIndex)
	target = y
	loss = ANNtf2_operations.calculateLossCrossEntropy(pred, target, datasetNumClasses, costCrossEntropyWithLogits)	
	acc = ANNtf2_operations.calculateAccuracy(pred, target)	#only valid for softmax class targets 
	#print("x = ", x)
	#print("y = ", y)
	#print("2 loss = ", loss)
	#print("2 acc = ", acc)
			
	return loss, acc


#if(LREANNtf_algorithm.supportMultipleNetworks):

def testBatchAllNetworksFinalLayer(batchX, batchY, datasetNumClasses, numberOfLayers):
	
	AfinalHiddenLayerList = []
	for networkIndex in range(1, numberOfNetworks+1):
		AfinalHiddenLayer = neuralNetworkPropagationLayer(batchX, networkIndex, numberOfLayers-1)
		AfinalHiddenLayerList.append(AfinalHiddenLayer)	
	AfinalHiddenLayerTensor = tf.concat(AfinalHiddenLayerList, axis=1)
	#print("AfinalHiddenLayerTensor = ", AfinalHiddenLayerTensor)
	
	pred = neuralNetworkPropagationAllNetworksFinalLayer(AfinalHiddenLayerTensor)
	acc = calculateAccuracy(pred, batchY)
	print("Combined network: Test Accuracy: %f" % (acc))
	
def trainBatchAllNetworksFinalLayer(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, costCrossEntropyWithLogits, display):
	
	AfinalHiddenLayerList = []
	for networkIndex in range(1, numberOfNetworks+1):
		AfinalHiddenLayer = neuralNetworkPropagationLayer(batchX, networkIndex, numberOfLayers-1)
		AfinalHiddenLayerList.append(AfinalHiddenLayer)	
	AfinalHiddenLayerTensor = tf.concat(AfinalHiddenLayerList, axis=1)
	#print("AfinalHiddenLayerTensor = ", AfinalHiddenLayerTensor)
	#print("AfinalHiddenLayerTensor.shape = ", AfinalHiddenLayerTensor.shape)
	
	executeOptimisationAllNetworksFinalLayer(AfinalHiddenLayerTensor, batchY, datasetNumClasses, optimizer)

	pred = None
	if(display):
		loss, acc = calculatePropagationLossAllNetworksFinalLayer(AfinalHiddenLayerTensor, batchY, datasetNumClasses, costCrossEntropyWithLogits)
		print("Combined network: batchIndex: %i, loss: %f, accuracy: %f" % (batchIndex, loss, acc))
						
def executeOptimisationAllNetworksFinalLayer(x, y, datasetNumClasses, optimizer):
	with tf.GradientTape() as gt:
		loss, acc = calculatePropagationLossAllNetworksFinalLayer(x, y, datasetNumClasses, costCrossEntropyWithLogits)
		
	Wlist = []
	Blist = []
	Wlist.append(LREANNtf_algorithm.WallNetworksFinalLayer)
	Blist.append(LREANNtf_algorithm.BallNetworksFinalLayer)
	trainableVariables = Wlist + Blist

	gradients = gt.gradient(loss, trainableVariables)
						
	if(suppressGradientDoNotExistForVariablesWarnings):
		optimizer.apply_gradients([
    		(grad, var) 
    		for (grad, var) in zip(gradients, trainableVariables) 
    		if grad is not None
			])
	else:
		optimizer.apply_gradients(zip(gradients, trainableVariables))
			
def calculatePropagationLossAllNetworksFinalLayer(x, y, datasetNumClasses, costCrossEntropyWithLogits):
	acc = 0	#only valid for softmax class targets 
	#print("x = ", x)
	pred = neuralNetworkPropagationAllNetworksFinalLayer(x)
	#print("calculatePropagationLossAllNetworksFinalLayer: pred.shape = ", pred.shape)
	target = y
	loss = calculateLossCrossEntropy(pred, target, datasetNumClasses, costCrossEntropyWithLogits)	
	acc = calculateAccuracy(pred, target)	#only valid for softmax class targets 

	return loss, acc
	
	
		
def loadDataset(fileIndex):

	global numberOfFeaturesPerWord
	global paddingTagIndex
	
	datasetNumFeatures = 0
	datasetNumClasses = 0
	
	fileIndexStr = str(fileIndex).zfill(4)
	if(dataset == "POStagSequence"):
		datasetType1FileNameX = dataset1FileNameXstart + fileIndexStr + datasetFileNameXend
		datasetType1FileNameY = dataset1FileNameYstart + fileIndexStr + datasetFileNameYend
	elif(dataset == "POStagSentence"):
		datasetType3FileNameX = dataset3FileNameXstart + fileIndexStr + datasetFileNameXend		
	elif(dataset == "SmallDataset"):
		if(trainMultipleFiles):
			datasetType2FileName = dataset2FileNameStart + fileIndexStr + datasetFileNameEnd
		else:
			datasetType2FileName = dataset2FileName
	elif(dataset == "wikiXmlDataset"):
		datasetType4FileName = dataset4FileNameStart + fileIndexStr + xmlDatasetFileNameEnd
			
	numberOfLayers = 0
	if(dataset == "POStagSequence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType1(datasetType1FileNameX, datasetType1FileNameY)
	elif(dataset == "POStagSentence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType3(datasetType3FileNameX, generatePOSunambiguousInput, onlyAddPOSunambiguousInputToTrain, useSmallSentenceLengths)
	elif(dataset == "SmallDataset"):
		datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType2(datasetType2FileName, datasetClassColumnFirst)
		numberOfFeaturesPerWord = None
		paddingTagIndex = None
	elif(dataset == "wikiXmlDataset"):
		articles = ANNtf2_loadDataset.loadDatasetType4(datasetType4FileName, AEANNsequentialInputTypesMaxLength, useSmallSentenceLengths, AEANNsequentialInputTypeMinWordVectors)

	if(dataset == "wikiXmlDataset"):
		return articles
	else:
		return numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp
		


#trainMinimal is minimal template code extracted from train based on trainMultipleNetworks=False, trainMultipleFiles=False, greedy=False;
def trainMinimal():
	
	networkIndex = 1	#assert numberOfNetworks = 1
	fileIndexTemp = 0	#assert trainMultipleFiles = False
	
	#generate network parameters based on dataset properties:
	numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = loadDataset(fileIndexTemp)

	#Model constants
	num_input_neurons = datasetNumFeatures  #train_x.shape[1]
	num_output_neurons = datasetNumClasses

	learningRate, trainingSteps, batchSize, displayStep, numEpochs = defineTrainingParameters(dataset, numberOfFeaturesPerWord, paddingTagIndex)
	numberOfLayers = defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths, numberOfFeaturesPerWord)
	defineNeuralNetworkParameters()
														
	#stochastic gradient descent optimizer
	optimizer = tf.optimizers.SGD(learningRate)
	
	for e in range(numEpochs):

		print("epoch e = ", e)
	
		fileIndex = 0
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = loadDataset(fileIndex)

		shuffleSize = datasetNumExamples
		trainDataIndex = 0

		trainData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize, datasetNumClasses)
		trainDataList = []
		trainDataList.append(trainData)
		trainDataListIterators = []
		for trainData in trainDataList:
			trainDataListIterators.append(iter(trainData))
		testBatchX, testBatchY = generateTFbatch(test_x, test_y, batchSize)

		for batchIndex in range(int(trainingSteps)):
			(batchX, batchY) = trainDataListIterators[trainDataIndex].get_next()	#next(trainDataListIterators[trainDataIndex])
			batchYactual = batchY
					
			display = False
			if(batchIndex % displayStep == 0):
				display = True	
			trainBatch(e, batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display)

		pred = neuralNetworkPropagationTest(testBatchX, networkIndex)
		print("Test Accuracy: %f" % (calculateAccuracy(pred, testBatchY)))

			
#this function can be used to extract a minimal template (does not support algorithm==LREANN);
def train(trainMultipleNetworks=False, trainMultipleFiles=False, greedy=False):
	
	networkIndex = 1	#assert numberOfNetworks = 1
	fileIndexTemp = 0	#assert trainMultipleFiles = False
	
	#generate network parameters based on dataset properties:
	numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = loadDataset(fileIndexTemp)

	#Model constants
	num_input_neurons = datasetNumFeatures  #train_x.shape[1]
	num_output_neurons = datasetNumClasses

	learningRate, trainingSteps, batchSize, displayStep, numEpochs = defineTrainingParameters(dataset, numberOfFeaturesPerWord, paddingTagIndex)
	numberOfLayers = defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths, numberOfFeaturesPerWord)
	defineNeuralNetworkParameters()

	#configure optional parameters;
	if(trainMultipleNetworks):
		maxNetwork = numberOfNetworks
	else:
		maxNetwork = 1
	if(trainMultipleFiles):
		minFileIndex = fileIndexFirst
		maxFileIndex = fileIndexLast
	else:
		minFileIndex = 0
		maxFileIndex = 0
	if(greedy):
		maxLayer = numberOfLayers
	else:
		maxLayer = 1
														
	#stochastic gradient descent optimizer
	optimizer = tf.optimizers.SGD(learningRate)
	
	for e in range(numEpochs):

		print("epoch e = ", e)
	
		#fileIndex = 0
		#trainMultipleFiles code;
		if(randomiseFileIndexParse):
			fileIndexShuffledArray = generateRandomisedIndexArray(fileIndexFirst, fileIndexLast)
		for f in range(minFileIndex, maxFileIndex+1):
			if(randomiseFileIndexParse):
				fileIndex = fileIndexShuffledArray[f]
			else:
				fileIndex = f
				
			numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = loadDataset(fileIndex)

			shuffleSize = datasetNumExamples
			trainDataIndex = 0

			#greedy code [not used by LREANNtf, retained for cross compatibility];
			for l in range(1, maxLayer+1):
				print("l = ", l)
				trainData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize, datasetNumClasses)
				trainDataList = []
				trainDataList.append(trainData)
				trainDataListIterators = []
				for trainData in trainDataList:
					print("trainData index x")
					trainDataListIterators.append(iter(trainData))
				testBatchX, testBatchY = ANNtf2_operations.generateTFbatch(test_x, test_y, batchSize)
				#testBatchX, testBatchY = (test_x, test_y)

				for batchIndex in range(int(trainingSteps)):
					(batchX, batchY) = trainDataListIterators[trainDataIndex].get_next()	#next(trainDataListIterators[trainDataIndex])
					batchYactual = batchY
					
					for networkIndex in range(1, maxNetwork+1):
						display = False
						#if(l == maxLayer):	#only print accuracy after training final layer
						if(batchIndex % displayStep == 0):
							display = True	
						trainBatch(e, batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display, l)
						
					#trainMultipleNetworks code;
					if(trainMultipleNetworks):
						#train combined network final layer
						trainBatchAllNetworksFinalLayer(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, costCrossEntropyWithLogits, display)

				#trainMultipleNetworks code;
				if(trainMultipleNetworks):
					testBatchAllNetworksFinalLayer(testBatchX, testBatchY, datasetNumClasses, numberOfLayers)
				else:
					pred = neuralNetworkPropagationTest(testBatchX, networkIndex)
					if(greedy):
						print("Test Accuracy: l: %i, %f" % (l, calculateAccuracy(pred, testBatchY)))
					else:
						print("Test Accuracy: %f" % (calculateAccuracy(pred, testBatchY)))

							
def trainLRE():
																
	#generate network parameters based on dataset properties:

	networkIndex = 1
	
	fileIndexTemp = 0	
	numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = loadDataset(fileIndexTemp)

	#Model constants
	num_input_neurons = datasetNumFeatures  #train_x.shape[1]
	num_output_neurons = datasetNumClasses
	if(algorithm == "LREANN"):
		if(algorithmLREANN == "LREANN_expAUANN"):
			num_output_neurons = LREANNtf_algorithm.calculateOutputNeuronsLREANN_expAUANN(datasetNumClasses)

	learningRate, trainingSteps, batchSize, displayStep, numEpochs = defineTrainingParameters(dataset, numberOfFeaturesPerWord, paddingTagIndex)
	numberOfLayers = defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths, numberOfFeaturesPerWord)
	defineNeuralNetworkParameters()
									
	noisySampleGeneration = False
	if(algorithm == "LREANN"):
		noisySampleGeneration, noisySampleGenerationNumSamples, noiseStandardDeviation = LREANNtf_algorithm.getNoisySampleGenerationNumSamples()
		if(noisySampleGeneration):
			batchXmultiples = tf.constant([noisySampleGenerationNumSamples, 1], tf.int32)
			batchYmultiples = tf.constant([noisySampleGenerationNumSamples], tf.int32)
			randomNormal = tf.initializers.RandomNormal()	#tf.initializers.RandomUniform(minval=-1, maxval=1)

	#stochastic gradient descent optimizer
	optimizer = tf.optimizers.SGD(learningRate)

	for e in range(numEpochs):

		print("epoch e = ", e)

		fileIndex = 0

		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = loadDataset(fileIndex)

		testBatchX, testBatchY = ANNtf2_operations.generateTFbatch(test_x, test_y, batchSize)
		
		shuffleSize = datasetNumExamples

		#new iteration method (only required for algorithm == "LREANN_expAUANN/LREANN_expCUANN"):	
		datasetNumClassesActual = datasetNumClasses
		trainDataIndex = 0
		
		if(algorithm == "LREANN"):
			if(algorithmLREANN == "LREANN_expAUANN"):
				currentClassTarget = 0
				generateClassTargetExemplars = False
				if(e == 0):
					generateClassTargetExemplars = True
				networkIndex = 1 #note LREANNtf_algorithmLREANN_expAUANN doesn't currently support multiple networks
				trainDataList = LREANNtf_algorithm.generateTFtrainDataFromNParraysLREANN_expAUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses)
				exemplarDataList = LREANNtf_algorithm.generateTFexemplarDataFromNParraysLREANN_expAUANN(train_x, train_y, networkIndex, shuffleSize, batchSize, datasetNumClasses, generateClassTargetExemplars)
				testBatchY = LREANNtf_algorithm.generateYActualfromYLREANN_expAUANN(testBatchY, num_output_neurons)
				datasetNumClassTargets = datasetNumClasses
				datasetNumClasses = LREANNtf_algorithm.generateNumClassesActualLREANN_expAUANN(datasetNumClasses, num_output_neurons)
				exemplarDataListIterators = []
				for exemplarData in exemplarDataList:
					exemplarDataListIterators.append(iter(exemplarData))
			elif(algorithmLREANN == "LREANN_expCUANN"):
				trainDataList = LREANNtf_algorithm.generateTFtrainDataFromNParraysLREANN_expCUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses)
			elif(algorithmLREANN == "LREANN_expXUANN"):
				currentClassTarget = 0
				generateClassTargetExemplars = False
				if(e == 0):
					generateClassTargetExemplars = True
				trainDataList = LREANNtf_algorithm.generateTFtrainDataFromNParraysLREANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses, generatePositiveSamples=True)
				datasetNumClassTargets = datasetNumClasses
				samplePositiveDataList = LREANNtf_algorithm.generateTFtrainDataFromNParraysLREANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses, generatePositiveSamples=True)
				if(XUANNnegativeSamplesComplement):
					sampleNegativeDataList = LREANNtf_algorithm.generateTFtrainDataFromNParraysLREANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses, generatePositiveSamples=False)					
				elif(XUANNnegativeSamplesAll):
					#implementation limitation (sample negative contains a selection of experiences from all classes, not just negative classes) - this simplification deemed valid under assumptions: calculations will be averaged over large negative batch and numberClasses >> 2
					sampleNegativeData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)	#CHECKTHIS
					sampleNegativeDataList = []
					sampleNegativeDataList.append(sampleNegativeData)
				elif(XUANNnegativeSamplesRandom):
					sampleNegativeDataList = LREANNtf_algorithm.generateTFtrainDataFromNParraysLREANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses, generatePositiveSamples=True)					
				samplePositiveDataListIterators = []
				for samplePositiveData in samplePositiveDataList:
					samplePositiveDataListIterators.append(iter(samplePositiveData))
				sampleNegativeDataListIterators = []
				for sampleNegativeData in sampleNegativeDataList:
					sampleNegativeDataListIterators.append(iter(sampleNegativeData))
			else:
				trainData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)
				trainDataList = []
				trainDataList.append(trainData)		
		else:
			trainData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)
			trainDataList = []
			trainDataList.append(trainData)
		trainDataListIterators = []
		for trainData in trainDataList:
			trainDataListIterators.append(iter(trainData))

		#original iteration method:
		#trainData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize):	
		#for batchIndex, (batchX, batchY) in enumerate(trainData.take(trainingSteps), 1):	

		#new iteration method:			
		#print("trainingSteps = ", trainingSteps)
		#print("batchSize = ", batchSize)

		for batchIndex in range(int(trainingSteps)):
			(batchX, batchY) = trainDataListIterators[trainDataIndex].get_next()	#next(trainDataListIterators[trainDataIndex])
			
			batchYactual = batchY
			if(algorithm == "LREANN"):
				if(algorithmLREANN == "LREANN_expAUANN"):
					(exemplarsX, exemplarsY) = exemplarDataListIterators[trainDataIndex].get_next()
					batchYactual = LREANNtf_algorithm.generateTFYActualfromYandExemplarYLREANN_expAUANN(batchY, exemplarsY)
				if(algorithmLREANN == "LREANN_expXUANN"):
					(samplePositiveX, samplePositiveY) = samplePositiveDataListIterators[trainDataIndex].get_next()
					if(XUANNnegativeSamplesRandom):
						foundTrainDataIndexNegative = False
						while not foundTrainDataIndexNegative:
							trainDataIndexNegative = np.random.randint(0, datasetNumClasses)
							if(trainDataIndexNegative != trainDataIndex):
								foundTrainDataIndexNegative = True
						(sampleNegativeX, sampleNegativeY) = sampleNegativeDataListIterators[trainDataIndexNegative].get_next()
					else:
						(sampleNegativeX, sampleNegativeY) = sampleNegativeDataListIterators[trainDataIndex].get_next()

			if(noisySampleGeneration):
				if(batchSize != 1):	#batchX.shape[0]
					print("error: noisySampleGeneration && batchSize != 1")
					exit()
				batchX = tf.tile(batchX, batchXmultiples)
				batchY = tf.tile(batchY, batchYmultiples)
				batchXnoise = tf.math.multiply(tf.constant(randomNormal(batchX.shape), tf.float32), noiseStandardDeviation)
				batchX = tf.math.add(batchX, batchXnoise)
				#print("batchX = ", batchX)
				#print("batchY = ", batchY)

			predNetworkAverage = tf.Variable(tf.zeros(datasetNumClasses))

			#print("datasetNumClasses = ", datasetNumClasses)
			#print("batchX.shape = ", batchX.shape)
			#print("batchY.shape = ", batchY.shape)

			#can move code to trainBatchLRE();
			if(algorithm == "LREANN"):
				if(algorithmLREANN == "LREANN_expHUANN"):
					batchYoneHot = tf.one_hot(batchY, depth=datasetNumClasses)
					executeLearningLREANN(batchX, batchYoneHot, networkIndex)
				elif(algorithmLREANN == "LREANN_expSUANN"):
					executeLearningLREANN(batchX, batchY, networkIndex)
				elif(algorithmLREANN == "LREANN_expAUANN"):
					#learning algorithm embedded in forward propagation of new class x experience following forward propagation of existing class x experience
					executeLearningLREANN_expAUANN(batchX, batchY, exemplarsX, exemplarsY, currentClassTarget, networkIndex)
				elif(algorithmLREANN == "LREANN_expCUANN"):
					executeLearningLREANN(batchX, batchY, networkIndex)	#currentClassTarget
				elif(algorithmLREANN == "LREANN_expXUANN"):
					executeLearningLREANN_expXUANN(batchX, batchY, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, networkIndex)
				elif(algorithmLREANN == "LREANN_expMUANN"):
					executeLearningLREANN(batchX, batchY, networkIndex)
				elif(algorithmLREANN == "LREANN_expRUANN"):
					executeLearningLREANN(batchX, batchY, networkIndex)
				if(batchIndex % displayStep == 0):
					pred = neuralNetworkPropagation(batchX, networkIndex)
					loss = ANNtf2_operations.calculateLossCrossEntropy(pred, batchYactual, datasetNumClasses, costCrossEntropyWithLogits)
					acc = ANNtf2_operations.calculateAccuracy(pred, batchYactual)
					print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
					predNetworkAverage = predNetworkAverage + pred

			if(algorithm == "LREANN"):
				if(algorithmLREANN == "LREANN_expAUANN"):
					#batchYactual = LREANNtf_algorithm.generateTFYActualfromYandExemplarYLREANN_expAUANN(batchY, exemplarsY)
					currentClassTarget = currentClassTarget+1
					if(currentClassTarget == datasetNumClassTargets):
						currentClassTarget = 0
					trainDataIndex = currentClassTarget
				elif(algorithmLREANN == "LREANN_expXUANN"):
					currentClassTarget = currentClassTarget+1
					if(currentClassTarget == datasetNumClassTargets):
						currentClassTarget = 0
					trainDataIndex = currentClassTarget

		pred = neuralNetworkPropagationTest(testBatchX, networkIndex)
		print("Test Accuracy: networkIndex: %i, %f" % (networkIndex, ANNtf2_operations.calculateAccuracy(pred, testBatchY)))

def generateRandomisedIndexArray(indexFirst, indexLast, arraySize=None):
	fileIndexArray = np.arange(indexFirst, indexLast+1, 1)
	#print("fileIndexArray = " + str(fileIndexArray))
	if(arraySize is None):
		np.random.shuffle(fileIndexArray)
		fileIndexRandomArray = fileIndexArray
	else:
		fileIndexRandomArray = random.sample(fileIndexArray.tolist(), arraySize)
	
	print("fileIndexRandomArray = " + str(fileIndexRandomArray))
	return fileIndexRandomArray
	
				
if __name__ == "__main__":
	if(algorithm == "LREANN"):
		trainLRE()
			
