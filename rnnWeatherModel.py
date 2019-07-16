#Use of any data, graphs, tables, maps or other products obtained through Alberta 
#Agriculture and Forestry's (AF) Alberta Climate Information Service (ACIS), whether
#direct or indirect, must be fully acknowledged and/or cited. This includes, but is 
#not limited to, all published, electronic or printed documents such as articles, 
#publications, internal reports, external reports, research papers, memorandums, 
#news reports, radio or print. Proper citation (subject to the documents' citing 
#style) includes: "Data provided by Alberta Agriculture and Forestry, Alberta 
#Climate Information Service (ACIS) https://agriculture.alberta.ca/acis (month and
#year when data was retrieved)" If the document contains an acknowledgment section, 
#then it must be noted that data was provided by the Alberta Climate Information 
#Service, found at https://agriculture.alberta.ca/acis.

#Imports
from __future__ import absolute_import, division, print_function, unicode_literals

#Imports for loading and saving files
import pathlib
import os
import pickle
import re

#For reading csv files
import numpy as np
import pandas as pd

#For plots
import matplotlib.pyplot as plt
import statistics
import time
import math
from numpy.polynomial.polynomial import polyfit
import matplotlib.ticker as plticker

#For ML
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras.losses

#For Dates
import datetime as dt
from datetime import timedelta
from datetime import date

#Get an input from the user in the range of [lowerLimit, upperLimit]
def getUserInput(lowerLimit, upperLimit, prompt):
	while(True):
		try:
			print(prompt)
			userInput = input()
			if int(userInput) >= lowerLimit and int(userInput) <= upperLimit:
				return int(userInput)
		except ValueError:
			print("Invalid Input")

#Load the input data and preprocess it
def loadData():
	#Choose whether or not to load a previously preprocessed dataframe or create and save a new one
	print("Enter in 1 if you wish to load an existing reformatted csv file, and 2 if any new csv files must be formatted")
	userChoice = getUserInput(1, 2, "Enter in a number between 1 and 2:")
	loadPreProcDataSetting = userChoice == 1

	#Path for readable csv file
	preProcCSVPath = r"PreprocessedDatasets\weatherDf.csv"
	#Path for functional dataframe pickled file
	preProcPklPath = r"PreprocessedDatasets\weatherDf.pickle"

	if loadPreProcDataSetting == False:
		#Get the input data and preprocess it	
		print("Loading weather data...")
		startTime = time.time()
		
		dataFolderPath = r"WeatherDatasets\Vulcan_Hourly_2012_2018\\"
		weatherDf = downloadData(dataFolderPath)
		weatherDf = formatData(weatherDf)

		#Get the data for the growing season
		weatherGSDf = parseGSData(weatherDf, loadOld=False)

		#Remove data from the growing season
		gsStartDate = '2018-04-01 00:00:00'
		i = 0
		gsStartDateIndex = -1
		for index in list(weatherDf.index):
			if str(index) == gsStartDate:
				gsStartDateIndex = i
				break
			i = i+1
		
		weatherDf = weatherDf[:gsStartDateIndex]	

		#Save the preprocessed data
		weatherDf.to_csv(preProcCSVPath)
		weatherDf.to_pickle(preProcPklPath)
		print("Weather data saved to {} and {}. ".format(preProcCSVPath, preProcPklPath) + timeEndTime(startTime))
	else:
		print("Loading preformatted weather data...")
		startTime = time.time()
		weatherDf = pd.read_pickle(preProcPklPath)
		
		#Get the data for the growing season
		weatherGSDf = parseGSData(weatherDf, loadOld=True)
		
		#Remove data from the growing season
		gsStartDate = '2018-04-01 00:00:00'
		i = 0
		gsStartDateIndex = -1
		for index in list(weatherDf.index):
			if str(index) == gsStartDate:
				gsStartDateIndex = i
				break
			i = i+1
		
		weatherGSDf = weatherGSDf[:gsStartDateIndex]	
		
		print("Weather data loaded from {}. ".format(preProcPklPath) + timeEndTime(startTime))

	return [weatherDf, weatherGSDf]

#Reformat the data for the final forecast target
#return the numpy array form of the growing season data.
def parseGSData(gsDf, loadOld=False):
	
	startTime = time.time()
	#Path for readable csv file
	preProcCSVPath = r"PreprocessedDatasets\weatherGSDf.csv"
	#Path for functional dataframe pickled file
	preProcPklPath = r"PreprocessedDatasets\weatherGSDf.pickle"

	if not loadOld or not os.path.isfile(preProcCSVPath) or not os.path.isfile(preProcPklPath):
		#Limit the data to the growing season
		startDate = '2018-04-01 00:00:00'
		endDate = '2018-09-30 23:00:00'
		i = 0
		startDateIndex = -1
		endDateIndex = -1
		for index in list(gsDf.index):
			if str(index) == startDate:
				startDateIndex = i
			if str(index) == endDate:
				endDateIndex = i
				break
			i = i+1
		
		gsDf = gsDf[startDateIndex:endDateIndex+1]		
		#Save the preprocessed data
		gsDf.to_csv(preProcCSVPath)
		gsDf.to_pickle(preProcPklPath)
		print("Growing Season data saved to {} and {}. ".format(preProcCSVPath, preProcPklPath) + timeEndTime(startTime))
	else:
		print("Loading preformatted growing season data...")
		gsDf = pd.read_pickle(preProcPklPath)
		print("Growing Season data loaded from {}. ".format(preProcPklPath) + timeEndTime(startTime))
		
	return gsDf
	
#Get a unique list of each element in the given list
def getUniqueElems(givenList):
	uniqueList = []
	for elem in givenList:
		if elem not in uniqueList:
			uniqueList.append(elem)
	return uniqueList

#Download and concatenate input csv data from a given data folder, 
#and returns the dataframe
def downloadData(dataFolderPath):
	readCSVList = []
	for subfolder in os.listdir(dataFolderPath):	
		for filename in sortedFilesInFolder(dataFolderPath + subfolder):
			fullPathFilename = dataFolderPath + subfolder + "\\" + filename
			readCSVList.append(pd.read_csv(fullPathFilename, encoding = 'unicode_escape', skiprows = 15))
			readCSVList[len(readCSVList)-1]["Station Name"] = [subfolder for i in range(len(readCSVList[len(readCSVList)-1].index))]
		print("Subfolder: {} read.".format(subfolder))
	
	csvDataframe = pd.concat(readCSVList, ignore_index = True, sort=False)
	
	return csvDataframe

#Returns a list of file paths given a folder path, sorted by timestamps
def sortedFilesInFolder(directoryPath):
    mTime = lambda f: os.stat(os.path.join(directoryPath, f)).st_mtime
    return list(sorted(os.listdir(directoryPath), key=mTime))


#Format the dataframe for normalization and splitting
def formatData(weatherDf):
	#Sort the concatenated dataframe by the Station Name
	#use mergesort because mergesort here is stable, preserving the ordering
	#of station and then date
	weatherDf = weatherDf.sort_values("Station Name", kind = 'mergesort')

	#Delete the columns of data where there is lots of missing data
	weatherDf = weatherDf.dropna(axis = 'columns', thresh = 10000)
	#Delete String columns and derived values like 'Hmdx' and 'Wind Chill'
	uselessColHeaders = ['Wind Dir (10s deg)', 'Wind Dir Flag', 'Wind Spd Flag', 'Stn Press Flag', 'Hmdx', 'Wind Chill']
	for header in uselessColHeaders:
		weatherDf.pop(header)
	weatherDfHeaders = list(weatherDf.columns)
	
	#Get the Station Names
	stationNameList = getUniqueElems(weatherDf["Station Name"])
	weatherDf.pop("Station Name")
	
	#Get the original header columns
	originalHeaderList = weatherDfHeaders[5:len(weatherDfHeaders)-1]

	#Set the index to be the date column 
	addZero = lambda a : "0" + str(a) if a < 10 else a 
	dateCol = ["{}-{}-{}T{}".format(year, addZero(month), addZero(day), time) for year, month, day, time in zip(weatherDf["Year"], weatherDf["Month"], weatherDf["Day"], weatherDf["Time"])]
	dateCol = np.array(dateCol, dtype = 'datetime64')
	weatherDf.set_index(dateCol, inplace = True)
	
	#remove all other date information
	weatherDf.pop("Date/Time")
	weatherDf.pop("Year")
	weatherDf.pop("Month")
	weatherDf.pop("Day")
	weatherDf.pop("Time")
	
	#Parse each weather station's data into a list of dataframes
	earlyDate = weatherDf.index[0]
	lateDate = weatherDf.index[len(weatherDf.index) - 1]
	
	earlyDateIndexList = []
	lateDateIndexList = []
	
	for i in range(len(weatherDf.index)):
		if weatherDf.index[i] == earlyDate:
			earlyDateIndexList.append(i)
		if weatherDf.index[i] == lateDate:
			lateDateIndexList.append(i)
			
	stationFrameList = []
	for earlyDateIndex, lateDateIndex in zip(earlyDateIndexList, lateDateIndexList):
		stationFrameList.append(weatherDf.iloc[range(earlyDateIndex, lateDateIndex + 1),:])
	
	#Compile each station data into the same row
	weatherDfNewHeader = pd.MultiIndex.from_product([stationNameList,originalHeaderList])
	
	weatherDfNew = pd.DataFrame(index = weatherDf.index[earlyDateIndexList[0]:lateDateIndexList[0]+1],
									   columns = weatherDfNewHeader)	
	
	for stationName, stationFrame in zip(stationNameList, stationFrameList):
		for ogHeader in originalHeaderList:
			weatherDfNew[stationName, ogHeader] = stationFrame[ogHeader]

	#Add Date data to the input data for all stations
	weatherDfNew['Date', 'Year'] = weatherDfNew.index.year
	weatherDfNew['Date', 'Day'] = weatherDfNew.index.dayofyear
	weatherDfNew['Date', 'Hour'] = weatherDfNew.index.hour

	#Delete the earliest rows of missing temperature values
	i = 0
	while np.isnan(weatherDfNew[stationNameList[0]][originalHeaderList[0]][i]) or np.isnan(weatherDfNew[stationNameList[3]][originalHeaderList[0]][i]) or np.isnan(weatherDfNew[stationNameList[3]][originalHeaderList[3]][i]):
		i = i+1	
	weatherDfNew = weatherDfNew.iloc[i:]
	
	#Delete the columns with any missing data
	weatherDfNew = weatherDfNew.dropna(axis='columns', thresh=40000)
	
	#Interpolate the missing values forward with linear interpolation
	weatherDfNew = weatherDfNew.interpolate(method ='linear', limit_direction ='forward') 

	#View the information about the final preprocessed data file
	#print(weatherDfNew.info())
	
	return weatherDfNew
	
#Get the Station names and the names of the individual headers for each station
def getStationNamesAndDataFields(weatherDf):
	skipPrimaryHeaderList = ['Date']
	
	primaryCols = [a[0] for a in list(weatherDf.columns) if a[0] not in skipPrimaryHeaderList]
	stationNameList = getUniqueElems(primaryCols)
	secondaryCols = [a[1] for a in list(weatherDf.columns) if a[0] not in skipPrimaryHeaderList]
	headerList = getUniqueElems(secondaryCols)
	
	return [stationNameList, headerList]
	
#determine the difference between the current time and the given start time
def timeEndTime(startTime):
	endTime = time.time()
	deltaTime = endTime - startTime
	if deltaTime % 60 < 1:
		timeString = "Time: {:5.3f} milliseconds.".format((deltaTime%60)*1000)
	else:
		timeString = "Time: {} minutes, {:5.3f} seconds.".format(math.floor(deltaTime/60.0), deltaTime % 60)
	
	return timeString

#Get the target header from the user
def getTargetHeader(headerList):
	for i in range(len(headerList)):
		print("i = {}: Header = {}".format(i, headerList[i]))
	headerIndex = getUserInput(0, len(headerList) -1,
						   "Please enter in the header index of the data you wish to forecast:")
	return headerList[headerIndex]

#Get the number of days to forecast in advance
def getForecastDayDifference():
	print("Please enter in how many days in advance you wish to forecast data:")
	return getUserInput(1, 365, "Enter in a number between 1 and 365")
	
#Create a dataframe of the target data.
#Data that is too recent still exists.
def makeTargetDf(weatherDf, targetHeader, foreDaysDiff):
	#Create the target columns
	tempWeatherDf = weatherDf.copy()
	stationNamesList = getStationNamesAndDataFields(weatherDf)[0]
	forecastHeader = 'Forecasted {}: {} Days in Advance'.format(targetHeader, foreDaysDiff)
	for stationName in stationNamesList:
		tempWeatherDf[forecastHeader, stationName] = weatherDf[stationName][targetHeader].shift(-foreDaysDiff)
	
	#Remove all columns but the target columns
	primaryHeaderList = getStationNamesAndDataFields(tempWeatherDf)[0]
	for primaryHeader in primaryHeaderList[:-1]:
		tempWeatherDf.pop(primaryHeader)
	tempWeatherDf.pop('Date')	
	
	#Get the list of dates that the predictions correspond to.
	#Dates of the data that is too recent will be extrapolated
	forecastDates = tempWeatherDf.index[foreDaysDiff:]
	endDate = np.datetime64(forecastDates[len(forecastDates)-1])
	generatedDates = [endDate + np.timedelta64(i+1, 'D') for i in range(foreDaysDiff)]
	forecastDates=forecastDates.union(pd.Index(generatedDates))
	tempWeatherDf.set_index(forecastDates, inplace=True)
		
	return tempWeatherDf

#Convert the input and target dataframes into numpy arrays.
#Get the normalized x train and test sets, the y train and test sets
#and the most recent salvaged x forecast prediction data
#Also get the scalers
def xyDataParse(weatherDf, labeledTargetDf, foreDaysDiff):
	#Get the dates of the y data
	yDates = labeledTargetDf.index[:-foreDaysDiff]
	yDatesForecast = labeledTargetDf.index[:]

	#Get the numpy arrays of the x and y data
	xData = weatherDf.values[:-foreDaysDiff]
	xDataForecast = weatherDf.values[:]
	yData = labeledTargetDf.values[:-foreDaysDiff]
	
	#Define the training-test split fraction and actual length
	trainTestSplitFrac = 0.8
	trainSplitQuant = int(trainTestSplitFrac * len(xData))
	 
	#Create the training-test split
	#Training data will be defined as the oldest data
	#Testing data will be defined as the newest data
	xDataTrain = xData[:trainSplitQuant]
	xDataTest = xData[trainSplitQuant:]
	yDataTrain = yData[:trainSplitQuant]
	yDataTest = yData[trainSplitQuant:]
	yDatesTrain = yDates[:trainSplitQuant]
	yDatesTest = yDates[trainSplitQuant:]

	#Scale the data using the sklearn MinMaxScaler
	xScaler = MinMaxScaler()
	xDataTrain = xScaler.fit_transform(xDataTrain)
	xDataTest = xScaler.transform(xDataTest)
	xDataForecast = xScaler.transform(xDataForecast)
	
	yScaler = MinMaxScaler()
	yDataTrain = yScaler.fit_transform(yDataTrain)
	yDataTest = yScaler.transform(yDataTest)
	
	return [xDataTrain, xDataTest, yDataTrain, yDatesTrain, yDataTest, yDatesTest, xScaler, yScaler, xDataForecast, yDatesForecast]

#Create the batch generator
#Will pick a batch of random shorter continuous time series from the training data
def batchGenerator(batchSize, timeSeriesLength, xDataTrain, yDataTrain):
    #Generator function for creating random batches of training data.
	while True:
		#Allocate arrays for the x and y batches
		xBatchShape = (batchSize, timeSeriesLength, xDataTrain.shape[1]) 
		xBatch = np.zeros(shape = xBatchShape)
		
		yBatchShape = (batchSize, timeSeriesLength, yDataTrain.shape[1]) 
		yBatch = np.zeros(shape = yBatchShape)
		
		#Create the batch with random time sequences
		for i in range(batchSize):
			#Get a random start index for the time series in the training data
			timeSeriesStartIndex = np.random.randint(xDataTrain.shape[0] - timeSeriesLength)
			
			#Copy the time sequence into the batch
			xBatch[i] = xDataTrain[timeSeriesStartIndex:timeSeriesStartIndex+timeSeriesLength]
			yBatch[i] = yDataTrain[timeSeriesStartIndex:timeSeriesStartIndex+timeSeriesLength]
		
		yield(xBatch, yBatch)

#Load or create a trained weather model
def getModel(xDataTrain, yDataTrain, batchGen, validationData):
	#Choose whether or not to load a previously trained model
	print("Enter in 1 if you wish to load existing model h5 files, 2 if any new models must be created, and 3 to resume training.")
	loadModelSetting = getUserInput(1, 3, "Enter in a number between 1 and 3:")
	
	#Paths for saving models and training history
	modelFilePath = 'WeatherModels\CurrentCheckpoints\save_{epoch:04d}_weatherModel_' + targetHeader.replace(' ','_') + '_Forecast_' + str(foreDaysDiff) + '.h5'
	trainingHistoryFilePath = r'WeatherModels\CurrentCheckpoints\trainingHistory_' + re.sub('[^A-Za-z0-9\_]', '-', targetHeader) + '_Forecast_' + str(foreDaysDiff) + '.csv'
	
	#Build model
	weatherModel = buildModel(xDataTrain.shape[1], yDataTrain.shape[1])
	#Create the callbacks
	callbackFuncList = createModelCallbackFunctions(modelFilePath)
	
	#Set the number of epochs
	custEpochs = 500
	
	#Load model if it exists
	if loadModelSetting == 1 or loadModelSetting == 3:
		print("Enter in the checkpoint number of the model you wish to load:")
		checkpointNum = getUserInput(1, custEpochs, "Enter in a number between 1 and {}.".format(custEpochs))
		modelFilePath = 'WeatherModels\CurrentCheckpoints\save_'+str(checkpointNum).zfill(4)+'_weatherModel_' + targetHeader.replace(' ','_') + '_Forecast_' + str(foreDaysDiff) + '.h5'
		
		if os.path.isfile(modelFilePath) and os.path.isfile(trainingHistoryFilePath):
			weatherModel = keras.models.load_model(modelFilePath, custom_objects={'lossMSEWithWarmup':lossMSEWithWarmup})
			trainingHistoryDf = pd.read_csv(trainingHistoryFilePath)
			print("weather model loaded from: ", modelFilePath)
	#Else build the model
	if loadModelSetting == 2 or loadModelSetting == 3:
		#Use 100 generated batches per epoch
		custStepsPerEpoch = 10
		
		#Train the model
		startTime = time.time()
		trainingHistory = weatherModel.fit_generator(
										generator = batchGen,  
										epochs = custEpochs,
										steps_per_epoch = custStepsPerEpoch,
										validation_data = validationData,
										callbacks = callbackFuncList)
										
		print("Model training complete. " + timeEndTime(startTime))
	
		#Record training history	
		trainingHistoryDf = pd.DataFrame(trainingHistory.history)
		trainingHistoryDf['epoch'] = trainingHistory.epoch
		
		print("Saving training history...")
		trainingHistoryDf.to_csv(trainingHistoryFilePath)
	
	#Display training history	
	print("\nEnd training loss values:")
	print(trainingHistoryDf.tail())
	plotHistory(trainingHistoryDf, targetHeader)
	
	return weatherModel


#Build the model
def buildModel(inputDimension, outputDimension):
	#Model is a keras sequential model
	weatherModel = keras.Sequential()
	
	#The GRU output dimensions are 256 days in a batch time sequence, 
	#the activation is the default tanh, and return the full sequence.
	weatherModel.add(layers.GRU(units=256, return_sequences = True,
								input_shape = (None, inputDimension,)))
	weatherModel.add(layers.Dropout(0.1))
	#The GRU output dimensions are 256 days in a batch time sequence, 
	#the activation is the default tanh, and return the full sequence.
	weatherModel.add(layers.GRU(units=256, return_sequences = True,
								input_shape = (None, inputDimension,)))
	weatherModel.add(layers.Dropout(0.15))
	#The GRU output dimensions are 256 days in a batch time sequence, 
	#the activation is the default tanh, and return the full sequence.
	weatherModel.add(layers.GRU(units=256, return_sequences = True,
								input_shape = (None, inputDimension,)))
	weatherModel.add(layers.Dropout(0.2))							
	#Output layer is a dense layer.
	weatherModel.add(layers.Dense(outputDimension, activation='tanh'))
	
	#Optimizer
	weatherOptimizer = keras.optimizers.Adam(lr=1e-3)
	
	#Compile the model
	weatherModel.compile(loss = lossMSEWithWarmup, optimizer = weatherOptimizer)
	
	#Print the model summary
	print(weatherModel.summary())
	
	return weatherModel

#Create a custom loss function, which will be MSE but with a warmup period of 
#50 days where we ignore its errors.
#The warmup is for when the model has only seen a few time-steps of input data
def lossMSEWithWarmup(yTrueLabels, yPredictions):
    # The shape of both input tensors are:
    #[batchSize, timeSeriesLength, yDataTrain.size[0]].
	
    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
	warmupDays = 48
	yPredictionConsider = yPredictions[:, warmupDays:, :]
	yLabelsConsider = yTrueLabels[:, warmupDays:, :]

    # These sliced tensors both have this shape:
    # [batchSize, timeSeriesLength - warmupDays, yDataTrain.size[0]]

	# Calculate the MSE loss for each value in these tensors.
	# This outputs a 3-rank tensor of the same shape.
	loss = tf.compat.v1.losses.mean_squared_error(labels=yLabelsConsider,
												  predictions=yPredictionConsider)

	# Keras may reduce this across the first axis (the batch)
	# but the semantics are unclear, so to be sure we use
	# the loss across the entire tensor, we reduce it to a
	# single scalar with the mean function.
	lossMean = tf.reduce_mean(loss)

	return lossMean

#Create the callback functions of the model
def createModelCallbackFunctions(modelSavePath):
	#Callback for early stopping
	earlyStopPatience = 25
	callbackEarlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = earlyStopPatience)
	
	#Callback for checkpointing/model saving. Only the weights are saved.
	callbackCheckpoint = keras.callbacks.ModelCheckpoint(filepath=modelSavePath, monitor = 'val_loss')
	
	#Callback for reducing the learning rate as validation loss fails to decrease
	#callbackReduceLearnRate = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor = 0.1, min_lr = 1e-4, patience = 3)
	
	#Compile callback list
	callbackList = [callbackEarlyStop, callbackCheckpoint]
	#, callbackReduceLearnRate]
	
	return callbackList
	
#Plot the Error vs Training Epoch for the model
def plotHistory(histDf, targetHeader):	
	
	#Create the graph for the trainging and value MAE
	plt.figure(dpi = 120)
	plt.xlabel('Epoch')
	plt.ylabel('(MinMax Scaled) Mean Abs Error Ignoring Warmup [{}]'.format(targetHeader))
	plt.plot(histDf['epoch'], histDf['loss'], label = 'Train MSE', linewidth = 1, )
	plt.plot(histDf['epoch'], histDf['val_loss'], label = 'Val MSE', linewidth = 1)
	plt.ylim(0, max(max(histDf['loss'][math.floor(len(histDf['loss'])*0.1):]), 
				max(histDf['val_loss'][math.floor(len(histDf['loss'])*0.1):]), 
				max(max(histDf['loss']), max(histDf['val_loss']))*0.01))
	
	plt.legend()
	
	#Show the graphs
	plt.show()
	plt.close()

#Evaluate the performance of the model on the test data
def evaluateModel(model, xDataTest, yDataTest):
	#Evaluate the final results
	lossResult = model.evaluate(x=np.expand_dims(xDataTest, axis=0),
								y=np.expand_dims(yDataTest, axis=0))	
	print("loss (MSE of minmaxScaled test set ignoring warmup period):", lossResult)
	
#Generate the plots of prediction and actual weather for each station as well as
#the average of all stations.
def plotPredAndRealWeather(model, xInputData, yLabelData, yScaler, stationNameList, dateList, warmupDays=48):
	startTime = time.time()

	#Input data for the model.
	xData = np.expand_dims(xInputData, axis=0)

	#inverse scale the output weather
	yData = yScaler.inverse_transform(yLabelData) 
		
	# Use the model to predict the output-signals.
	yPredictions = model.predict(xData)

	# The output of the model is scaled by the y scaler
	# Do an inverse map to get it back to the scale
	# of the original data-set.
	yPredictions = yScaler.inverse_transform(yPredictions[0])
		
	#Add the data fields
	yData = addAverageField(yData)
	yPredictions = addAverageField(yPredictions)
	
	stationAverageLabel = "Station Average"
	if stationAverageLabel not in stationNameList:
		stationNameList.append(stationAverageLabel)
		
	# For each output-signal.
	for stationNameIndex in range(len(stationNameList))[-1:]:
		# Get the output predictions from the model.
		stationPrediction = yPredictions[:, stationNameIndex]
		
		# Get the true labels from the data-set.
		stationTrueWeather = yData[:, stationNameIndex]
		
		#Plot how the predicted temperatures vs the actual truth
		printEgregiousPredictions(stationTrueWeather, stationPrediction)
		calculateErrorWithWarmup(stationTrueWeather, stationPrediction, warmupDays)
		plotPredVsTrueScatter(stationTrueWeather, stationPrediction)
		plotErrDistr(stationTrueWeather, stationPrediction)
		plotPredAndTruthTimeline(stationTrueWeather, stationPrediction, stationNameList[stationNameIndex], dateList)
	print("Evaluation complete. " + timeEndTime(startTime))
	
#Add an average column to a numpy array
def addAverageField(arr):
	#Create the average list
	aveList = []
	for row in arr:
		sum = 0
		for a in row:
			sum = sum + a
		ave = sum/len(row)
		aveList.append(ave)
		
	aveList = np.array([aveList])
	aveList = aveList.transpose()
	
	#Concatenate the arrays
	arr = np.concatenate([arr, aveList], axis=1)
	return arr

#Calculate the MAE and MSE of a set
def calculateErrorWithWarmup(testLabels, testPredictions, warmup):
	#These will be predictions with an AE > egregiousDifference degrees C
	warmupCount = 0
	sum = 0
	squaredSum = 0
	for testPred, testLab in zip(testPredictions, testLabels):
		if warmupCount < warmup:
			warmupCount = warmupCount + 1
			continue
		else:
			sum = sum + abs(testPred - testLab)
			squaredSum = squaredSum + (testPred - testLab)**2
	mae = sum / len(testLabels)
	mse = squaredSum / len(testLabels)
	print("MAE = {:5.3f}, MSE = {:5.3f}".format(mae, mse))


#Print the most egregious model predictions
def printEgregiousPredictions(testLabels, testPredictions):
	#These will be predictions with an AE > egregiousDifference degrees C
	egregiousDifference = 25
	print("\nThe predictions with an AE > {} degrees C".format(egregiousDifference))
	egregiousPredictionsCount = 0
	for testPred, testLab in zip(testPredictions, testLabels):
		if abs(testPred - testLab) > egregiousDifference:
			egregiousPredictionsCount = egregiousPredictionsCount + 1
			print("prediction = {:5.3f},	true value = {:5.3f},	MAE = {:5.3f}".format(
				   testPred, testLab, abs(testPred - testLab)))
	print("Above is Displaying {} of {} ({:5.3f}%) test predictions.".format(
		  egregiousPredictionsCount, len(testPredictions), egregiousPredictionsCount/len(testPredictions)*100))

#Plot the test labels vs the test predictions
def plotPredVsTrueScatter(testLabels, testPredictions):
	plt.scatter(testLabels, testPredictions, s=10)
	plt.xlabel("True Values [{}]".format(targetHeader).replace(".", " "))
	plt.ylabel("Predictions [{}]".format(targetHeader).replace(".", " "))
	plt.axis('equal')
	plt.axis('square')	
	
	xRange = (plt.xlim()[1] - plt.ylim()[0])
	yRange = (plt.ylim()[1] - plt.ylim()[0])
	
	plt.xlim(plt.xlim()[0] - xRange*0.05, plt.xlim()[1] + xRange*0.05)
	plt.ylim(plt.ylim()[0] - yRange*0.05, plt.ylim()[1] + yRange*0.05)
	#Plots the linear line of best fit
	_ = plt.plot([-30, 30],
				 [-30, 30], color='green')	
	b, m = polyfit(testLabels, testPredictions, 1)
	print("b = {}, m = {} for linear regression of predicted values vs true values".format(b, m))
	plt.plot(testLabels, [b+m*tlabel for tlabel in testLabels], '-', color='red')
	plt.show()
	plt.close()
	
	return (b, m)

#Plot the error distribution of some test predictions
def plotErrDistr(testLabels, testPredictions):
	error = [testPred - testLab for testPred, testLab in zip(testPredictions, testLabels)]
	plt.hist(error, bins = 50, rwidth = 0.92)
	plt.xlabel("Prediction Error (Prediction - True Value)[{}]".format(targetHeader.replace(".", " ")))
	_ = plt.ylabel("Count (No. of Predictions)")
	plt.show()
	plt.close()
	
#Plot the predicted weather forecast and the actual weather on a timeline
def plotPredAndTruthTimeline(stationTrueWeather, stationPrediction, stationName,dateList, warmupDays = 48):
	# Make the plotting-canvas bigger.
	#plt.figure(figsize=(14,5))

	#Get the list of dates in string yyyy-mm-dd format
	dateList = ["{}-{}-{}".format(str(date)[:4], str(date)[5:7], str(date)[8:10]) for date in list(dateList)]
	
	fig, ax = plt.subplots()

	# Plot and compare the two signals.
	ax.plot(dateList, stationTrueWeather, label='True Weather')
	ax.plot(dateList, stationPrediction, label='Predicted Weather')

	# this locator puts ticks at regular intervals
	loc = plticker.MultipleLocator(base=7.0) 
	ax.xaxis.set_major_locator(loc)
	
	fig.autofmt_xdate()
	
	# Plot grey box for warmup-period.
	p = plt.axvspan(0, warmupDays, facecolor='black', alpha=0.15)

	# Plot labels etc.
	plt.ylabel(stationName)
	plt.legend()
	plt.show()
	plt.close()

#Forecast and evaluate the data for the desired growing season.
def forecastGS(model, xDataForecast, gsDf, yScaler, stationNameList, warmupDays=0):	
	startTime = time.time()
	
	#Input data for the model.
	xData = np.expand_dims(xDataForecast, axis=0)
	
	#Convert the growing season dataframe into a numpy array of the same format as yPredictions
	growSeasonArr = gsDf.values
	growSeasonLength = len(growSeasonArr)
	yData = growSeasonArr
	
	# Use the model to predict the output-signals.
	yPredictions = model.predict(xData)

	# The output of the model is scaled by the y scaler
	# Do an inverse map to get it back to the scale
	# of the original data-set.
	yPredictions = yScaler.inverse_transform(yPredictions[0])
	
	#Only plot and analyze the data of the growing season
	yPredictions = yPredictions[-growSeasonLength:]
	dateList = gsDf.index
	
	#Add the data fields
	yData = addAverageField(yData)
	yPredictions = addAverageField(yPredictions)
	
	stationAverageLabel = "Station Average"
	if stationAverageLabel not in stationNameList:
		stationNameList.append(stationAverageLabel)
	
	print("Model evaluation complete. " + timeEndTime(startTime))
	
	# For each output-signal.
	for stationNameIndex in range(len(stationNameList))[-1:]:
		# Get the output predictions from the model.
		stationPrediction = yPredictions[:, stationNameIndex]
		
		# Get the true labels from the data-set.
		stationTrueWeather = yData[:, stationNameIndex]
		
		#Plot how the predicted temperatures vs the actual truth
		printEgregiousPredictions(stationTrueWeather, stationPrediction)
		calculateErrorWithWarmup(stationTrueWeather, stationPrediction, warmupDays)
		plotPredVsTrueScatter(stationTrueWeather, stationPrediction)
		plotErrDistr(stationTrueWeather, stationPrediction)
		plotPredAndTruthTimeline(stationTrueWeather, stationPrediction, stationNameList[stationNameIndex], dateList, warmupDays = warmupDays)
		printCustStats(stationTrueWeather, stationPrediction)
		np.savetxt(r"WeatherModelsPredictions\stationHourlyPredictions_F{}.csv".format(4392),
				   np.column_stack((stationPrediction, stationPrediction)), delimiter=",", fmt='%s', header='Hourly Data, Hourly Data')

#Forecast and evaluate the data for the desired growing season, using daily averages
def forecastGSMin(model, xDataForecast, gsDf, yScaler, stationNameList, warmupDays=0):	
	startTime = time.time()
	
	#Input data for the model.
	xData = np.expand_dims(xDataForecast, axis=0)
	
	#Convert the growing season dataframe into a numpy array of the same format as yPredictions
	growSeasonArr = gsDf.values
	growSeasonLength = len(growSeasonArr)
	yData = growSeasonArr
	
	# Use the model to predict the output-signals.
	yPredictions = model.predict(xData)

	# The output of the model is scaled by the y scaler
	# Do an inverse map to get it back to the scale
	# of the original data-set.
	yPredictions = yScaler.inverse_transform(yPredictions[0])
	
	#Only plot and analyze the data of the growing season
	yPredictions = yPredictions[-growSeasonLength:]
	dateList = gsDf.index
		
	#Add the data fields
	yData = addAverageField(yData)
	yPredictions = addAverageField(yPredictions)
	
	stationAverageLabel = "Station Average"
	if stationAverageLabel not in stationNameList:
		stationNameList.append(stationAverageLabel)
	
	
	#Calculate and use the average daily min
	yData = dailyAttribute([row[len(yData[0])-1] for row in yData], attribute='min')
	yPredictions = dailyAttribute([row[len(yPredictions[0])-1] for row in yPredictions], attribute='min')
	dateList = dailyAttribute(dateList, attribute='date')
	
	stationMinLabel = "Station Average Daily Minimum"
	if stationMinLabel not in stationNameList:
		stationNameList.append(stationMinLabel)
	
	print("Model evaluation complete. " + timeEndTime(startTime))
	
	# For each output-signal.
	for stationNameIndex in range(len(stationNameList))[-1:]:
		# Get the output predictions from the model.
		stationPrediction = yPredictions
		
		# Get the true labels from the data-set.
		stationTrueWeather = yData
		
		#Plot how the predicted temperatures vs the actual truth
		printEgregiousPredictions(stationTrueWeather, stationPrediction)
		calculateErrorWithWarmup(stationTrueWeather, stationPrediction, warmupDays)
		plotPredVsTrueScatter(stationTrueWeather, stationPrediction)
		plotErrDistr(stationTrueWeather, stationPrediction)
		plotPredAndTruthTimeline(stationTrueWeather, stationPrediction, stationNameList[stationNameIndex], dateList, warmupDays = warmupDays)
		printCustStats(stationTrueWeather, stationPrediction)
		np.savetxt(r"WeatherModelsPredictions\stationDailyMinPredictions_F{}.csv".format(4392),
				   np.column_stack((stationPrediction, stationPrediction)), delimiter=",", fmt='%s', header='Daily Min Temp, Daily Min Temp')
		

#Return a list of daily minimums from the given numpy array of hourly average values
def dailyAttribute(hourlyArr, attribute='ave'):
		
	attFunction = {'min': lambda x: min(x),
				   'ave': lambda x: ave(x),
				   'max': lambda x: max(x),
				   'date': lambda x: date(x[0].year, x[0].month, x[0].day)}
	dailyMinList = []
	i = 1
	j = 0
	while i < len(hourlyArr):
		if i % 24 == 0:	
			dailyMinList.append(attFunction[attribute](hourlyArr[j:i]))
			j = i
		i = i + 1
	dailyMinList.append(min(hourlyArr[j:i]))
	
	return dailyMinList

#return the average value of a list
def ave(arr):
	sum = 0
	for a in arr:
		sum = sum + a
	return sum / len(arr)

#Print a bunch of statistical measures
def printCustStats(trueVals, predVals, numEpochs=30, numStepsPerEpochs=10, batchSize=64, k=21, p=1002757):
	#Find R squared
	########################################################################

	#Calculate the mean of the true values
	meanY = ave(trueVals)
	print("Mean of true values = {}".format(meanY))
	
	#Calculate sseRes
	sseRes = 0
	for yi, fi in zip(trueVals, predVals):
		sseRes = sseRes + (yi-fi)**2
	print("SSE = {}".format(sseRes))
	
	#Calculate sseTot
	sseTot = 0
	for yi in trueVals:
		sseTot = sseTot + (yi-meanY)**2
	print("SST = {}".format(sseTot))	
		
	#Calculate R squared
	rSqrd = 1 - sseRes/sseTot
	print("R squared = {}".format(rSqrd))
	
	########################################################################
	
	
	#Find Adjusted R squared 
	########################################################################
	#n = sample size = # of epochs * # of steps per epochs * batch size
	n = numEpochs * numStepsPerEpochs * batchSize
	
	#k = number of independent variables in the regression equation
	k = k
	
	#Calculate adjusted R squared
	rSqrdAdj = 1 - (1-rSqrd)*((n-1)/(n-(k+1)))
	print("Adjusted R squared = {}".format(rSqrdAdj))
	
	########################################################################
	
	#p = number of parameters of the model 
	p = p 
	
	#Find Schwarz's Bayesian criterion (or BIC) (Schwarz, 1978)
	bic = n*math.log(sseRes/n)+p*math.log(n)
	print("Schwarz's Bayesian Criterion (BIC) = {}".format(bic))
	
	#Find Akaike's information criterion (Akaike, 1969)
	aic = n*math.log(sseRes/n)+2*p
	print("Akaike's Information Criterion (AIC) = {}".format(aic))
	
	#Find Corrected AIC (Hurvich and Tsai, 1989)
	aicc = n*math.log(sseRes/n)+(n+p)/(1-(p+2)/n)
	print("Corrected AIC = {}".format(aicc))
	
	

#DATA PREPROCESSING
####################################################################################

#Load and reformat the input data
weatherDf, growSeasonDf = loadData()	

#Get information about the reformatted data
#print(weatherDf.info())
stationNameList, headerList = getStationNamesAndDataFields(weatherDf)

#Get the target header
targetHeader = getTargetHeader(headerList)

#Get the number of days in advance to predict, and convert it to hours.
foreDaysDiff = getForecastDayDifference()
foreDaysDiff = foreDaysDiff*24

#Create the labeled target dataframe, which uses shifted data
labeledTargetDf = makeTargetDf(weatherDf, targetHeader, foreDaysDiff)
growingSeasonTargetDf = makeTargetDf(growSeasonDf, targetHeader, 0)

#Convert the input and target dataframes into numpy arrays.
#Get the normalized x train and test sets, the y train and test sets
#and the most recent salvaged x forecast prediction data
xDataTrain, xDataTest, yDataTrain, yDatesTrain, yDataTest, yDatesTest, xScaler, yScaler, xDataForecast, yDatesForecast = xyDataParse(weatherDf, labeledTargetDf, foreDaysDiff)

#Size of each batch
batchSize = 64
#How many hours are in each time series in each batch
#6 weeks currently
sequenceLength = 24 * 7 * 6
#Create the batch generator
batchGen = batchGenerator(batchSize, sequenceLength, xDataTrain, yDataTrain) 

#Create the validation set as the entire (labeled) test set
validationData = (np.expand_dims(xDataTest, axis=0), np.expand_dims(yDataTest, axis=0))

####################################################################################



#CREATE AND EVALUATE THE RNN
####################################################################################

#Create the model
weatherModel = getModel(xDataTrain, yDataTrain, batchGen, validationData)

#Calculate the error of the model
evaluateModel(weatherModel, xDataTest, yDataTest)


#Make predictions for the training data
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~Evaluating Training Data Performance~~~~~~~~~~~~~~~~~~~~~~~~~")
plotPredAndRealWeather(weatherModel, xDataTrain, yDataTrain, yScaler, stationNameList, yDatesTrain)

#Make predictions for the testing data
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~Evaluating Testing Data Performance~~~~~~~~~~~~~~~~~~~~~~~~~")
plotPredAndRealWeather(weatherModel, xDataTest, yDataTest, yScaler, stationNameList, yDatesTest)

####################################################################################


#FORECAST THE FUTURE
####################################################################################

if foreDaysDiff == 24*183:
	print("\n~~~~~~~~~~~~~~~~~~~~~~~~~Evaluating Growing Season Performance~~~~~~~~~~~~~~~~~~~~~~~~~")
	forecastGS(weatherModel, xDataForecast, growingSeasonTargetDf, yScaler, stationNameList)
	forecastGSMin(weatherModel, xDataForecast, growingSeasonTargetDf, yScaler, stationNameList)

####################################################################################

	
	