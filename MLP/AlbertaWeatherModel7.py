#!C:\Users\jakel\AppData\Local\Programs\Python\Python37\python.exe
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


#import packages for ML	
from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import os

#For reading csv files
import numpy as np
import pandas as pd

#For plotting history
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numpy.polynomial.polynomial import polyfit
import seaborn as sns
import statistics
import time
import math

#For ML
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow import feature_column
from sklearn.model_selection import train_test_split

#For Dates
import datetime as dt
from datetime import timedelta
from datetime import date


#For string reformatting
import re
from re import sub


#Load and preprocess the data
#Read CSV weather data file into a DataFame
dataURLPath = r"WeatherDatasets\Vulcan_2011_2019\\"

readCSVList = []
for filename in os.listdir(dataURLPath):
	readCSVList.append(pd.read_csv(dataURLPath + filename, encoding = 'unicode_escape'))

	
weatherDataframe = pd.concat(readCSVList, ignore_index = True)


#use mergesort because mergesort here is stable
weatherDataframe = weatherDataframe.sort_values("Station Name", kind = 'mergesort')

#Get the headers of the weather data
weatherDataframeHeaders = list(weatherDataframe.columns)

#Preprocess the data:
#Convert the Date column into Year, Month, and Date Columns
#Parse Year, Month, and Day from dd-mmm-yyyy date
monthDict = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
dateCol = weatherDataframe.pop(weatherDataframeHeaders[1])
weatherDataframe['Year'] = [2000 + int(dateColElem[-2:]) for dateColElem in dateCol]
weatherDataframe['Month'] = [monthDict[dateColElem[3:6]] for dateColElem in dateCol]
weatherDataframe['Day'] = [int(dateColElem[:2]) for dateColElem in dateCol]

#Delete the columns of data where there is missing data
weatherDataframe = weatherDataframe.dropna(axis = 'columns')

#Save the original dataframe
originalWeatherDataframe = weatherDataframe.copy()

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
	
#Get the headers of the weather data (ignoring station name as the header 0)
weatherDataframeHeaders = list(weatherDataframe.columns)
print("The headers of the data are:")
for i in range(len(weatherDataframeHeaders))[1:-3]:
	print("i = {}: Header = {}".format(i, weatherDataframeHeaders[i]))
	
originalTargetHeaderIndexList = []
headerIndex = getUserInput(1, len(weatherDataframeHeaders) -4,
						   "Please enter in the header indices of the data you wish to forecast:")
while headerIndex != 0:
	if headerIndex not in originalTargetHeaderIndexList:
		originalTargetHeaderIndexList.append(headerIndex)	
	headerIndex = getUserInput(0, len(weatherDataframeHeaders) -4,
							   "Enter in 0 to stop, and a header index otherwise:")

#Get how far in advance the model should forecast data
print("Please enter in how many days in advance you wish to forecast data:")
daysDifference = getUserInput(1, 365, "Enter in a number between 1 and 365:")

#Set whether or not to use saved models
print("Enter in 1 if you wish to load existing model h5 files, and 2 if any new models must be created.")
loadFilesBool = getUserInput(1, 2, "Enter in a number between 1 and 2:")

#Returns a probability value from daysDifference in the future
def forecastProbabilityData(dataframe, header, currentIndex, daysDifference):
	return (dataframe[header][currentIndex + daysDifference])/100.0

#Returns a value from daysDifference in the future
def forecastData(dataframe, header, currentIndex, daysDifference):
	return dataframe[header][currentIndex + daysDifference]

#Returns a value from daysDifference in the past
def previousData(dataframe, header, currentIndex, daysDifference):
	return dataframe[header][currentIndex - daysDifference]

#Returns a date in yyyy-mm-dd format from a given yyyy-mm-dd format date,
#daysDifference days into the future.
def convertDate(givenDate, daysDifference):
	dateElemList = givenDate.split("-")
	initialDate = date(int(dateElemList[0]), int(dateElemList[1]), int(dateElemList[2]))
	finalDate = initialDate + timedelta(days = daysDifference)
	return "{}-{}-{}".format(finalDate.year, finalDate.month, finalDate.day)

#Normalize the data
#THIS MUST APPLY TO ALL INPUTS, INCLUDING TEST DATASETS AND NEW DATA 
def normalize(x, trainStats):
	tempStdList = []
	for std in trainStats['std']:
		if(std == 0.0):
			tempStdList.append(1)
		else:
			tempStdList.append(std)
	return (x - trainStats['mean']) / tempStdList

#Build the model:
def buildModel():
	print("Building model...")
	model = keras.Sequential([
		layers.Dense(512, activation='tanh', input_shape=[len(normedTrainFrame.keys())]),
		layers.Dense(512, activation='tanh'),
		layers.Dense(256, activation='tanh'),
		layers.Dense(256, activation='tanh'),
		layers.Dense(128, activation='tanh'),
		layers.Dense(64, activation='tanh'),
		layers.Dense(32, activation='tanh'),
		layers.Dense(1)
	])

	#optimizer = tf.keras.optimizers.RMSprop(0.001)
	optimizer = 'adam'

	model.compile(loss='mse',
				  optimizer = optimizer,
				  metrics=['mae', 'mse'])
	return model

#Train the model and plot the history
#Display training progress by printing a single dot for each completed epoch 
#Called on every batch
class PrintDot(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
		if epoch % 100 == 0: print('')
		print('.', end='')

#Plot the Error vs Training Epoch for the model
def plotHistory(history, targetHeader):
	hist = pd.DataFrame(history.history)
	
	hist['epoch'] = history.epoch
	
	#Create the graph for the trainging and value MAE
	plt.figure(dpi = 120)
	plt.xlabel('Epoch')
	plt.ylabel('Mean Abs Error [{}]'.format(targetHeader.replace(".", " ")))
	plt.plot(hist['epoch'], hist['mae'], label = 'Train Error', linewidth = 1, )
	plt.plot(hist['epoch'], hist['val_mae'], label = 'Val Error', linewidth = 1)
	plt.ylim(0, max(max(hist['mae'][math.floor(len(hist['mae'])*0.1):]), 
				max(hist['val_mae'][math.floor(len(hist['mae'])*0.1):]), max(max(hist['mae']), max(hist['val_mae']))*0.01))
	
	plt.legend()

	#Create the graph for the trainging and value MSE
	plt.figure(dpi = 120)
	plt.xlabel('Epoch')
	plt.ylabel('Mean Square Error [${}^2$]'.format(targetHeader.replace(".", " ")))
	plt.plot(hist['epoch'], hist['mse'], label='Train Error', linewidth = 1)
	plt.plot(hist['epoch'], hist['val_mse'], label = 'Val Error',  linewidth = 1)
	plt.ylim(0, max(max(hist['mse'][math.floor(len(hist['mse'])*0.1):]), 
				max(hist['val_mse'][math.floor(len(hist['mse'])*0.1):]), max(max(hist['mse']), max(hist['val_mse']))*0.01))
	plt.legend()
	
	#Show the graphs
	plt.show()
	plt.close()

#Some of the preprocessing for a weatherDataframe converts the townships into location data columns,
#Converts the dtype of columns into floats, and sanitizes the column headers
def preprocessDataframe(weatherDataframe):

	#Convert Township column into 3 location columns for latitude, longitude, and altitude
	weatherDataframeHeaders = list(weatherDataframe.columns)
	ogTownshipList = weatherDataframe.pop(weatherDataframeHeaders[0])
	townshipLatitudeList = []
	townshipLongitudeList = []
	townshipAltitudeList = []
	
	for township in ogTownshipList:
		townshipLatitudeList.append(townshipLocationData[township][0])
		townshipLongitudeList.append(townshipLocationData[township][1])
		townshipAltitudeList.append(townshipLocationData[township][2])
	
	weatherDataframe['Latitude'] = townshipLatitudeList
	weatherDataframe['Longitude'] = townshipLongitudeList
	weatherDataframe['Altitude'] = townshipAltitudeList

	#Convert dtypes of weatherDataframe columns from object to float64
	weatherDataframe = weatherDataframe.apply(pd.to_numeric, errors='ignore')
	
	#Convert any remaining object dtype column to a numeric one. (Especially to remove commas from numbers)
	weatherDataframeHeaders = list(weatherDataframe.columns)
	for header in weatherDataframeHeaders:
		if weatherDataframe[header].dtype == 'object':
			weatherDataframe[header] = [float(elem.replace(',','')) for elem in weatherDataframe[header]]
	
	#Print the dtypes of the weather data
	#print(weatherDataframe.dtypes)

	#Clean the headers to match one of these regular expressions: Replace all invalid chars with '.'
	#[A-Za-z0-9.][A-Za-z0-9_.\\-/]* (for scopes at the root)
	#[A-Za-z0-9_.\\-/]* (for other scopes)
	weatherDataframeHeaders = list(weatherDataframe.columns)
	weatherDataframe.rename(columns = {oldName:re.sub('[^A-Za-z0-9\_]', '.', oldName) for oldName in weatherDataframeHeaders}, inplace = True)
	weatherDataframeHeaders = list(weatherDataframe.columns)

	#Update the target header 
	weatherDataframeHeaders = list(weatherDataframe.columns)
	for i in range(len(weatherDataframeHeaders)):
		if 'Forecasted.' in weatherDataframeHeaders[i]:
			targetHeader = weatherDataframeHeaders[i]

	return [weatherDataframe, targetHeader]

#Dictionary for the latitude, longitude, and elevation (m) of a township name
townshipLocationData = {'Blackie AGCM':[50.5458, -113.6403, 1019.00], 
						'Champion AGDM':[50.2953, -113.3467, 995.00],
						'Mossleigh AGCM':[50.6726, -113.3487, 965.00],
						'Queenstown':[50.7000, -112.9167, 944.00],
						'Travers AGCM':[50.3038, -112.8626, 955.00]}

#NOTES
#TO FIND THE PREDICTED VALUE FOR testPredictions[i] IN THE EXCEL FILE:
#ExcelRowNumber = i + daysDifference + 2
#E.g. for i = 237, the excel row number of the forecasted data is row 419 (419 = 237 + 3*60 + 2)

#Create a forecast regression for each target header
for targetHeaderIndex in originalTargetHeaderIndexList:
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~CREATING A MODEL~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	startTime = time.time()
	
	#Reset the dataframe
	weatherDataframe = originalWeatherDataframe.copy()
	weatherDataframeHeaders = list(weatherDataframe.columns)
	
	
	
	#Create the previous data columns (i days in the past)
	#Extremely roughly, 5 previous days can be processed in a minute.
	daysPreviousDifference = 30
	nonDuplicateHeaders = ['Station Name']
	for i in range(daysPreviousDifference):
		if i % 5 == 0:
			print(".", end = "")
		for header in weatherDataframeHeaders:
			if header in nonDuplicateHeaders:
				continue
			tempList = []
			for j in range(len(weatherDataframe)):
				if((date(int(weatherDataframe['Year'][0]), int(weatherDataframe['Month'][0]), int(weatherDataframe['Day'][0])) - 
				date(int(weatherDataframe['Year'][j]), int(weatherDataframe['Month'][j]), int(weatherDataframe['Day'][j]))).days > -(daysPreviousDifference)):
					tempList.append(pd.NaT)
				else:
					tempList.append(previousData(weatherDataframe, header, j, i+1))
			weatherDataframe["{} -{} Days".format(header, i+1)] = tempList
	
	#Delete any rows that do not contain any data (because they are too late)
	weatherDataframe = weatherDataframe.dropna()
	weatherDataframe.reset_index(inplace=True)
	weatherDataframe.pop("index")
	weatherDataframeHeaders = list(weatherDataframe.columns)
	
	
	
	#Create a target column for the forecasted data using user choice
	tempList = []
	for i in range(len(weatherDataframe)):
		if((date(int(weatherDataframe['Year'][len(weatherDataframe) - 1]), int(weatherDataframe['Month'][len(weatherDataframe) - 1]), int(weatherDataframe['Day'][len(weatherDataframe) - 1])) - 
			date(int(weatherDataframe['Year'][i]), int(weatherDataframe['Month'][i]), int(weatherDataframe['Day'][i]))).days < daysDifference):
			tempList.append(pd.NaT)
		elif 'Probability' in weatherDataframeHeaders[targetHeaderIndex]:
			tempList.append(forecastProbabilityData(weatherDataframe, weatherDataframeHeaders[targetHeaderIndex], i, daysDifference))
		else:
			tempList.append(forecastData(weatherDataframe, weatherDataframeHeaders[targetHeaderIndex], i, daysDifference))

	targetHeader = 'Forecasted.' + weatherDataframeHeaders[targetHeaderIndex]
	weatherDataframe[targetHeader] = tempList

	#Salvage the recent rows that do not contain any data for the future model.
	recentWeatherDataframe = weatherDataframe.loc[pd.isnull(weatherDataframe.loc[:, targetHeader]), : ]	

	#Delete any rows that do not contain any data (because they are too recent)
	weatherDataframe = weatherDataframe.dropna()
	
	weatherDataframe = preprocessDataframe(weatherDataframe)[0]
	recentWeatherDataframe, targetHeader = preprocessDataframe(recentWeatherDataframe)

	#Create test and training dataframes.
	#testFrame contains no rows from trainFrame
	trainFrame = weatherDataframe.sample(frac = 0.8, random_state = int(math.floor(time.time() % 10)))
	testFrame = weatherDataframe.drop(trainFrame.index)

	
	#Get statistics for each header except for the target
	#And any nonnumerical column
	#(We want to use these to normalize the inputs and not the output)
	trainStats = trainFrame.describe()
	trainStats.pop(targetHeader)
	trainStats = trainStats.transpose()

	#print(trainStats)
	#print(trainFrame)
	#print(testFrame)

	#Split features from labels
	trainLabels = trainFrame.pop(targetHeader)
	testLabels = testFrame.pop(targetHeader)
	recentWeatherDataframe.pop(targetHeader)
	
	#print(trainLabels)
	#print(testLabels)

	normedTrainFrame = normalize(trainFrame, trainStats)
	normedTestFrame = normalize(testFrame, trainStats)
	normedFutureFrame = normalize(recentWeatherDataframe, trainStats)
	
	#print(normedTrainFrame)

	endTime = time.time()
	print("\nSecondary preprocessing is complete. \nTime: {} minutes, {} seconds.".format(math.floor((endTime - startTime)/60.0), (endTime - startTime) % 60))

	#If a file exists and the setting is to load files, load it as an HDF5 file
	#Recreate the exact same model, including weights and optimizer.
	modelFilePath = 'WeatherModels7\weatherModel_' + targetHeader.replace('.','_') + '.h5'
	
	if os.path.isfile(modelFilePath) and loadFilesBool == True:
		weatherModel = keras.models.load_model(modelFilePath)
		print("weather model loaded from: ", modelFilePath)
	else:
		weatherModel = buildModel()

		custEpochs = 1000
		
		#Stop training early if the val_loss stops decreasing for 10 epochs in a row
		earlyStopCallbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 150),
							  keras.callbacks.ModelCheckpoint(filepath=modelFilePath, monitor = 'val_loss', save_best_only=True),
							  PrintDot()]
		
		print("Training the model...")
		startTime = time.time()
		
		history = weatherModel.fit(
		  normedTrainFrame, trainLabels, epochs = custEpochs, batch_size = 512, shuffle = True,
		  validation_split = 0.2, verbose=0, callbacks=earlyStopCallbacks)
		
		endTime = time.time()
		print("\nTraining the model with {} epochs is complete. \nTime: {} minutes, {} seconds.".format(custEpochs, math.floor((endTime - startTime)/60.0), (endTime - startTime) % 60))
		  
		print()

		hist = pd.DataFrame(history.history)
		hist['epoch'] = history.epoch
		
		print("\nEnd training loss values:")
		print(hist.tail())

		##############################################################################
		plotHistory(history, targetHeader)
		##############################################################################

		# Save entire model to a HDF5 file
		weatherModel.save(modelFilePath)
		print("\nModel saved to: ", modelFilePath)

	#Print the model summary
	print("\nModel Summary:")
	weatherModel.summary()
	
	#Test the model with test data
	loss, mae, mse = weatherModel.evaluate(normedTestFrame, testLabels, verbose=0)
	print("Testing set Mean Abs Error: {:5.8f} ({})".format(mae, targetHeader))

	#Make predictions
	testPredictions = weatherModel.predict(normedTestFrame).flatten()

	#Print the most egregious model predictions
	#These will be predictions with an AE > 2*MAE
	print("\nThe predictions with an AE > 2*MAE:")
	
	egregiousPredictionsCount = 0
	for i in range(len(testPredictions)):
		if i in testLabels and abs(testPredictions[i] - testLabels[i]) > mae * 2 :
			egregiousPredictionsCount = egregiousPredictionsCount + 1
			print("i = {}: 	 prediction = {:5.3f},	true value = {:5.3f},	MAE = {:5.3f},	MAE multiple = {:5.3f}".format(
				i, testPredictions[i], testLabels[i], abs(testPredictions[i] - testLabels[i]), abs(testPredictions[i] - testLabels[i])/mae))
	print("Above is Displaying {} of {} ({:5.3f}%) test predictions.".format(
		  egregiousPredictionsCount, len(testPredictions), egregiousPredictionsCount/len(testPredictions)*100))

	#Plot the predictions vs the true values of the test set
	plt.scatter(testLabels, testPredictions)
	plt.xlabel("True Values [{}]".format(targetHeader).replace(".", " "))
	plt.ylabel("Predictions [{}]".format(targetHeader).replace(".", " "))
	plt.axis('equal')
	plt.axis('square')	
	
	xRange = (plt.xlim()[1] - plt.ylim()[0])
	yRange = (plt.ylim()[1] - plt.ylim()[0])
	
	plt.xlim(plt.xlim()[0] - xRange*0.05, plt.xlim()[1] + xRange*0.05)
	plt.ylim(plt.ylim()[0] - yRange*0.05, plt.ylim()[1] + yRange*0.05)
	#Plots the linear line of best fit
	_ = plt.plot([plt.xlim()[0] - xRange*0.05, plt.xlim()[1] + xRange*0.05],
				 [plt.ylim()[0] - yRange*0.05, plt.ylim()[1] + yRange*0.05], color='green')	
	b, m = polyfit(testLabels, testPredictions, 1)
	print("b = {}, m = {} for linear regression of predicted values vs true values".format(b, m))
	plt.plot(testLabels, b+m*testLabels, '-', color='red')
	plt.show()
	plt.close()

	#Plot error distribution
	error = testPredictions - testLabels
	plt.hist(error, bins = 40, rwidth = 0.92)
	plt.xlabel("Prediction Error (Prediction - True Value)[{}]".format(targetHeader.replace(".", " ")))
	_ = plt.ylabel("Count (No. of Predictions)")
	plt.show()
	plt.close()

	print()






	#Create a prediction model using the most recent data.
	#Create 1 plot per station.
	
	
	
	
	#Reconstruct a date column for the recentWeatherDataframe
	recentDateList = []
	for i in range(len(recentWeatherDataframe['Year'])):
		recentDateList.append("{}-{}-{}".format(list(recentWeatherDataframe['Year'])[i], 
												list(recentWeatherDataframe['Month'])[i],
												list(recentWeatherDataframe['Day'])[i]))
	recentWeatherDataframe['Date'] = recentDateList
	
	
	#Create a list of dataframes, one for each weather station
	earlyDate = recentDateList[0]
	lateDate = recentDateList[len(recentDateList) - 1]
	
	earlyDateIndexList = []
	lateDateIndexList = []
	
	for i in range(len(recentWeatherDataframe['Date'])):
		if list(recentWeatherDataframe['Date'])[i] == earlyDate:
			earlyDateIndexList.append(i)
		if list(recentWeatherDataframe['Date'])[i] == lateDate:
			lateDateIndexList.append(i)
	recentWeatherDataframe.pop('Date')
	
	futureFrameStationList = []
	recentDateStationList = []
	for earlyDateIndex, lateDateIndex in zip(earlyDateIndexList, lateDateIndexList):
		futureFrameStationList.append(normedFutureFrame.iloc[range(earlyDateIndex, lateDateIndex + 1),:])
		recentDateStationList.append(recentDateList[earlyDateIndex: lateDateIndex + 1])
	
	futurePredictionsList = [weatherModel.predict(futureFrame).flatten() for futureFrame in futureFrameStationList]
	
	
	for i in range(len(recentDateStationList)):
		forecastedDateList = [convertDate(date, daysDifference) for date in recentDateStationList[i]]
		recentDateStationList[i] = forecastedDateList
	
	
	plt.figure(dpi = 120)
	plt.xlabel('Date')
	plt.ylabel("Predictions [{}]".format(targetHeader).replace(".", " "))
	
	stationNameList = ['Blackie AGCM', 
						'Champion AGDM',
						'Mossleigh AGCM',
						'Queenstown',
						'Travers AGCM']
	for futurePrediction, recentDate, stationName in zip(futurePredictionsList, recentDateStationList, stationNameList):
		plt.plot(recentDate, futurePrediction, label = stationName, linewidth = 1)
	
	plt.gca().grid(True, linewidth = 0.6, linestyle=':')
	plt.xticks(rotation = 60)
	plt.subplots_adjust(left = 0.12, right = 0.999, top = 0.95, bottom = 0.15)
	plt.legend()
	
	
	plt.title("Forecasted Data for Weather Stations for the Future {} Days".format(daysDifference))
	
	plt.show()
	plt.close()
	
	
	#Create the average forecast graph
	plt.figure(dpi = 120)
	plt.xlabel('Date')
	plt.ylabel("Predictions [{}]".format(targetHeader).replace(".", " "))
	

	predictionAveList = [0 for i in range(len(futurePredictionsList[0]))]
	for futurePrediction in futurePredictionsList:
		for i in range(len(futurePrediction)):
			predictionAveList[i] = predictionAveList[i] + futurePrediction[i]
	predictionAveList = [prediction/len(stationNameList) for prediction in predictionAveList]

	plt.plot(recentDate, predictionAveList, label = 'Average of Stations', linewidth = 1)
	
	plt.gca().grid(True, linewidth = 0.6, linestyle=':')
	plt.xticks(rotation = 60)
	plt.subplots_adjust(left = 0.12, right = 0.999, top = 0.95, bottom = 0.15)
	plt.legend()
	
	
	plt.title("Forecasted Data for Weather Stations for the Future {} Days".format(daysDifference))
	
	plt.show()
	plt.close()

	#Predict Frost Days
	predictionFrostList = []
	for predictionTemp in predictionAveList:
		if(predictionTemp <= 0):
			predictionFrostList.append(1)
		else:
			predictionFrostList.append(0)
			
	plt.plot(recentDate, predictionFrostList, label = 'Frost Occurance', linewidth = 1)
	
	plt.gca().grid(True, linewidth = 0.6, linestyle=':')
	plt.xticks(rotation = 60)
	plt.subplots_adjust(left = 0.12, right = 0.999, top = 0.95, bottom = 0.15)
	plt.legend()
	
	
	plt.title("Forecasted Data for Weather Stations for the Future {} Days".format(daysDifference))
	
	plt.show()
	plt.close()
	
	
	
	
	
	
	#NOW WITH WACK ADJUSTMENT
	#Plot the ADJUSTED predictions vs the true values of the test set
	adjustedTestPredictions = [(1/m)*(prediction - b) for prediction in testPredictions]
	plt.scatter(testLabels, adjustedTestPredictions)
	plt.xlabel("True Values [{}]".format(targetHeader).replace(".", " "))
	plt.ylabel("Adjusted Predictions [{}]".format(targetHeader).replace(".", " "))
	plt.axis('equal')
	plt.axis('square')	
	
	xRange = (plt.xlim()[1] - plt.ylim()[0])
	yRange = (plt.ylim()[1] - plt.ylim()[0])
	
	plt.xlim(plt.xlim()[0] - xRange*0.05, plt.xlim()[1] + xRange*0.05)
	plt.ylim(plt.ylim()[0] - yRange*0.05, plt.ylim()[1] + yRange*0.05)
	#Plots the linear line of best fit
	_ = plt.plot([plt.xlim()[0] - xRange*0.05, plt.xlim()[1] + xRange*0.05],
				 [plt.ylim()[0] - yRange*0.05, plt.ylim()[1] + yRange*0.05], color='green')	
	b, m = polyfit(testLabels, adjustedTestPredictions, 1)
	print("b = {}, m = {} for linear regression of predicted values vs true values".format(b, m))
	plt.plot(testLabels, b+m*testLabels, '-', color='red')
	plt.show()
	plt.close()

	#Plot error distribution
	error = adjustedTestPredictions - testLabels
	plt.hist(error, bins = 40, rwidth = 0.92)
	plt.xlabel("Prediction Error (Prediction - True Value)[{}]".format(targetHeader.replace(".", " ")))
	_ = plt.ylabel("Count (No. of Predictions)")
	plt.show()
	plt.close()


	#Create an ADJUSTED prediction model using the most recent data.
	#Create 1 plot per station.
	
	
	
	
	#Reconstruct a date column for the recentWeatherDataframe
	
	

		
	
	adjustedFuturePredictionsList = []
	for futurePrediction in futurePredictionsList:
		adjustedFuturePredictionsList.append([(1/m)*(prediction - b) for prediction in futurePrediction])
	
	plt.figure(dpi = 120)
	plt.xlabel('Date')
	plt.ylabel("Predictions [{}]".format(targetHeader).replace(".", " "))
	
	stationNameList = ['Blackie AGCM', 
						'Champion AGDM',
						'Mossleigh AGCM',
						'Queenstown',
						'Travers AGCM']
	for futurePrediction, recentDate, stationName in zip(adjustedFuturePredictionsList, recentDateStationList, stationNameList):
		plt.plot(recentDate, futurePrediction, label = stationName, linewidth = 1)
	
	plt.gca().grid(True, linewidth = 0.6, linestyle=':')
	plt.xticks(rotation = 60)
	plt.subplots_adjust(left = 0.12, right = 0.999, top = 0.95, bottom = 0.15)
	plt.legend()
	
	
	plt.title("Adjusted Forecasted Data for Weather Stations for the Future {} Days".format(daysDifference))
	
	plt.show()
	plt.close()
	
	
	#Create the average forecast graph
	plt.figure(dpi = 120)
	plt.xlabel('Date')
	plt.ylabel("Predictions [{}]".format(targetHeader).replace(".", " "))
	

	predictionAveList = [0 for i in range(len(adjustedFuturePredictionsList[0]))]
	for futurePrediction in adjustedFuturePredictionsList:
		for i in range(len(futurePrediction)):
			predictionAveList[i] = predictionAveList[i] + futurePrediction[i]
	predictionAveList = [prediction/len(stationNameList) for prediction in predictionAveList]

	plt.plot(recentDate, predictionAveList, label = 'Average of Stations', linewidth = 1)
	
	plt.gca().grid(True, linewidth = 0.6, linestyle=':')
	plt.xticks(rotation = 60)
	plt.subplots_adjust(left = 0.12, right = 0.999, top = 0.95, bottom = 0.15)
	plt.legend()
	
	
	plt.title("Forecasted Data for Weather Stations for the Future {} Days".format(daysDifference))
	
	plt.show()
	plt.close()

	#Predict Frost Days
	predictionFrostList = []
	for predictionTemp in predictionAveList:
		if(predictionTemp <= 0):
			predictionFrostList.append(1)
		else:
			predictionFrostList.append(0)
			
	plt.plot(recentDate, predictionFrostList, label = 'Frost Occurance', linewidth = 1)
	
	plt.gca().grid(True, linewidth = 0.6, linestyle=':')
	plt.xticks(rotation = 60)
	plt.subplots_adjust(left = 0.12, right = 0.999, top = 0.95, bottom = 0.15)
	plt.legend()
	
	
	plt.title("Forecasted Data for Weather Stations for the Future {} Days".format(daysDifference))
	
	plt.show()
	plt.close()
