# Underground Mobility Sensing Project using multi-sensory data with Machine Learning
> You can run the code directly from Google colab here: https://colab.research.google.com/drive/1AaXqVFb-3jz5-0sYdyr-hQEkTokgjhrR?usp=sharing (downloading dataset is not needed)

Location information about commuter activities is vital for planning for travel disruptions and infrastructural development. However, GPS connectivity is lost when underground in our MRT networks. 

The Mobility Sensing Project aims to find innovative and novel ways to conduct location sensing using multi-sensory data collected in smartphones (Iphone 12 Pro and S6 Edge).

## Data Pipeline Architecture

![](https://raw.githubusercontent.com/kaiyang7766/UndergroundMobilitySensing/main/img/Pipeline.png)
>Overview of Data Pipeline Architecture

## Data info

![](https://raw.githubusercontent.com/kaiyang7766/UndergroundMobilitySensing/main/img/Data.PNG)
>Key variables: Linear Acceleration (Acc_Lin), Acceleration (Acc), Bar_Pressure, Gyrometer (Gyr), Magnetism (Mag)

## Data Cleaning and Preprocessing
First, data is cleaned by normalizing all time and variable units:

	import matplotlib.pyplot as plt
	from datetime import datetime
	import calendar

	def definetimerange (data,startTime,endTime): #get time range
		global time_range
		time_range=data[(data['Time']>startTime)&(data['Time']<endTime)]

	def normalizetime (data,startTime): #change date format to Timestamp
		global cleaned_time
		cleaned_time=(data['Timestamp']-calendar.timegm(datetime.strptime(startTime, "%Y-%m-%d %H:%M:%S.000").utctimetuple())*1000)/1000

	def normalizepressure (data): #change pressure units
		global cleaned_pressure
		cleaned_pressure=data['Bar_Pressure']/10

Then, we determine the number of mode changed (from 'Idle' to 'Moving'):

	def modeChanged(data):
	  totalNumberOfModeChanged = 0
	  timestampList = [[data['Mode'][0],data['Time'][0],data['Cleaned_Time'][0]]] #list of list
	  timeStampLastMode = data['Cleaned_Time'][0]
	  durationOfLastMode = 0
	  current_mode = data['Mode'][0]
	  for i in range(len(data)):
		mode = data['Mode'][i]
		if mode != current_mode:
		  totalNumberOfModeChanged += 1
		  duration = data['Cleaned_Time'][i] - timeStampLastMode
		  timestampList[-1][2] = duration #only after next change of mode we are able to calculate the duration of last mode, without this the mode and duration output will be interchanged

		  #insert end time for previous mode as current timestamp
		  timestampList[-1].append(data['Time'][i])

		  #insert new entries
		  timestampList.append([mode,data['Time'][i],0]) #0 is just preparation for adding duration in the next loop

		  #update new count
		  current_mode = mode
		  timeStampLastMode = data['Cleaned_Time'][i]
	#  print("Total number of Mode changed is :",totalNumberOfModeChanged)
	  return timestampList

There are substantial errors in the duration between each station recorded by phone due to lack of sensitivity of both the phone and the app, therefore amendments to these errors are made:

	def findErrorDuration(timestampList):
	  errorCount = 0
	  global errorList
	  errorList = []
	  for item in timestampList:
		if item[2] < 20:
		  errorCount+=1
		  errorList.append([item[1],item[2]])
	  print("Total number of error entries is :",errorCount)
	  print(errorList)
	  return errorCount, errorList

	def removeBackwardTimestampError(modeList):
	  newModeList = [modeList[0]]
	  removeList = []
	  currentTime = modeList[0][1]
	  modeList.pop(0)
	  while modeList != []:
		if modeList[0][1] <= currentTime:
		  removeList.append(modeList[0])
		  modeList.remove(modeList[0])
		else:
		  newModeList.append(modeList[0])
		  currentTime = modeList[0][1]
		  newModeList[-2][3] = currentTime
		  modeList.remove(modeList[0])
	  return newModeList

	def recalculateDuration(modelist):
	  for i in range(len(modelist)-1):
		if isinstance(modelist[0][1],str) == True:
		  endtime = datetime.strptime(modelist[i][3], '%Y-%m-%d %H:%M:%S')
		  starttime = datetime.strptime(modelist[i][1], '%Y-%m-%d %H:%M:%S')
		else:
		  endtime = modelist[i][3]
		  starttime = modelist[i][1]
		time = endtime - starttime
		modelist[i][2] = time.total_seconds()
	  return modelist

	def removeErrorDuration(modelist):
	  newlist = []
	  for i in range(len(modelist)-1):
		if modelist[i][2] < 5: #less than 5 seconds
		  pass
		else:
		  newlist.append(modelist[i])
	  newlist.append(modelist[-1]) #add last item in
	  return newlist

	def selectTimestampKey(modelist):
	  newlist = []
	  for i in modelist:
		newlist.append(i[1])
	  return newlist

	def findRepetitiveMode(modelist):
	  repetitivelist = []
	  mode = modelist[0][0]
	  for i in range(1,len(modelist)):
		if modelist[i][0] == mode:
		  print("Repetitve mode error at ",modelist[i])
		  repetitivelist.append(modelist[i])
		else:
		  mode = modelist[i][0]
	  return repetitivelist

	def removeRepetitiveMode(modelist, repetitivelist):
	  newlist = []
	  print("remove starting here!")
	  for i in range(len(modelist)-1):
		if modelist[i] in repetitivelist:
		  print(modelist[i],"is going to be removed")
		  newlist[-1][3] = modelist[i][3]
		else:
		  newlist.append(modelist[i])
	  return newlist

In order to perform supervised machine learning, information on each name of stations are appended through a key of words:

	def appendStation(modelist,stationlist):
	  tempStationList = stationlist[:] #copy without referencing, if not stationlist will get pop and cannot be reused
	  for item in modelist:
		if item[0] == "Idle" or item[0] == "PMD": #put PMD because TE line got PMD as Idle
		  print(tempStationList[0])
		  item.append(tempStationList[0])
		  tempStationList.pop(0)
		else:
		  item.append("Moving")
	  return modelist

	def addStationToDf(data,modelist):
	  data['Station'] = '0'
	  while modelist != []:
		for i in range(len(data['Time'])):
		  if str(data['Time'][i]) == modelist[0][1]:
			data['Station'][i] = modelist[0][4]
			print(data['Station'][i],'is added!!!')
			modelist.pop(0)
			break
	  return data

	def fillEmptyStationToDf(data):
	  temp = data['Station'][0]
	  for i in range(len(data['Station'])):
		if data['Station'][i] != '0':
		  temp = data['Station'][i]
		else:
		  data['Station'][i] = temp
	  return data

## Data Visualization
Let's take a look on the bar pressure visualization on Downtown line:
![](https://raw.githubusercontent.com/kaiyang7766/UndergroundMobilitySensing/main/img/Downtownline_BarPressure.PNG)
>iPhone 12 Pro has a much higher sensitiivty than S6 Edge.

Then, visualization on the mode of MRT:
![](https://raw.githubusercontent.com/kaiyang7766/UndergroundMobilitySensing/main/img/Downtownline_iphone_barpressure.PNG)
>'Idle' mode indicates the MRT is not moving, while 'MRT' mode indicates the MRT is in motion.

## Features Selection
### Permutation Features Importance
![](https://raw.githubusercontent.com/kaiyang7766/UndergroundMobilitySensing/main/img/PermutationFeaturesImportance.png)
>The higher the score, the more important (influencial) is the parameter

Thus, Gyrometer readings (Gyr_X, Gyr_Y, Gyr_X) are omitted from the prediction.
### Dendogram
![](https://raw.githubusercontent.com/kaiyang7766/UndergroundMobilitySensing/main/img/Dendogram.PNG)
>The closer the parameters are grouped, the more similar the effect they have on the results prediction

### Correlation Matrix
![](https://raw.githubusercontent.com/kaiyang7766/UndergroundMobilitySensing/main/img/CorrelationMatrix.png)
>Range between -1 to 1: with -1 meaning perfect negative correlation and 1 meaning perfect positive correlation

#### Final features selected
Bar Pressure, Acceleration Z, Acceleration Y, Magnetic Field Y, Magnetic Field Z are the parameters selected to perform prediction.

## Random Forest
Random Forest is trained with 10 estimators and 50 max depth using 70% of the data as training dataset

### Direct Prediction results
![](https://raw.githubusercontent.com/kaiyang7766/UndergroundMobilitySensing/main/img/DirectPredicting.PNG)
>Direct prediction (iPhone model predict on iPhone dataset etc) are good, with F1-score close to 1.

### Cross Prediction results
![](https://raw.githubusercontent.com/kaiyang7766/UndergroundMobilitySensing/main/img/CrossPredicting.PNG)
>Cross prediction (iPhone model predict on S6 Edge dataset etc) are bad, with F1-score close to 0.

## Clustering
Due to bad cross prediction results, clustering is carried out to improve the prediction.
### Silhoutte Analysis
In order to get the optimum number of clusters, silhoutte analysis showed that three clusters are the optimum number:
![](https://raw.githubusercontent.com/kaiyang7766/UndergroundMobilitySensing/main/img/SilhoutteAnalysis.PNG)
>Three clusters give a silhoutte score of around 0.8

### K-means clustering visualization
![](https://raw.githubusercontent.com/kaiyang7766/UndergroundMobilitySensing/main/img/ClusteringVisualization.PNG)
>Three clusters distribution are distinct, such that one cluster occupied the zero signal, one occupied the positive signal while one occupied the negative signal.

### Random forest model based on cluster
Random forest is trained based on the clustering results, and the prediction results are great as below:
![](https://raw.githubusercontent.com/kaiyang7766/UndergroundMobilitySensing/main/img/ClusteringPrediction.PNG)
>All three clusters give F1-score above 0.8, which is quite accurate.

# Conclusion
Without GPS data, multi-sensory data collected in smartphones can be used to:

1. Predict travel mode of high accuracy with current model
2. Cluster and predict current station

Leads to possible underground location sensing using the sensors found in smartphones.

## Suggestions for improvement
Prediction can be done based on mode sequence, which increases accuracy after each prediction
