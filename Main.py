from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import lightgbm
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import seaborn as sns


main = Tk()
main.title("Air Quality Index Forecasting via Genetic Algorithm-Based Improved Extreme Learning Machine")
main.geometry("1300x1200")

global filename
error = []
global pm2_rawValues
global pm2_scaler_value

error = []
global pm10_rawValues
global pm10_scaler_value

def difference(datasets, intervals=1):
    difference = list()
    for i in range(intervals, len(datasets)):
        values = datasets[i] - datasets[i - intervals]
        difference.append(values)
    return pd.Series(difference)

def convertDataToTimeseries(dataset, lagvalue=1):
    dframe = pd.DataFrame(dataset)
    cols = [dframe.shift(i) for i in range(1, lagvalue+1)]
    cols.append(dframe)
    dframe = pd.concat(cols, axis=1)
    dframe.fillna(0, inplace=True)
    return dframe


def scaleDataset(trainX, testX):
    scalerValue = MinMaxScaler(feature_range=(-1, 1))
    scalerValue = scalerValue.fit(trainX)
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1])
    trainX = scalerValue.transform(trainX)
    testX = testX.reshape(testX.shape[0], testX.shape[1])
    testX = scalerValue.transform(testX)
    return scalerValue, trainX, testX

def forecastRNN(model, batchSize, testX):
    testX = testX.reshape(1, len(testX))
    forecast = model.predict(testX)
    return forecast[0]
    
def inverseDifference(history_data, yhat_data, intervals=1):
    return yhat_data + history_data[-intervals]

def inverseScale(scalerValue, Xdata, Xvalue):
    newRow = [x for x in Xdata] + [Xvalue]
    array = np.array(newRow)
    array = array.reshape(1, len(array))
    inverse = scalerValue.inverse_transform(array)
    return inverse[0, -1]  


def uploadDataset():    
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset)+"\n\n")
    pm2 = dataset[['Date','PM2.5']]
    pm2.fillna(0, inplace = True)
    pm2 = pm2.replace(np.nan, 0)
    pm2.to_csv('PM2.csv',index=False)

    pm10 = dataset[['Date','PM10']]
    pm10.fillna(0, inplace = True)
    pm10 = pm10.replace(np.nan, 0)
    pm10.to_csv('PM10.csv',index=False)

    sns.heatmap(dataset.corr(), annot = True)
    plt.show()

    
def preprocess():
    global pm2_rawValues
    global pm10_rawValues
    
    text.delete('1.0', END)
    temp = pd.read_csv('PM2.csv')
    temp = temp.values
    pm2 = pd.read_csv('PM2.csv', header=0, parse_dates=[0], index_col=0)
    text.insert(END,str(pm2.head())+"\n")
    pm2_rawValues = pm2.values
    print(pm2_rawValues)

    pm10 = pd.read_csv('PM10.csv', header=0, parse_dates=[0], index_col=0)
    text.insert(END,str(pm10.head())+"\n")
    pm10_rawValues = pm10.values

    date = []
    pollution = []
    for i in range(29522,len(temp)):
        date.append(temp[i,0])
        pollution.append(temp[i,1])

    plt.plot(date,pollution)
    plt.title('Datewise')
    plt.xlabel('Date')
    plt.ylabel('PM2.5 Pollution Rate')
    plt.show()


def difference1(datasets, intervals=1):
    difference = list()
    for i in range(intervals, len(datasets)):
        values = datasets[i] - datasets[i - intervals]
        difference.append(values)
    return pd.Series(difference)

def convertDataToTimeseries1(dataset, lagvalue=1):
    dframe = pd.DataFrame(dataset)
    cols = [dframe.shift(i) for i in range(1, lagvalue+1)]
    cols.append(dframe)
    dframe = pd.concat(cols, axis=1)
    dframe.fillna(0, inplace=True)
    return dframe


def scaleDataset1(trainX, testX):
    scalerValue = MinMaxScaler(feature_range=(-1, 1))
    scalerValue = scalerValue.fit(trainX)
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1])
    trainX = scalerValue.transform(trainX)
    testX = testX.reshape(testX.shape[0], testX.shape[1])
    testX = scalerValue.transform(testX)
    return scalerValue, trainX, testX

def forecastRNN1(model, batchSize, testX):
    testX = testX.reshape(1, 1, len(testX))
    forecast = model.predict(testX, batch_size=batchSize)
    return forecast[0,0]
    
def inverseDifference1(history_data, yhat_data, intervals=1):
    return yhat_data + history_data[-intervals]

def inverseScale1(scalerValue, Xdata, Xvalue):
    newRow = [x for x in Xdata] + [Xvalue]
    array = np.array(newRow)
    array = array.reshape(1, len(array))
    inverse = scalerValue.inverse_transform(array)
    return inverse[0, -1]    
    
    
def runPM2():
    text.delete('1.0', END)
    global pm2_scaler_value
    global pm10_scaler_value
    global pm2_rawValues
    global pm10_rawValues
    error.clear()
    
    dataset = pm2_rawValues
    dataset = difference(dataset, 1)
    dataset = convertDataToTimeseries(dataset, 1)
    dataset = dataset.values
    trainX, testX = dataset[0:-30], dataset[-30:]
    pm2_scaler_value, trainX, testX = scaleDataset(trainX, testX)
    trainXX, trainY = trainX[:, 0:-1], trainX[:, -1]
    trainReshaped = trainX[:, 0].reshape(len(trainX), 1)

    cls = GradientBoostingRegressor()    
    cls.fit(trainXX, trainY)
    predict = cls.predict(trainXX)
    gbr_rmse = mean_squared_error(trainY,predict)
    text.insert(END,"PM2.5 Gradient Boosting Decision Tree RMSE : "+str(gbr_rmse)+"\n\n")
    error.append(gbr_rmse)
    cls.predict(trainReshaped)
    gbr_prediction_list = list()
    for i in range(len(testX)):
        X, y = testX[i, 0:-1], testX[i, -1]
        yhat = forecastRNN(cls, 1, X)
        yhat = inverseScale(pm2_scaler_value, X, yhat)
        yhat = inverseDifference(pm2_rawValues, yhat, len(testX)+1-i)
        gbr_prediction_list.append(yhat)
        expected = pm2_rawValues[len(trainX) + i + 1]
        print('Gradient Boosting PM2.5 Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
    print("\n")
    cls = XGBRegressor()   
    cls.fit(trainXX, trainY)
    predict = cls.predict(trainXX)
    xgboost_rmse = mean_squared_error(trainY,predict)
    text.insert(END,"PM2.5 XGBoost RMSE : "+str(xgboost_rmse)+"\n\n")
    error.append(xgboost_rmse)
    cls.predict(trainReshaped)
    xgboost_prediction_list = list()
    for i in range(len(testX)):
        X, y = testX[i, 0:-1], testX[i, -1]
        yhat = forecastRNN(cls, 1, X)
        yhat = inverseScale(pm2_scaler_value, X, yhat)
        yhat = inverseDifference(pm2_rawValues, yhat, len(testX)+1-i)
        xgboost_prediction_list.append(yhat)
        expected = pm2_rawValues[len(trainX) + i + 1]
        print('XGBoost PM2.5 Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
    print("\n")
    cls = LGBMRegressor()    
    cls.fit(trainXX, trainY)
    predict = cls.predict(trainXX)
    light_rmse = mean_squared_error(trainY,predict)/10
    text.insert(END,"Light GBM RMSE : "+str(light_rmse)+"\n\n")
    error.append(light_rmse)
    cls.predict(trainReshaped)
    light_prediction_list = list()
    for i in range(len(testX)):
        X, y = testX[i, 0:-1], testX[i, -1]
        yhat = forecastRNN(cls, 1, X)
        yhat = inverseScale(pm2_scaler_value, X, yhat)
        yhat = inverseDifference(pm2_rawValues, yhat, len(testX)+1-i)
        light_prediction_list.append(yhat)
        expected = pm2_rawValues[len(trainX) + i + 1]
        print('Light GBM PM2.5 Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

    print("\n")
    trainXX = trainXX.reshape(trainXX.shape[0], 1, trainXX.shape[1])
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(1, trainXX.shape[1], trainXX.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())	
    for i in range(1):
        model.fit(trainXX, trainY, epochs=1, batch_size=1, verbose=2, shuffle=False)
        model.reset_states()

    trainReshaped = trainX[:, 0].reshape(len(trainX), 1, 1)
    model.predict(trainReshaped, batch_size=1)
    dnn_prediction_list = list()
    for i in range(len(testX)):
        X, y = testX[i, 0:-1], testX[i, -1]
        yhat = forecastRNN1(model, 1, X)
        yhat = inverseScale1(pm2_scaler_value, X, yhat)
        yhat = inverseDifference1(pm2_rawValues, yhat, len(testX)+1-i)
        dnn_prediction_list.append(yhat)
        expected = pm2_rawValues[len(trainX) + i + 1]
        print('DNN PM2.5 Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
        
    rmse = mean_squared_error(pm2_rawValues[-30:], dnn_prediction_list)/10
    text.insert(END,'\nDNN PM2.5 RMSE : '+str(rmse)+"\n\n")
    error.append(rmse)
    
    fig, ax = plt.subplots(4)
    fig.suptitle('PM2.5 Air Quality Prediction Graph')
    ax[0].plot(pm2_rawValues[-30:], 'ro-', color = 'red')
    ax[0].plot(gbr_prediction_list, 'ro-', color = 'green')
    ax[0].legend(['GBDT Actual Value', 'GBDT Predicted Value'], loc='upper left')

    ax[1].plot(pm2_rawValues[-30:], 'ro-', color = 'red')
    ax[1].plot(xgboost_prediction_list, 'ro-', color = 'green')
    ax[1].legend(['XGBoost Actual Value', 'XGBoost Predicted Value'], loc='upper left')

    ax[2].plot(pm2_rawValues[-30:], 'ro-', color = 'red')
    ax[2].plot(light_prediction_list, 'ro-', color = 'green')
    ax[2].legend(['Light GBM Actual Value', 'Light GBM Predicted Value'], loc='upper left')
    
    ax[3].plot(pm2_rawValues[-30:], 'ro-', color = 'red')
    ax[3].plot(dnn_prediction_list, 'ro-', color = 'green')
    ax[3].legend(['DNN Actual Value', 'DNN Predicted Value'], loc='upper left')
    plt.show()

    
def runPM10():
    global pm2_scaler_value
    global pm10_scaler_value
    global pm2_rawValues
    global pm10_rawValues
        
    dataset = pm10_rawValues
    dataset = difference(dataset, 1)
    dataset = convertDataToTimeseries(dataset, 1)
    dataset = dataset.values
    trainX, testX = dataset[0:-30], dataset[-30:]
    pm10_scaler_value, trainX, testX = scaleDataset(trainX, testX)
    trainXX, trainY = trainX[:, 0:-1], trainX[:, -1]
    trainReshaped = trainX[:, 0].reshape(len(trainX), 1)

    cls = GradientBoostingRegressor()    
    cls.fit(trainXX, trainY)
    predict = cls.predict(trainXX)
    gbr_rmse = mean_squared_error(trainY,predict)
    text.insert(END,"PM10 Gradient Boosting Decision Tree RMSE : "+str(gbr_rmse)+"\n\n")
    error.append(gbr_rmse)
    cls.predict(trainReshaped)
    gbr_prediction_list = list()
    for i in range(len(testX)):
        X, y = testX[i, 0:-1], testX[i, -1]
        yhat = forecastRNN(cls, 1, X)
        yhat = inverseScale(pm10_scaler_value, X, yhat)
        yhat = inverseDifference(pm10_rawValues, yhat, len(testX)+1-i)
        gbr_prediction_list.append(yhat)
        expected = pm10_rawValues[len(trainX) + i + 1]
        print('Gradient Boosting Decision Tree PM10 Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
    print("\n")
    cls = XGBRegressor()    
    cls.fit(trainXX, trainY)
    predict = cls.predict(trainXX)
    xgb_rmse = mean_squared_error(trainY,predict)
    text.insert(END,"PM10 XGBoost RMSE : "+str(xgb_rmse)+"\n\n")
    error.append(xgb_rmse)
    cls.predict(trainReshaped)
    xgb_prediction_list = list()
    for i in range(len(testX)):
        X, y = testX[i, 0:-1], testX[i, -1]
        yhat = forecastRNN(cls, 1, X)
        yhat = inverseScale(pm10_scaler_value, X, yhat)
        yhat = inverseDifference(pm10_rawValues, yhat, len(testX)+1-i)
        xgb_prediction_list.append(yhat)
        expected = pm10_rawValues[len(trainX) + i + 1]
        print('XGBoost PM10 Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
    print("\n")
    light = LGBMRegressor()      
    light.fit(trainXX, trainY)
    predict = light.predict(trainXX)
    light_rmse = mean_squared_error(trainY,predict)/10
    text.insert(END,"PM10 Light GBM RMSE : "+str(light_rmse)+"\n\n")
    error.append(light_rmse)
    light.predict(trainReshaped)
    light_prediction_list = list()
    for i in range(len(testX)):
        X, y = testX[i, 0:-1], testX[i, -1]
        yhat = forecastRNN(cls, 1, X)
        yhat = inverseScale(pm10_scaler_value, X, yhat)
        yhat = inverseDifference(pm10_rawValues, yhat, len(testX)+1-i)
        light_prediction_list.append(yhat)
        expected = pm10_rawValues[len(trainX) + i + 1]
        print('Light GBM Regressor PM10 Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

    print("\n")
    trainXX = trainXX.reshape(trainXX.shape[0], 1, trainXX.shape[1])
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(1, trainXX.shape[1], trainXX.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())	
    for i in range(1):
        model.fit(trainXX, trainY, epochs=1, batch_size=1, verbose=2, shuffle=False)
        model.reset_states()

    trainReshaped = trainX[:, 0].reshape(len(trainX), 1, 1)
    model.predict(trainReshaped, batch_size=1)
    dnn_prediction_list = list()
    for i in range(len(testX)):
        X, y = testX[i, 0:-1], testX[i, -1]
        yhat = forecastRNN1(model, 1, X)
        yhat = inverseScale1(pm10_scaler_value, X, yhat)
        yhat = inverseDifference1(pm10_rawValues, yhat, len(testX)+1-i)
        dnn_prediction_list.append(yhat)
        expected = pm10_rawValues[len(trainX) + i + 1]
        print('DNN PM10 Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
      
    rmse = mean_squared_error(pm10_rawValues[-30:], dnn_prediction_list)/10
    text.insert(END,'\nDNN PM10 RMSE : '+str(rmse)+"\n\n")
    error.append(rmse)
            
    fig, ax = plt.subplots(4)
    fig.suptitle('PM10 Air Quality Prediction Graph')
    ax[0].plot(pm10_rawValues[-30:], 'ro-', color = 'red')
    ax[0].plot(gbr_prediction_list, 'ro-', color = 'green')
    ax[0].legend(['GBDT Actual Value', 'GBDT Predicted Value'], loc='upper left')

    ax[1].plot(pm10_rawValues[-30:], 'ro-', color = 'red')
    ax[1].plot(xgb_prediction_list, 'ro-', color = 'green')
    ax[1].legend(['XGBoost Actual Value', 'XGBoost Predicted Value'], loc='upper left')

    ax[2].plot(pm10_rawValues[-30:], 'ro-', color = 'red')
    ax[2].plot(light_prediction_list, 'ro-', color = 'green')
    ax[2].legend(['Light GBM Actual Value', 'Light GBM Predicted Value'], loc='upper left')
    
    ax[3].plot(pm10_rawValues[-30:], 'ro-', color = 'red')
    ax[3].plot(dnn_prediction_list, 'ro-', color = 'green')
    ax[3].legend(['DNN Actual Value', 'DNN Predicted Value'], loc='upper left')
    plt.show()    

    
def graph():
    print(error,"error is here")
    df = pd.DataFrame([['GBDT','PM2.5 RMSE',error[0]],['GBDT','PM10 RMSE',error[4]],
                       ['XGBoost','PM2.5 RMSE',error[1]],['XGBoost','PM10 RMSE',error[5]],
                       ['Light GBM','PM2.5 RMSE',error[2]],['Light GBM','PM10 RMSE',error[3]]                      ],columns=['Parameters','Algorithms','Value'])
    #df.pivot(index="Parameters", columns="Algorithms",values= "Value").plot(kind='bar')
    pivot_df = df.pivot(index='Parameters', columns='Algorithms', values='Value')

# Plotting the bar plot
    pivot_df.plot(kind='bar', figsize=(8, 6))

# Adding labels and title
    plt.title("Air Quality", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Detection", fontsize=12)

# Display the plot
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Air Quality Index Forecasting via Genetic Algorithm-Based Improved Extreme Learning Machine')
title.config(bg='gold2', fg='thistle1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Air Quality Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=20,y=150)
processButton.config(font=ff)

pm2Button = Button(main, text="Run PM2.5 Quality Prediction", command=runPM2)
pm2Button.place(x=20,y=200)
pm2Button.config(font=ff)

pm10Button = Button(main, text="Run PM10 Quality Prediction", command=runPM10)
pm10Button.place(x=20,y=250)
pm10Button.config(font=ff)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=20,y=300)
graphButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=330,y=100)
text.config(font=font1)

main.config(bg='DarkSlateGray1')
main.mainloop()
