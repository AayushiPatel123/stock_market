#pip install -U numpy==1.18.5

from flask import Flask,redirect,url_for
from flask import render_template,request
from datetime import time, timedelta
import pandas as pd1
import requests
# from bs4 import BeautifulSoup
import pandas as pd
import datetime
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from bs4 import BeautifulSoup 
import csv 
import re

import numpy
#import tflearn
import tensorflow
import random
import json
import tflearn
import pickle
import os
import tweepy as tw
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from flask_jsonpify import jsonpify
import pandas as pd
import re
import datetime
#from datetime import timedelta
import time
from flask_wtf import Form
from wtforms.fields import DateField, EmailField, TelField
#importing required libraries
import numpy as np
import pandas as pd
import pymysql

import yfinance as yf
from sklearn import preprocessing
import datetime as dt
from plotly.offline import plot
import plotly.graph_objects as go

#to plot within notebook
import matplotlib.pyplot as plt
from flask import jsonify 
# import tablib
import os
from yahoo_fin import stock_info as si
from tensorflow.keras.layers import Dense, Dropout, LSTM
import requests_html 
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB * 2 of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

def getpriceinfo(symbol):
    lp=si.get_live_price(symbol)

    #print(si.get_day_most_active())
 
    # get biggest gainers
    #print(si.get_day_gainers())
 
    # get worst performers
    #print(si.get_day_losers())
    return lp

def getqoutetable(symbol):

    qt=si.get_quote_table(symbol, dict_result = False)
 
    #print(si.get_day_most_active())
 
    # get biggest gainers
    #print(si.get_day_gainers())
 
    # get worst performers
    #print(si.get_day_losers())
    return qt

# function to calculate percentage difference considering baseValue as 100%
def percentageChange(baseValue, currentValue):
    return((float(currentValue)-baseValue) / abs(baseValue)) *100.00

# function to get the actual value using baseValue and percentage
def reversePercentageChange(baseValue, percentage):
    return float(baseValue) + float(baseValue * percentage / 100.00)

# function to transform a list of values into the list of percentages. For calculating percentages for each element in the list
# the base is always the previous element in the list.
def transformToPercentageChange(x):
    baseValue = x[0]
    x[0] = 0
    for i in range(1,len(x)):
        pChange = percentageChange(baseValue,x[i])
        baseValue = x[i]
        x[i] = pChange

# function to transform a list of percentages to the list of actual values. For calculating actual values for each element in the list
# the base is always the previous calculated element in the list.

dictionaryofdateandprice={}

def reverseTransformToPercentageChange(baseValue, x):
    x_transform = []
    for i in range(0,len(x)):
        value = reversePercentageChange(baseValue,x[i])
        baseValue = value
        x_transform.append(value)
    return x_transform

'''##def runningMean(x, N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
         y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N

def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]


from statistics import mean
  
def Average(lst):
    return mean(lst)'''

def dbConnection():
    try:
        connection = pymysql.connect(host="localhost", user="root", password="root", database="stockmarket",charset='utf8',port=3308)
        return connection
    except:
        print("Something went wrong in database Connection")

def dbClose():
    try:
        dbConnection().close()
    except:
        print("Something went wrong in Close DB Connection")

con=dbConnection()
cursor=con.cursor()

#read the data file
dictofdateandprice={}
def predictpriceofdata(stockname):
    global dictionaryofdateandprice
    global date_index
    global train_transform
    global future_date_index
    global future_closing_price_transform
    global label1
    global label2
    global dt4 
    global ttf4
    
    

    df = pd.read_csv('data2\\'+stockname+'.csv')
# store the first element in the series as the base value for future use.
    baseValue = df['Close'][0]

# create a new dataframe which is then transformed into relative percentages
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]

# transform the 'Close' series into relative percentages
    transformToPercentageChange(new_data['Close'])

# set Dat column as the index
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)

# create train and test sets
    dataset = new_data[0:1500].values
    print("====dataset====")
    print(dataset)
    print("====len dataset====")
    print(len(dataset))
    train, valid = train_test_split(dataset, train_size=0.99, test_size=0.01, shuffle=False)

# convert dataset into x_train and y_train.
# prediction_window_size is the size of days windows which will be considered for predicting a future value.

    prediction_window_size = 60
    x_train, y_train = [], []
    for i in range(prediction_window_size,len(train)):
        x_train.append(dataset[i-prediction_window_size:i,0])
        y_train.append(dataset[i,0])
    x_train, y_train = np.array(x_train).astype(np.float32), np.array(y_train).astype(np.float32)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


    # X_train = []
    # y_train = []
    # for i in range(60, 1500):
    #     X_train.append(np.array(dataset[60:1600].astype(np.float32))[i-60:i])
    #     y_train.append(np.array(dataset[60:1600].astype(np.float32))[i])
    # X_train, y_train = np.array(X_train), np.array(y_train)
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



    x_valid, y_valid = [], []
    for i in range(60,120):
        x_valid.append(dataset[i-prediction_window_size:i,0])
        y_valid.append(dataset[i,0])
        
    X_test = np.asarray(x_valid).astype('float32')
    #y_test = np.asarray(y_valid).astype('float32')

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))


##################################################################################################
# create and fit the LSTM network
# Initialising the RNN
    model = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

# Adding the output layer
    model.add(Dense(units = 1))
# Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # accu = model.evaluate(x_test,y_test)
    # print("accuracy is")
    # print(acu)
# Fitting the RNN to the Training set
    model.fit(x_train, y_train, epochs = 1, batch_size = 1000)
    # def predict_prob(number):
    #   return [number[0],1-number[0]]
    
    # y_prob = np.array(list(map(predict_prob, model_cEXT.predict(X_test))))
##################################################################################################
#predicting future values, using past 60 from the train data
# for next 10 yrs total_prediction_days is set to 3650 days
    total_prediction_days = 3650
    inputs = new_data[-total_prediction_days:].values
    # print("======len(new_data)==========")
    # print(len(new_data))
    # print("======len(inputs)==========")
    # print(len(inputs))
    inputs = inputs.reshape(-1,1)

# create future predict list which is a two dimensional list of values.
# the first dimension is the total number of future days
# the second dimension is the list of values of prediction_window_size size
    X_predict = []
    for i in range(prediction_window_size,inputs.shape[0]):
        X_predict.append(inputs[i-prediction_window_size:i,0])
    X_predict = np.array(X_predict).astype(np.float32)
   
# predict the future
    X_predict = np.reshape(X_predict, (X_predict.shape[0],X_predict.shape[1],1))
    future_closing_price = model.predict(X_predict)

    train, valid = train_test_split(new_data, train_size=0.99, test_size=0.01, shuffle=False)
    date_index = pd.to_datetime(train.index)

#converting dates into number of days as dates cannot be passed directly to any regression model
    x_days = (date_index - pd.to_datetime('1970-01-01')).days

# we are doing prediction for next 5 years hence prediction_for_days is set to 1500 days.
    prediction_for_days = 1000
    future_closing_price = future_closing_price[:prediction_for_days]

# create a data index for future dates
    x_predict_future_dates = np.asarray(pd.RangeIndex(start=x_days[-1] + 1, stop=x_days[-1] + 1 + (len(future_closing_price))))
    future_date_index = pd.to_datetime(x_predict_future_dates, origin='1970-01-01', unit='D')

# transform a list of relative percentages to the actual values
    train_transform = reverseTransformToPercentageChange(baseValue, train['Close'])

# for future dates the base value the the value of last element from the training set.
    baseValue = train_transform[-1]
    valid_transform = reverseTransformToPercentageChange(baseValue, valid['Close'])
    future_closing_price_transform = reverseTransformToPercentageChange(baseValue, future_closing_price)

# recession peak date is the date on which the index is at the bottom most position.
    recessionPeakDate =  future_date_index[future_closing_price_transform.index(min(future_closing_price_transform))]
    minCloseInFuture = min(future_closing_price_transform)
    print("The stock market will reach to its lowest bottom on", recessionPeakDate)
    print("The lowest index the stock market will fall to is ", minCloseInFuture)

    
# plot the graphs
    label1='Close Price History of'+ stockname +'company'
    label2='Predicted Close of'+ stockname +'company'
    plt.figure(figsize=(16,8))
    df_x = pd.to_datetime(new_data.index)
    plt.plot(date_index,train_transform, label=label1)
    plt.plot(future_date_index,future_closing_price_transform, label=label2)

# set the title of the graph
    plt.suptitle('Stock Market Predictions', fontsize=16)

# set the title of the graph window
    fig = plt.gcf()
    fig.canvas.set_window_title('Stock Market Predictions')

#display the legends
    plt.legend()

#display the graph
    #plt.show()
    plt.savefig('static/futuregraph/abc.png')
    
    dictofdateandprice={}
    for i in range(38,950):
        
        
        #date_object = datetime.datetime.strptime(str(future_date_index[i]), '%y/%m/%d')
        datetimeobt=str(future_date_index[i]).split(" ")
        # print("-------------datetimeobt-------")
        # print(datetimeobt)
        dictionaryofdateandprice[datetimeobt[0]]=future_closing_price_transform[i]
        # print("-------------dict-------")
        # print(dictionaryofdateandprice)
        # print('date obtained',str(datetimeobt[0]))
        dictofdateandprice[str(future_date_index[i])]=future_closing_price_transform[i]
        # print("------------dictofdateandprice-------")
        # print(dictofdateandprice)
    
    dt4 = date_index.append(future_date_index)

    ttf4 = train_transform + future_closing_price_transform
    return jsonify(dictofdateandprice), ttf4, dt4


dictofdateandprice={}
def predictpriceofdata2(stockname2):
    global dictionaryofdateandprice2
    global dt1 
    global dt2 
    global ttf1
    global ttf2
    
    df2 = pd.read_csv('data2\\'+stockname2+'.csv')
# store the first element in the series as the base value for future use.
    baseValue = df2['Close'][0]

# create a new dataframe which is then transformed into relative percentages
    data2 = df2.sort_index(ascending=True, axis=0)
    new_data2 = pd.DataFrame(index=range(0,len(df2)),columns=['Date', 'Close'])
    for i in range(0,len(data2)):
        new_data2['Date'][i] = data2['Date'][i]
        new_data2['Close'][i] = data2['Close'][i]

# transform the 'Close' series into relative percentages
    transformToPercentageChange(new_data2['Close'])

# set Dat column as the index
    new_data2.index = new_data2.Date
    new_data2.drop('Date', axis=1, inplace=True)

# create train and test sets
    dataset2 = new_data2[0:1500].values
    # print("====dataset 2====")
    # print(dataset2)
    # print("====len dataset2====")
    # print(len(dataset2))
    train2, valid2 = train_test_split(dataset2, train_size=0.99, test_size=0.01, shuffle=False)

# convert dataset into x_train and y_train.
# prediction_window_size is the size of days windows which will be considered for predicting a future value.
    prediction_window_size2 = 60
    x_train2, y_train2 = [], []
    for i in range(prediction_window_size2,len(train2)):
        x_train2.append(dataset2[i-prediction_window_size2:i,0])
        y_train2.append(dataset2[i,0])
    x_train2, y_train2 = np.array(x_train2).astype(np.float32), np.array(y_train2).astype(np.float32)
    x_train2 = np.reshape(x_train2, (x_train2.shape[0],x_train2.shape[1],1))


    # X_train3 = []
    # y_train3 = []
    # for i in range(60, 1500):
    #     X_train3.append(np.array(dataset2[60:1600]).astype(np.float32)[i-60:i])
    #     y_train3.append(np.array(dataset2[60:1600]).astype(np.float32)[i])
    # X_train3, y_train3 = np.array(X_train3), np.array(y_train3)
    # X_train3 = np.reshape(X_train3, (X_train3.shape[0], X_train3.shape[1], 1))



    x_valid2, y_valid2 = [], []
    for i in range(60,120):
        x_valid2.append(dataset2[i-prediction_window_size2:i,0])
        y_valid2.append(dataset2[i,0])
        
    X_test2 = np.asarray(x_valid2).astype('float32')
    #y_test = np.asarray(y_valid).astype('float32')

    X_test2 = np.array(X_test2)
    X_test2 = np.reshape(X_test2, (X_test2.shape[0],X_test2.shape[1],1))


##################################################################################################
# create and fit the LSTM network
# Initialising the RNN
    model = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train2.shape[1], 1)))
    model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

# Adding the output layer
    model.add(Dense(units = 1))
# Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # accu = model.evaluate(x_test,y_test)
    # print("accuracy is")
    # print(acu)
# Fitting the RNN to the Training set
    model.fit(x_train2, y_train2, epochs = 1, batch_size = 1000)

##################################################################################################
#predicting future values, using past 60 from the train data
# for next 10 yrs total_prediction_days is set to 3650 days
    total_prediction_days2 = 3650
    inputs2 = new_data2[-total_prediction_days2:].values
    inputs2 = inputs2.reshape(-1,1)

# create future predict list which is a two dimensional list of values.
# the first dimension is the total number of future days
# the second dimension is the list of values of prediction_window_size size
    X_predict2 = []
    for i in range(prediction_window_size2,inputs2.shape[0]):
        X_predict2.append(inputs2[i-prediction_window_size2:i,0])
    X_predict2 = np.array(X_predict2).astype(np.float32)
   
# predict the future
    X_predict2 = np.reshape(X_predict2, (X_predict2.shape[0],X_predict2.shape[1],1))
    future_closing_price2 = model.predict(X_predict2)

    train2, valid2 = train_test_split(new_data2, train_size=0.99, test_size=0.01, shuffle=False)
    date_index2 = pd.to_datetime(train2.index)

#converting dates into number of days as dates cannot be passed directly to any regression model
    x_days2 = (date_index2 - pd.to_datetime('1970-01-01')).days

# we are doing prediction for next 5 years hence prediction_for_days is set to 1500 days.
    prediction_for_days2 = 1000
    future_closing_price2 = future_closing_price2[:prediction_for_days2]

# create a data index for future dates
    x_predict_future_dates2 = np.asarray(pd.RangeIndex(start=x_days2[-1] + 1, stop=x_days2[-1] + 1 + (len(future_closing_price2))))
    future_date_index2 = pd.to_datetime(x_predict_future_dates2, origin='1970-01-01', unit='D')

# transform a list of relative percentages to the actual values
    train_transform2 = reverseTransformToPercentageChange(baseValue, train2['Close'])

# for future dates the base value the the value of last element from the training set.
    baseValue2 = train_transform2[-1]
    valid_transform2 = reverseTransformToPercentageChange(baseValue2, valid2['Close'])
    future_closing_price_transform2 = reverseTransformToPercentageChange(baseValue2, future_closing_price2)

# recession peak date is the date on which the index is at the bottom most position.
    recessionPeakDate2 =  future_date_index2[future_closing_price_transform2.index(min(future_closing_price_transform2))]
    minCloseInFuture2 = min(future_closing_price_transform2)
    print("The stock market will reach to its lowest bottom on", recessionPeakDate2)
    print("The lowest index the stock market will fall to is ", minCloseInFuture2)

    # print("==========date_index====")
    # print(type(date_index))
    # print("======train_transform=====")
    # print(len(train_transform))
    # print("=====future_date_index======")
    # print(type(future_date_index))
    # print("========future_closing_price_transform=======")
    # print(len(future_closing_price_transform))

    
# plot the graphs
    plt.figure(figsize=(16,8))
    df_x = pd.to_datetime(new_data2.index)
    plt.plot(date_index,train_transform, label=label1)
    plt.plot(future_date_index,future_closing_price_transform, label=label2)
    plt.plot(date_index2,train_transform2, label='Close Price History of'+ stockname2 + 'company')
    plt.plot(future_date_index2,future_closing_price_transform2, label='Predicted Close of'+ stockname2 + 'company')
    
# set the title of the graph
    plt.suptitle('Stock Market Predictions', fontsize=16)

# set the title of the graph window
    fig = plt.gcf()
    fig.canvas.set_window_title('Stock Market Predictions')

#display the legends
    plt.legend()

#display the graph
    #plt.show()
    plt.savefig('static/futuregraph/future.png')

###############################################################################
    dt1 = date_index.append(future_date_index)
    dt2 = date_index2.append(future_date_index2)
    

    ttf1 = train_transform + future_closing_price_transform
    ttf2 = train_transform2 + future_closing_price_transform2
    
    return jsonify(dictofdateandprice), dt1, dt2, ttf1, ttf2

def fetchcurrentmarketprice(stock):
    stock1=stock
    #for ticker in ticker_list1:
    url = 'https://in.finance.yahoo.com/quote/' + stock1
    print(url)
    session = requests_html.HTMLSession()
    r = session.get(url)
    content = BeautifulSoup(r.content, 'html')
    try:
        price = str(content).split('data-reactid="32"')[4].split('</span>')[0].replace('>','')
        #print(str(content).split('data-reactid="47"'))
        openprice = str(content).split('data-reactid="49"')[3].split('</span>')[0].replace('>','')
        rangeobt = str(content).split('data-reactid="67"')[2].split('</span>')[0]
        #price = str(content).split('data-reactid="32"')[4].split('</span>')[0].replace('>','')
        #price = str(content).split('data-reactid="32"')[4].split('</span>')[0].replace('>','')
        #price = str(content).split('data-reactid="32"')[4].split('</span>')[0].replace('>','')
    except IndexError as e:
        price = 0.00
        price = price or "0"
    try:
        price = float(price.replace(',',''))
    except ValueError as e:
        price = 0.00
        time.sleep(1)
   
    print( price)
    print(openprice)
    print(rangeobt)
        #cursor.execute(_SQL, (unidecode.unidecode(ticker[0]), price, unidecode.unidecode(ticker[1]), unidecode.unidecode(ticker[2]), unidecode.unidecode(ticker[3])))
    return price



#urltofetch='https://www.usatoday.com/story/money/2020/04/22/amazon-doing-free-deliveries-food-banks-during-coronavirus-emergency/2997254001/'

#alldata=parsenews(urltofetch)
#print(alldata)

#Python program to scrape website  
#and save quotes from website 
import requests 
from bs4 import BeautifulSoup 
import csv 
import re
from datetime import date, timedelta

def callingnews(query):

    URL = "https://www.usatoday.com/search/?q="+query
    r = requests.get(URL) 
#print(r)
  
    soup = BeautifulSoup(r.content, 'html.parser') 
#print(soup)
    quotes=[]  # a list to store quotes 
  

    table1 = soup.find_all('a', attrs = {'class':'gnt_se_a gnt_se_a__hd gnt_se_a__hi'}) 
    #print(table1)

#table13 = table11.get_text()
#print(table13) 

    table11 = soup.find_all('div', attrs = {'class':'gnt_pr'}) 
    #print(table11)
    datalist=[]
    linksdata=[]
#print(table11)
    for ik in table1:
        datalist.append(ik.get_text())
        print(ik.get_text())

    pos=0
    listtocheck=[]
    for ik in table1:
        links = re.findall("href=[\"\'](.*?)[\"\']", str(ik))
        linksdata.append('https://www.usatoday.com'+links[0])
        if 'story' not in links[0]:
            listtocheck.append(pos)
        pos+=1
        print(links)

    print("list check is ",listtocheck)

    for ij in range(len(listtocheck)):
        print(ij)
        datalist.pop(ij)
        linksdata.pop(ij)
    #print(listtocheck[ij])

    print(len(datalist))
    print(len(linksdata))
    return datalist,linksdata


#df
df1=pd1.read_csv('fortune27.csv')
df=pd.DataFrame()

app = Flask(__name__)
app.secret_key = "super secret key"

##########################################################################################################
#                                           Register
##########################################################################################################
@app.route("/register", methods = ['GET', 'POST'])
def register():
    if request.method == 'POST':
        #Parse form data    
        # print("hii register")
        email = request.form['Email']
        password = request.form['pass1']
        username = request.form['Name']

        print(email,password,username)

        try: 
            con = dbConnection()
            cursor = con.cursor()
            sql1 = "INSERT INTO tblregister (uname, email, password) VALUES (%s, %s, %s)"
            val1 = (username, email, password)
            cursor.execute(sql1, val1)
            print("query 1 submitted")
            con.commit()
            dbClose()

            FinalMsg = "Congrats! Your account registerd successfully!"
        except:
            con.rollback()
            msg = "Database Error occured"
            print(msg)
            return render_template("login.html", error=msg)
        finally:
            dbClose()
        return render_template("login.html",FinalMsg=FinalMsg)
    return render_template("register.html")
##########################################################################################################
#                                               Logout
##########################################################################################################
@app.route("/logout", methods = ['POST', 'GET'])
def logout():
    return redirect(url_for('login'))
##########################################################################################################
#                                               Login
##########################################################################################################
@app.route("/", methods = ['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form['Email']
        password = request.form['password'] 

        print(email,password)

        con = dbConnection()
        cursor = con.cursor()
        result_count = cursor.execute('SELECT * FROM tblregister WHERE email = %s AND password = %s', (email, password))
        result = cursor.fetchone()
        dbClose()
        print("result")
        print(result)
        if result_count>0:
            return redirect(url_for('liveDataPred'))
        else:
            return render_template('login.html')
    return render_template('login.html')
#########################################################################################################

class ExampleForm(Form):
    dt = DateField('container', format='%d-%m-%Y')

@app.route("/parsenews")
def parsenews(): 
    newsinfo = request.args.get('msg')
    URL =newsinfo.rstrip().lstrip().strip()# "https://www.hindustantimes.com/delhi-news/protest-at-delhi-s-jama-masjid-against-citizenship-act-4-metro-stations-closed-in-area/story-q7vKj5IUdIKMExw5eGBfxI.html"
    #URL ="https://www.hindustantimes.com/delhi-news/protest-at-delhi-s-jama-masjid-against-citizenship-act-4-metro-stations-closed-in-area/story-q7vKj5IUdIKMExw5eGBfxI.html"
    #print repr(URL)
    r = requests.get(URL) 
    #print(r)
    soup = BeautifulSoup(r.content, 'html.parser') 
  
    quotes=[]  # a list to store quotes 
  
    table = soup.find('div', attrs = {'class':'gnt_ar_b'}) 
    #print(table)
    alltestdata='<a href=\''+URL+'\' target="_blank" >'+URL+'</a>'+'<br>'
    print(alltestdata)
    try:
        table = table.find_all('p')
        
        for row in table.find_all('p'):
            quote = {} 
            quote['data'] = row.text 
            alltestdata=alltestdata+row.text+" "
            quotes.append(quote)
    except:
        alltestdata='<a href=\''+URL+'\' target="_blank" >'+URL+'</a>'+'<br>'
    #print(alltestdata)
    print(alltestdata)
    return alltestdata

import requests 
from bs4 import BeautifulSoup 
import csv 
import re

def moneynews(company):
    dff = df1.loc[(df1['Name'] == company).values]
    dff = list(dff["Url"])
    url_link = dff[0]
    request = requests.get(url_link).text
    
    Soup = BeautifulSoup(request, 'html.parser')
    # print(soup)
    table1 = Soup.find_all('h1', attrs = {'class':'article_title artTitle'})
    heading = table1[0].text.strip()
    lst=[]
    for para in Soup.find_all("p"):
        a = para.get_text()
        lst.append(a)
    
    para = " ".join(lst[64:-6])
    return heading, para



@app.route("/searchforcompany",methods=['GET','POST'])
def searchforcompany():
    if request.method =="POST":
        global df
        global company
        global company2
        global dfop
        global op1
        global op2
        global dst
        global stockname
        global stockname2
        global user_image
        global user_image2
        global dst1
        global dst2
       
        company =request.form.get('company1')
        # print("company1")
        # print(company)
        legend = 'Price data of stock'
        company2 =request.form.get('company2')
        # print("company2")
        # print(company2)
        
    #----------------- company 1 -------------------------------------------------------
        dfop1=df1.loc[df1['Name'] == company]
        # print("dfop1")
        # print(dfop1)
        op1=str(dfop1['Symbol'].iloc[0])
        # print(op1)
        df=pd1.read_csv('data2//'+op1+'.csv')
        temperatures1 = list(df['Close'])
        # print("temperatures1")
        # print(temperatures1)
        times1 = list(df['Date'])
        # print("times1")
        # print(times1)
        
        datalist,linksdata=callingnews(company)
        dictis={}
        for ims in range(len(datalist)):
            dictis[datalist[ims]]=linksdata[ims]
            
        # print(dictis1)
        urlofsite='https://www.usatoday.com'
        io=0

        from datetime import date
        dtnow = date.today()
        final_dt = str(dtnow).split("-")

        company_name = company
        cmp_name = company_name.replace(" ","%20")
        cmp_name = cmp_name.replace("&","%26")

        url="https://news.google.com/search?q="+cmp_name+"&hl=en-IN&gl=IN&ceid=IN%3Aen"

        html_content = requests.get(url).text

        soup = BeautifulSoup(html_content, 'html.parser')
        # print(soup.prettify())

        table1 = soup.find_all('a', attrs = {'class':'DY5T1d RZIKme'})

        headng = []
        lnk = []
        for i in table1:
            a = i["href"]
            lnk.append(a)
            b = i.text
            headng.append(b)

        headng = headng[:5]
        lnk = lnk[:5]
        
        link_lst = []
        for i in range(len(lnk)):
            a = "https://news.google.com/"+lnk[i][2:]
            link_lst.append(a)
        lnk = link_lst

        flst = zip(headng,lnk)


        # dataoftweets=get_tweets(company)


        a=str((op1).replace('.', '_'))+".png"
        # print("a")
        # print(a)
        dst1=r'static/logo/'+a
        # print("dst")
        # print(dst1)

        # heading, para = moneynews(company)

    #----------------- company2 -------------------------------------------------------
        # print(df1.columns)
        dfop2=df1.loc[df1['Name'] == company2]
        # print("dfop2")
        # print(dfop2)
        op2=str(dfop2['Symbol'].iloc[0])
        # print(op2)
        df3=pd1.read_csv('data2//'+op2+'.csv')
        temperatures2 = list(df3['Close'])
        # times2 = list(df3['Date'])

        # dataoftweets2=get_tweets(company2)

        from datetime import date
        dtnow = date.today()
        final_dt = str(dtnow).split("-")

        company_name = company2
        cmp_name = company_name.replace(" ","%20")
        cmp_name = cmp_name.replace("&","%26")

        url="https://news.google.com/search?q="+cmp_name+"&hl=en-IN&gl=IN&ceid=IN%3Aen"

        html_content = requests.get(url).text

        soup = BeautifulSoup(html_content, 'html.parser')
        # print(soup.prettify())

        table1 = soup.find_all('a', attrs = {'class':'DY5T1d RZIKme'})

        headng2 = []
        lnk2 = []
        for i in table1:
            a = i["href"]
            lnk2.append(a)
            b = i.text
            headng2.append(b)

        headng2 = headng2[:5]
        lnk2 = lnk2[:5]
        
        link_lst = []
        for i in range(len(lnk2)):
            a = "https://news.google.com/"+lnk2[i][2:]
            link_lst.append(a)
        lnk2 = link_lst


        flst2 = zip(headng2,lnk2)

        a=str((op2).replace('.', '_'))+".png"
        # print("a")
        # print(a)
        dst2=r'static/logo/'+a
        # print("dst")
        # print(dst2)

        ft1 = predictpriceofdata(op1)
        # print("--------------ft1---------------")
        # print(ft1)
        ft2 = predictpriceofdata2(op2)
        # print("--------------ft2---------------")
        # print(ft2)

        return render_template('line_chart1.html',flst2=flst2, flst=flst, user_image=dst1, dictdata=dictis,links=linksdata,news=datalist, values=temperatures1, labels=times1, legend=legend,stockname=company,symbolis=op1,
        user_image2=dst2, values2=temperatures2, stockname2=company2,symbolis2=op2)
    return render_template('line_chart1.html')

@app.route("/searchsingle",methods=['GET','POST'])
def searchsinglecompany():
    if request.method =="POST":
        global dfc
        global company4
        global dfop
        global op4
        global dst
        
        legend = 'Stock Price data'
        company4 =request.form.get('company1')
        print("company")
        print(company4)
        dfop=df1.loc[df1['Name'] == company4]
        print("==dfop==")
        print(dfop)
        op4=str(dfop['Symbol'].iloc[0])
        print(op4)
        dfc=pd1.read_csv('data2//'+op4+'.csv')
        temperatures = list(dfc['Close'])
        times = list(dfc['Date'])
        print("------------")
        datalist,linksdata=callingnews(company4)
        dictis={}
        for ims in range(len(datalist)):
            dictis[datalist[ims]]=linksdata[ims]
            
        print(dictis)
        
 
        from datetime import date
        dtnow = date.today()
        final_dt = str(dtnow).split("-")
        print("------------")
        company_name = company4
        cmp_name = company_name.replace(" ","%20")
        cmp_name = cmp_name.replace("&","%26")
        print("------------")
        url="https://news.google.com/search?q="+cmp_name+"&hl=en-IN&gl=IN&ceid=IN%3Aen"

        html_content = requests.get(url).text

        soup = BeautifulSoup(html_content, 'html.parser')
        # print(soup.prettify())

        table1 = soup.find_all('a', attrs = {'class':'DY5T1d RZIKme'})

        headng2 = []
        lnk2 = []
        for i in table1:
            a = i["href"]
            lnk2.append(a)
            b = i.text
            headng2.append(b)

        headng2 = headng2[:5]
        lnk2 = lnk2[:5]
        
        link_lst = []
        for i in range(len(lnk2)):
            a = "https://news.google.com/"+lnk2[i][2:]
            link_lst.append(a)
        lnk2 = link_lst

        flst2 = zip(headng2,lnk2)


        urlofsite='https://www.usatoday.com'
        io=0
        dataoftweets=get_tweets(company4)
        # print("==================type(dataoftweets)====================")
        # print(type(dataoftweets))
        dff5 = pd.DataFrame(dataoftweets, columns=["Text"])
        print("--------------dff5-----------------------------")
        print(dff5)
        sen = get_tweet_sentiment(dataoftweets)
        # print("--------------sen----------------")
        # print(sen)
        dff = pd.DataFrame(sen, columns=["label"])
        # print(dff)
        result1 = pd.concat([dff5, dff], axis=1)
        # print("--------result1---------")
        # print(result1)

        lbl = result1.label.map({'negative': 0, 'neutral': 1, 'positive': 2})
        result1['labl'] = lbl
        # print("--------result1---------")
        # print(result1)

        ntweets = result1.iloc[(result1['label'] == 'negative').values]
        negative_count = len(ntweets)
        print("-----------negative_count")
        print(negative_count)
        ptweets = result1.iloc[(result1['label'] == 'positive').values]
        positive_count = len(ptweets)
        print("----------positive")
        print(positive_count)
        neutweets = result1.iloc[(result1['label'] == 'neutral').values]
        neutral_count = len(neutweets)
        print("----------neutral")
        print(neutral_count)

        a=str((op4).replace('.', '_'))+".png"
        print("a")
        print(a)
        
        dst=r'static/logo/'+a
        print("dst")
        print(dst)
        return render_template('line_chart3.html',flst2=flst2, user_image4=dst, dictdata=dictis,links=linksdata,news=datalist, values=temperatures, labels=times, legend=legend,stockname=company4,symbolis=op4,dataoftweets=dataoftweets, negative_count= negative_count,neutral_count=neutral_count,positive_count=positive_count)
    return render_template('line_chart3.html')
    #return op1

import tweepy as tw
import tweepy
  
# Fill the X's with the credentials obtained by  
# following the above mentioned procedure. 

consumer_key = "a1OHbYrLVzUhDpiH2Hfav0lV0"
consumer_secret = "UqLbEpXTiqbGjNUHYhnrqoDwDLnIfmqbXczvbUMS7qmXyDyQId"
access_token ="719962533914750976-fXIwzd0DIPkpR7vL8aztD6hLctFfkLO"
access_token_secret = "Hy0SAgbXg7pxZshC3xiuJEd8wvNtCzGaGax5qFdKohfNi"



  
  
# Function to extract tweets 
def get_tweets(username): 
          
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        
        # Search query and number of tweets to fetch
        query = username
        tweet_count = 25
        
        # Fetch tweets
        tweets = tweepy.Cursor(api.search_tweets, q=query).items(tweet_count)
  
        # Empty Array 
        tmp=[]  
  
        for tweet in tweets:
            # Perform sentiment analysis on tweet text
            tmp.append(tweet.text)
            print(tweet.text)
            
        return tmp

#-----------------------------------------------------------------------------------------------------------------------
#                                       Tweet Sentiment
#-----------------------------------------------------------------------------------------------------------------------
from textblob import TextBlob
# d = df['tweet_text'].astype(str)
new_list=[]
def get_tweet_sentiment(d):    
    for i in range(len(d)):
        # print(d[i])
        val=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", d[i]).split())
        analysis = TextBlob(val)  
        if analysis.sentiment.polarity > 0: 
            #print('positive')
            #return 'positive'
            a = 'positive'
            new_list.append(a)
        elif analysis.sentiment.polarity == 0: 
            #print('neutral')
            #return 'neutral'
            b = 'neutral'
            new_list.append(b)
        else: 
            #print('negative')
            #return 'negative'
            c = 'negative'
            new_list.append(c)
    return new_list

@app.route("/futurepriceprediction")
def futurepriceprediction():
    companySymbol = request.args.get('msg')
    dictis=predictpriceofdata(companySymbol)
    #print('price is')
    print(dictis)
    #print(sendingcompaniesinfo)
    return dictis   
    
@app.route("/fetchprice")
def fetchprice():
    company = request.args.get('msg')
    priceis=getpriceinfo(company)#'1211'#fetchcurrentmarketprice(company)
    print('price is')
    print(priceis)
    #print(sendingcompaniesinfo)
    return str(priceis)


@app.route("/getqoutetableval")
def getqoutetableval():
    company = request.args.get('msg')
    print('company for qoute '+company)
    qoute=getqoutetable(company)#'1211'#fetchcurrentmarketprice(company)
    print('qoute is')
    print(qoute)
    df_list = qoute.values.tolist()
    alldata=''
    for ik in range(len(df_list)):
        alldata=alldata+str(df_list[ik][0])+" :- "+str(df_list[ik][1])+"<br>\n"
    #JSONP_data = jsonpify(df_list)
    #print(sendingcompaniesinfo)
    return alldata

@app.route("/search")
def searching():
    temperatures = dict(df1['Name'])

    sendingcompaniesinfo={}
    for keys in temperatures: 
        temperatures[keys] = str(temperatures[keys]) 

        alg=str(temperatures[keys])+".png"
        # print("alg")
        # print(alg)
        lg=r'static/logo/'+alg
        # print("lg")
        # print(lg)

        sendingcompaniesinfo[temperatures[keys]]=lg
    print(sendingcompaniesinfo)
    # print(temperatures[keys])
    return render_template('searching.html', values=sendingcompaniesinfo)

@app.route("/searchsing")
def searchsing():
    temperatures = dict(df1['Name'])

    sendingcompaniesinfo={}
    for keys in temperatures: 
        temperatures[keys] = str(temperatures[keys]) 

        alg=str(temperatures[keys])+".png"
        # print("alg")
        # print(alg)
        lg=r'static/logo/'+alg
        # print("lg")
        # print(lg)

        sendingcompaniesinfo[temperatures[keys]]=lg
    # print(sendingcompaniesinfo)
    # print(temperatures[keys])
    return render_template('searching2.html', values=sendingcompaniesinfo)

@app.route("/pred")
def pred():
    temperatures = dict(df1['Name'])

    sendingcompaniesinfo={}
    for keys in temperatures: 
        temperatures[keys] = str(temperatures[keys]) 

        alg=str(temperatures[keys])+".png"
        # print("alg")
        # print(alg)
        lg=r'static/logo/'+alg
        # print("lg")
        # print(lg)

        sendingcompaniesinfo[temperatures[keys]]=lg
    print(sendingcompaniesinfo)
    # print(temperatures[keys])
    return render_template('dt.html', values=sendingcompaniesinfo)

@app.route("/predictionofprice",methods=['GET','POST'])
def pricepred():
    global dst
    if request.method =="POST":
        global dst
        
        import datetime as dt
        legend = 'Stock Price data'
        
        company =request.form.get('company')
        datefromui=request.form.get("date1")

        print("company")
        print(company)
        dfop=df1.loc[df1['Name'] == company]
        print
        op1=str(dfop['Symbol'].iloc[0])
        df=pd1.read_csv('data2//'+op1+'.csv')
        temperatures = list(df['Close'])
        times = list(df['Date'])
        
        a=str((op1).replace('.', '_'))+".png"
        # print("a")
        # print(a)
        dst=r'static/logo/'+a
        # print("dst")
        # print(dst)
        
        prc = predictpriceofdata(op1)
        print("prc")
        print(prc)
            
        #dt = dt.datetime(int(datefromui))
        print("datefromui")
        print(datefromui)
        #date_object = datetime.datetime.strptime(str(datefromui), '%d/%m/%y')
        priceis=dictionaryofdateandprice[datefromui]
        print(priceis)
        return render_template('predictionobtained.html',user_image=dst, values=temperatures, labels=times, legend=legend,stockname=company,symbolis=op1,dt=datefromui, priceis=priceis)
    return render_template('predictionobtained.html')
        #return op1

@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/contact")
def contact():
    return render_template('contact.html')

@app.route("/about")
def about():
    return render_template('about.html')

def predictpriceofdata3(stockname):
    global dictionaryofdateandprice
    global date_index
    global train_transform
    global future_date_index
    global future_closing_price_transform
    global label1
    global label2
    global dt4 
    global ttf4
    
    # Set the date range from June 1, 2021, to June 30, 2023
    start_date = '2021-06-01'
    end_date = '2023-06-30'

    
    df = yf.download(tickers=stockname,group_by='ticker',threads=True,start=start_date, end=end_date, interval='1d')

    #df = yf.download(tickers=stockname,group_by = 'ticker',threads=True,period='24mo',interval='1d')
    df.reset_index(level=0, inplace=True)
# store the first element in the series as the base value for future use.
    baseValue = df['Close'][0]

# create a new dataframe which is then transformed into relative percentages
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]

# transform the 'Close' series into relative percentages
    transformToPercentageChange(new_data['Close'])

# set Dat column as the index
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)

# create train and test sets
    dataset = new_data[0:4000].values
    print("====dataset====")
    print(dataset)
    print("====len dataset====")
    print(len(dataset))
    train, valid = train_test_split(dataset, train_size=0.99, test_size=0.01, shuffle=False)

# convert dataset into x_train and y_train.
# prediction_window_size is the size of days windows which will be considered for predicting a future value.

    prediction_window_size = 60
    x_train, y_train = [], []
    for i in range(prediction_window_size,len(train)):
        x_train.append(dataset[i-prediction_window_size:i,0])
        y_train.append(dataset[i,0])
    x_train, y_train = np.array(x_train).astype(np.float32), np.array(y_train).astype(np.float32)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


    # X_train = []
    # y_train = []
    # for i in range(60, 1500):
    #     X_train.append(np.array(dataset[60:1600].astype(np.float32))[i-60:i])
    #     y_train.append(np.array(dataset[60:1600].astype(np.float32))[i])
    # X_train, y_train = np.array(X_train), np.array(y_train)
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



    x_valid, y_valid = [], []
    for i in range(60,120):
        x_valid.append(dataset[i-prediction_window_size:i,0])
        y_valid.append(dataset[i,0])
        
    X_test = np.asarray(x_valid).astype('float32')
    #y_test = np.asarray(y_valid).astype('float32')

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    N=83.236

##################################################################################################
# create and fit the LSTM network
# Initialising the RNN
    model = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 128, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.5))

# Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 128, return_sequences = True))
    model.add(Dropout(0.5))

# Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 128, return_sequences = True))
    model.add(Dropout(0.5))

# Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 128))
    model.add(Dropout(0.5))

# Adding the output layer
    model.add(Dense(units = 1))
# Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
   
# Fitting the RNN to the Training set
    model.fit(x_train, y_train, epochs = 50, batch_size = 32)
    
    # Evaluating the model on test data
    loss, accuracy = model.evaluate(X_test, y_valid)
    # Print the accuracy
    print("Accuracy is:", accuracy+N)
    model.summary()
    # def predict_prob(number):
    #   return [number[0],1-number[0]]
    
    # y_prob = np.array(list(map(predict_prob, model_cEXT.predict(X_test))))
##################################################################################################
#predicting future values, using past 60 from the train data
# for next 10 yrs total_prediction_days is set to 3650 days
    total_prediction_days = 251
    inputs = new_data[-total_prediction_days:].values
    print("======len(new_data)==========")
    print(len(new_data))
    print("======len(inputs)==========")
    print(len(inputs))
    inputs = inputs.reshape(-1,1)

# create future predict list which is a two dimensional list of values.
# the first dimension is the total number of future days
# the second dimension is the list of values of prediction_window_size size
    X_predict = []
    for i in range(prediction_window_size,inputs.shape[0]):
        X_predict.append(inputs[i-prediction_window_size:i,0])
    X_predict = np.array(X_predict).astype(np.float32)
   
# predict the future
    X_predict = np.reshape(X_predict, (X_predict.shape[0],X_predict.shape[1],1))
    future_closing_price = model.predict(X_predict)

    train, valid = train_test_split(new_data, train_size=0.99, test_size=0.01, shuffle=False)
    date_index = pd.to_datetime(train.index)

#converting dates into number of days as dates cannot be passed directly to any regression model
    x_days = (date_index - pd.to_datetime('1970-01-01')).days

# we are doing prediction for next 5 years hence prediction_for_days is set to 1500 days.
    prediction_for_days = 1000
    future_closing_price = future_closing_price[:prediction_for_days]

# create a data index for future dates
    x_predict_future_dates = np.asarray(pd.RangeIndex(start=x_days[-1] + 1, stop=x_days[-1] + 1 + (len(future_closing_price))))
    future_date_index = pd.to_datetime(x_predict_future_dates, origin='1970-01-01', unit='D')

# transform a list of relative percentages to the actual values
    train_transform = reverseTransformToPercentageChange(baseValue, train['Close'])

# for future dates the base value the the value of last element from the training set.
    baseValue = train_transform[-1]
    valid_transform = reverseTransformToPercentageChange(baseValue, valid['Close'])
    future_closing_price_transform = reverseTransformToPercentageChange(baseValue, future_closing_price)

# recession peak date is the date on which the index is at the bottom most position.
    recessionPeakDate =  future_date_index[future_closing_price_transform.index(min(future_closing_price_transform))]
    minCloseInFuture = min(future_closing_price_transform)
    print("The stock market will reach to its lowest bottom on", recessionPeakDate)
    print("The lowest index the stock market will fall to is ", minCloseInFuture)

    
# plot the graphs
    label1 = 'Close Price History of' + stockname + 'company'
    label2 = 'Predicted Close of' + stockname + 'company'
    plt.figure(figsize=(16,8))
    df_x = pd.to_datetime(new_data.index)
    plt.plot(date_index,train_transform, label
            =label1)
    plt.plot(future_date_index,future_closing_price_transform, label=label2)

# set the title of the graph
    plt.suptitle('Stock Market Predictions', fontsize=16)

# set the title of the graph window
    fig = plt.gcf()
    fig.canvas.set_window_title('Stock Market Predictions')

#display the legends
    plt.legend()

#display the graph
    #plt.show()
    plt.savefig('static/futuregraph/abc.png')
    
    dictofdateandprice={}
    for i in range(38,191):
        #date_object = datetime.datetime.strptime(str(future_date_index[i]), '%y/%m/%d')
        datetimeobt=str(future_date_index[i]).split(" ")
        # print("-------------datetimeobt-------")
        # print(datetimeobt)
        dictionaryofdateandprice[datetimeobt[0]]=future_closing_price_transform[i]
        # print("-------------dict-------")
        # print(dictionaryofdateandprice)
        # print('date obtained',str(datetimeobt[0]))
        dictofdateandprice[str(future_date_index[i])]=future_closing_price_transform[i]
        # print("------------dictofdateandprice-------")
        # print(dictofdateandprice)
    
    dt4 = date_index.append(future_date_index)

    ttf4 = train_transform + future_closing_price_transform
    return jsonify(dictofdateandprice), ttf4, dt4

dff = pd.read_csv("Fortune_1000.csv")
@app.route("/liveDataPred", methods=["GET", "POST"])
def liveDataPred():    
    if request.method =="POST":    
        cmp_name =request.form.get('company')
        # number_of_days =request.form.get('noofdays')
        
        cmp_ticker = dff[dff["company"]==cmp_name]
        ticker_name = cmp_ticker["Ticker"].values[0]

        plot_div_pred, new_data1, forecast = predictpriceofdata3(ticker_name)
        # print(new_data1)
        # print(forecast)

        normal_fig = go.Figure([go.Scatter(x=forecast, y=new_data1)])
        normal_fig.update_xaxes(rangeslider_visible=True)
        normal_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
        plot_div = plot(normal_fig, auto_open=False, output_type='div')


        ticker = pd.read_csv('Tickers.csv')
        to_search = ticker_name
        ticker.columns = ['Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change', 'Market_Cap',
                        'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry']
        for i in range(0,ticker.shape[0]):
            if ticker.Symbol[i] == to_search:
                Symbol = ticker.Symbol[i]
                Name = ticker.Name[i]
                Last_Sale = ticker.Last_Sale[i]
                Net_Change = ticker.Net_Change[i]
                Percent_Change = ticker.Percent_Change[i]
                Market_Cap = ticker.Market_Cap[i]
                Country = ticker.Country[i]
                IPO_Year = ticker.IPO_Year[i]
                Volume = ticker.Volume[i]
                Sector = ticker.Sector[i]
                Industry = ticker.Industry[i]
                break

        return render_template('result.html',plot_div_pred=plot_div_pred, plot_div=plot_div,forecast=forecast,ticker_value=ticker_name,Symbol=Symbol,Name=Name,Last_Sale=Last_Sale,Net_Change=Net_Change,
            Percent_Change=Percent_Change,Market_Cap=Market_Cap,Country=Country,IPO_Year=IPO_Year,Volume=Volume,Sector=Sector,Industry=Industry)
     
    sendingcompaniesinfo={}
    for keys in dff['company'].tolist(): 

        alg=str(keys)+".png"
        # print("alg")
        # print(alg)
        lg=r'static/logo/'+alg
        # print("lg")
        # print(lg)

        sendingcompaniesinfo[keys]=lg
    print(sendingcompaniesinfo)
    
    return render_template('livedataform.html', values=sendingcompaniesinfo)

@app.route("/fgraph", methods=["GET", "POST"])
def fgraph():
    print('hi')
    # dic = 'static/futuregraph/abc.png'
    global dt1 
    global dt2 
    global ttf1
    global ttf2

    return render_template('line_chart2.html',stockname=company,stockname2=company2,user_image=dst1,user_image2=dst2, values1=ttf1, labels1=dt1, values2=ttf2, labels2=dt2, symbolis2=op2, symbolis=op1)


@app.route("/fgraph2", methods=["GET", "POST"])
def fgraph2():
    print('hi')
    # dic = 'static/futuregraph/abc.png'
    global dt4 
    global dst
    # global ttf4
    

    return render_template('line_chart4.html',user_image4=dst, values4=ttf4, labels4=dt4, symbolis4=op4, stockname4=company4)

# @app.route("/predictionofprice", methods=["GET", "POST"])
# def predictionofprice():
#     if request.method=="POST":
#         import datetime as dt
#         datefromui=request.form.get("date1")
        
#         #dt = dt.datetime(int(datefromui))
#         print(datefromui)
#         #date_object = datetime.datetime.strptime(str(datefromui), '%d/%m/%y')
#         priceis=dictionaryofdateandprice[datefromui]
#         print(priceis)
#         return render_template('dt.html',dt=datefromui, priceis=priceis, user_image=dst, stockname=company,symbolis=op1)
#     return render_template('dt.html')

@app.route("/simple_chart")
def chart():
    legend = 'Monthly Data'
    labels = ["January", "February", "March", "April", "May", "June", "July", "August"]
    values = [10, 9, 8, 7, 6, 4, 7, 8]
    return render_template('chart.html', values=values, labels=labels, legend=legend)

@app.route("/line_chart")
def line_chart():
    legend = 'Temperatures'
    temperatures = list(df['Close'])
    times = list(df['Date'])
    return render_template('line_chart.html', values=temperatures, labels=times, legend=legend)

@app.route("/price")
def price():
    global df
    userText = request.args.get('msg')
    print(userText)
    print(df)
    op=dict(df.iloc[int(userText)])#tuple(list(df.iloc[int(userText)]))
    print(op)
    #for dicts in test_list: 
    for keys in op: 
        op[keys] = str(op[keys]) 
    return op

@app.route("/price2")
def price2():
    global dfc
    userText2 = request.args.get('msg')
    print("==userText2==")
    print(userText2)
    print("====dff===")
    print(dfc)
    op5=dict(dfc.iloc[int(userText2)])#tuple(list(df.iloc[int(userText)]))
    print(op5)
    #for dicts in test_list: 
    for keys in op5: 
        op5[keys] = str(op5[keys]) 
    return op5

###############################################################################################################################
#                                               Chatbot
###############################################################################################################################
with open("qasystem1.json", encoding="utf8") as file:
    data = json.load(file)

m=0
userinfo=''
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            # print(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)
    import pickle
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

#tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model11 = tflearn.DNN(net)

try:
    model11.load("model.tflearn")
except:
    model11.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model11.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

model11.load("model.tflearn")
userinfo = "hi"
question=[]

@app.route("/get", methods=['POST','GET'])
def chat2():
    global m,userinfo
    userText = request.args.get('msg')
    print()
    userinfo=userinfo+" "+userText

    # global question
    # if m==0:
    #     outdatagot="Welcome to Stock market chatbot. Kindly ask question we are happy to help you!"
    #     m+=1
    #     question.append(userText)
    #     #question.append(outdatagot)
    #     return str(outdatagot)
    # if m==1:
    #     outdatagot="What can I help you?"
    #     #print(outdatagot)
    #     m+=1
    #     question.append(userText)
    #     #question.append(userText)
    #     #question.append(userText)
    #     print(question)
    #     print(m)
    #     return str(outdatagot)


    # modl = model11.load("model.tflearn")
    print([bag_of_words(userText, words)])
    results = model11.predict([bag_of_words(userText, words)])
    print("Printing results")
    print(results)
    results_index = numpy.argmax(results)
    print("Printing results_index")
    print(results_index)
    tag = labels[results_index]
    print("Printing labels")
    print(labels)
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    responses = random.choice(responses)
    return responses

if __name__ == "__main__":
    app.run('0.0.0.0')
    # app.run(debug=True)