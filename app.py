import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
loaded_model = load_model('lstm.h5')

from PIL import Image
image = Image.open('crypto.jpg')

today = date.today()
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=1825)  # 5 years data
d2 = d2.strftime("%Y-%m-%d")
start_date = d2
st.title("Cryptocurrencies Prediction using LSTM for next 1 month")
st.image(image)
st.subheader("What is Cryptocurrency???")
st.write("""
You must have heard or invested in any cryptocurrency once in your life. 
It is a digital medium of exchange that is encrypted and decentralized. 
Many people use cryptocurrencies as a form of investing because it gives great returns even in a short period. 
Bitcoin, Ethereum, Dogecoin & many more coins are among the popular cryptocurrencies today.
""")
coins = Image.open("coins.jpg")
st.image(coins)
a = st.write("### Select the crpytocurrency for prediction")
selected_stock= st.selectbox(" Select " ,
                            ("XRP-USD","BTC-USD","ETH-USD","DOGE-USD","ADA-USD",
                             "BNB-USD","DOT-USD","SHIB-USD","TRX-USD","MATIC-USD"))
st.write("### Selected cryptocurrency : ", selected_stock )

def data_load(ticker):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("loading data.....")
data = data_load(selected_stock)
data_load_state.text("loading data... done")

#data
st.subheader("Historical Data")
time_period = st.selectbox("TIME PERIOD",("5 DAYS","15 DAYS","1 MONTH","3 MONTHS","6 MONTHS","1 YEAR"))  
if time_period == "5 DAYS" :
    st.write("Historical Data of past 5 Days")
    st.write(data.tail(5))
elif time_period == "15 DAYS" :
     st.write("Historical Data of past 15 Days")
     st.write(data.tail(15))  
elif time_period == "1 MONTH" :
     st.write("Historical Data of past 1 Month")
     st.write(data.tail(30))  
elif time_period == "3 MONTHS" :
      st.write("Historical Data of past 3 Months")
      st.write(data.tail(90))
elif time_period == "6 MONTHS" : 
      st.write("Historical Data of past 6 Months")
      st.write(data.tail(180))
elif time_period == "1 YEAR" :
       st.write("Historical Data of past 1 year")
       st.write(data.tail(365))   
else :
       st.write("Historical Data of past 10 Days")
       st.write(data.tail(10))

#Describing data
st.subheader("Data Description of past 5 years :")
st.write(data.describe())

#visualization
st.subheader("Interactive Price Chart")
def plot_data():
    figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                            open=data["Open"],
                                            high=data["High"],
                                            low=data["Low"],
                                            close=data["Close"])])
    figure.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(figure)

plot_data()

df = data['Close'] # forecast close
from sklearn.preprocessing import MinMaxScaler
import numpy as np
scaler = MinMaxScaler(feature_range = (0,1))
data1 = scaler.fit_transform(np.array(df).reshape(-1,1))

# train test split
train_size = int(len(data1)*0.80)
test_size = len(data1) - train_size
train_data, test_data = data1[0:train_size,:],data1[train_size:len(data1),:1]

def create_data(dataset,time_step = 1):
    dataX, dataY = [],[]
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# past 100 days
time_step = 100
x_train, y_train = create_data(train_data,time_step)
x_test,y_test = create_data(test_data,time_step)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

train_predict = loaded_model.predict(x_train)
test_predict = loaded_model.predict(x_test)
##Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

st.write("### Comparision between original close price vs predicted close price'")
# Final Graph
# Plotting
look_back = 100
# shift train prediction for plotting 
trainPredict = np.empty_like(data1)
trainPredict[:,:] = np.nan
trainPredict[look_back:len(train_predict)+look_back, :] = train_predict

# shift test prediction
testPredict = np.empty_like(data1)
testPredict[:,:] = np.nan
testPredict[len(train_predict) + (look_back * 2)+1:len(data1)-1, :] = test_predict

# plot baseline and predictions
fig2 = plt.figure(figsize = (12,6))
plt.plot(scaler.inverse_transform(data1))
plt.plot(trainPredict)
plt.plot(testPredict)
plt.title("Comparision between original close price vs predicted close price ")
st.pyplot(fig2)

print("Blue indicates the Complete Data")
print("Green indicates the Predicted Data")
print("Orange indicates the Train Data")


loaded_modelf = load_model('lstmf.h5')
#forecast for next 1 month
#Generate the input and output sequences
n_lookback = 100  # length of input sequences (lookback period)
n_forecast = 30  # length of output sequences (forecast period)

X = []
Y = []

for i in range(n_lookback, len(data1) - n_forecast + 1):
    X.append(data1[i - n_lookback: i])
    Y.append(data1[i: i + n_forecast])

X = np.array(X)
Y = np.array(Y)

# generate the forecasts
X_ = data1[- n_lookback:]  # last available input sequence
X_ = X_.reshape(1, n_lookback, 1)

Y_ = loaded_modelf.predict(X_).reshape(-1, 1)
Y_ = scaler.inverse_transform(Y_)

df_future = pd.DataFrame(columns=['Date','Forecast'])
df_future['Date'] = pd.date_range(start=end_date, periods=n_forecast)
df_future['Forecast'] = Y_.flatten()
st.subheader("Forecast for next 1 month :")
forecast = Image.open("forecast.jpg")
st.image(forecast)
st.write(df_future)

st.caption("Created by Ritika Malviya")
