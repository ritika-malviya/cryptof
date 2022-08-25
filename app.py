import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px

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
st.sidebar.title("Crpytocurrencies Prediction")
st.sidebar.image(image)
st.subheader("What is Cryptocurrency???")
st.write("""
You must have heard or invested in any cryptocurrency once in your life. 
It is a digital medium of exchange that is encrypted and decentralized. 
Many people use cryptocurrencies as a form of investing because it gives great returns even in a short period. 
Bitcoin, Ethereum, Dogecoin & many more coins are among the popular cryptocurrencies today.
""")
selected_stock= st.sidebar.selectbox("Select the crpytocurrency for prediction",
                                     ("BTC-USD","ETH-USD","XRP-USD","DOGE-USD","ADA-USD",
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
st.subheader("Raw Data")
st.write(data.tail())

#Describing data
st.subheader("Data Description of 5 years :")
st.write(data.describe())

#visualization
def plot_data():
    figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                            open=data["Open"],
                                            high=data["High"],
                                            low=data["Low"],
                                            close=data["Close"])])
    figure.update_layout(title_text = "Interactive Price Chart",xaxis_rangeslider_visible=True)
    st.plotly_chart(figure)

plot_data()

df = data['Close'] # forecast close
from sklearn.preprocessing import MinMaxScaler
import numpy as np
scaler = MinMaxScaler(feature_range = (0,1))
data1 = scaler.fit_transform(np.array(df).reshape(-1,1))

# train test split
train_size = int(len(data1)*0.75)
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

# Comparision of original stock close price and predicted close price
plotdf = pd.DataFrame({'date': data['Date'],
                       'original': data['Close'],
                      'train_predicted': trainPredict.reshape(1,-1)[0].tolist(),
                      'test_predicted': testPredict.reshape(1,-1)[0].tolist()})
fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original'],plotdf['train_predicted'],
                                          plotdf['test_predicted']],
              labels={'value':'Stock price','date': 'Date'})
fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.show()


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
st.subheader("Forecast for next 1 month from" , start_date)
st.write(df_future)
