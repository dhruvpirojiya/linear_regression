import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error, r2_score
import yfinance as yf
st.title('stock forecasting')
user_input = st.text_input('enter stock ticker')

# Fetching data for Apple Inc. from '12-12-2016' to '12-12-2017'
df = yf.download(user_input, start='2016-12-12', end='2019-12-12')

# Now you can proceed with your operations on the data


#describing data of 2 year
st.subheader('data from 2019-2020')
st.write(df.describe())

#graph of the given data
st.subheader('closing price vs time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Open)
st.pyplot(fig)

#100 day moving averages
st.subheader('100 day moving averages')
m100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(m100,'g')
plt.plot(df.Close,'r')
st.pyplot(fig)
df.reset_index()
print(df)


# implementation of linear regression 
print('hello dp')

y1 = pd.array(df['Close'])


#data_traning1 = pd.DataFrame(df['Volume'][0:int(len(df)*0.70)])
#data_testing1 = pd.DataFrame(df['Volume'][int(len(df)*0.70):int(len(df))])


#create indepent variables

x = range(755)

for n in x:
  print(n)

x1 = np.array(x).reshape(-1,1)


#n = np.array(data_traning).reshape(-1,1)
#m = np.array(data_traning1)

#n1 = np.array(data_testing).reshape(-1,1)
#m1 = np.array(data_testing1)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1,y1,test_size=0.25,random_state=0)

lr = LinearRegression()

lr.fit(x_train,y_train)
print("intercet",lr.intercept_)
print("slop",lr.coef_)
y_pred = lr.predict(x_test)
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred,color='r')
plt.show()
z = r2_score(y_test, y_pred)
print('effecicy of model:-') 
#fig = plt.figure(figsize=(12,6))
#plt.plot(y_pred,'b')
#st.pyplot(fig)
fig,  ax = plt.subplots()
ax.plot(x_test,y_pred)
st.pyplot(fig)
st.subheader('intercept of data set:-')
st.subheader(lr.intercept_)
st.subheader('slope of data set:-')
st.subheader(lr.coef_)
st.subheader(' effecency of model in pecentage(%):-')
st.subheader(z*100)