from sklearn import metrics
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import AutoReg as AR
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
from sklearn.metrics import mean_absolute_error


#-----------------------------------------------------------------

# Q-1

# part a

d=pd.read_csv("daily_covid_cases.csv")

d1=d.copy()

plt.plot(d1['Date'],d1['new_cases'])
plt.ylabel('New confirmed cases')
plt.xlabel('Month-year')
plt.show()

# part b
d1['lagged_1']=d1.new_cases.shift(periods=-1)
print(d1)
corr = d1.corr(method="pearson")
print(corr['lagged_1']['new_cases'])


#part c
plt.scatter(d1['new_cases'],d1['lagged_1'])
plt.xlabel('New confirmed cases')
plt.ylabel('lagged_1 values')
plt.show()
# part d
l= ['lagged_1',  'lagged_2' ,'lagged_3','lagged_4'  , 'lagged_5', 'lagged_6']
for i in l[1:]:
    d1[i]=d1.new_cases.shift(periods=-(l.index(i)+1))
print(d1)
corr2 = d1.corr(method="pearson")
print(corr2)


n=[]

for i in l:
    
    n.append(corr2[i]['new_cases'])
print(n)
plt.plot(['day1','day2','day3','day4','day5','day6'],n)
plt.xlabel('lagged days')
plt.ylabel(' corr values')
plt.show()

# part E

plot_acf(d1['new_cases'],lags=6)
plt.xlabel('lagged days')
plt.ylabel(' corr values')
plt.show()

#-----------------------------------------------------------------

# Q-2

series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
d1=series.copy()
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]


# part 1
window = 5 # The lag=1
model = AR(train, lags=window)
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
print(list(coef))

# part 2

history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
mape = np.mean(np.abs((test - predictions)/test))*100
#mape = mean_absolute_error(test, predictions)*100
print(mape)


plt.plot(test,predictions, color='green')
plt.show()
plt.scatter(test,predictions, color='red')
plt.show()

#-----------------------------------------------------------------

# Q-3

series1 = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
d2=series1.copy()
test_size = 0.35 # 35% for testing
X = series1.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]


RMSE_values=[]
MAPE_values=[]
windows=[1,5,10,15,25]
for j in windows:
    model=AR(train,lags=j)
    model_fit=model.fit()
    coef=model_fit.params
    history = train[len(train)-j:]
    history = [history[i] for i in range(len(history))]
    predictions=[]
    for t in range(len(test)):
        
        length = len(history)
        lag = [history[i] for i in range(length-j,length)]
        yhat = coef[0] # Initialize to w0
        for d in range(j):
            
            yhat += coef[d+1] * lag[j-d-1] # Add other values
        obs = test[t]
        predictions.append(yhat) #Append predictions to compute RMSE later
            
        history.append(obs) 
    RMSE = sqrt(mean_squared_error(test, predictions))
    RMSE_values.append(RMSE)
    test_MAPE= mean_absolute_error(test,predictions)
    MAPE_values.append(test_MAPE)
print(MAPE_values)
print(RMSE_values)
plt.bar(windows, RMSE_values, color ='maroon')
plt.xlabel("Lag Values ----->")
plt.ylabel("RMSE Values ------>")
plt.title("RMSE Plot for different value of lags")
plt.show()
plt.bar(windows, MAPE_values, color ='blue')
plt.xlabel("Lag Values ------->")
plt.ylabel("MAPE Values ------>")
plt.title("MAPE Plot for different value of lags")
plt.show()

#-----------------------------------------------------------------

# Q-4

d_x=pd.read_csv("daily_covid_cases.csv")
series_1 = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
d_1=d_x.copy()
test_size = 0.35 # 35% for testing
X = series_1.values
tst_sz_1 = math.ceil(len(X)*test_size)
train_1, test_1 = X[:len(X)-tst_sz_1], X[len(X)-tst_sz_1:]

la=1
for i in range(1,len(d_1)):
    
    d_1['lagged_'+str(i)]=d_1.new_cases.shift(periods=-i)
    corr = d_1.corr(method="pearson")
    if(abs(corr['lagged_'+str(i)]['new_cases'])<(2/((len(train))**0.5))):
        la=i-1
        break

model_1=AR(train_1,lags=la)
model_fit_1=model_1.fit()
coef_1=model_fit_1.params
history_1 = train[len(train_1)-la:]
history_1 = [history_1[i] for i in range(len(history_1))]
predictions_1=[]
for t in range(len(test_1)):
    
    length_1 = len(history_1)
    lag_1 = [history_1[i] for i in range(length_1-la,length_1)]
    yhat_1 = coef_1[0] # Initialize to w0
    for d in range(la):
        
        yhat_1 += coef_1[d+1] * lag_1[la-d-1] # Add other values
    obs_1 = test_1[t]
    predictions_1.append(yhat_1) #Append predictions to compute RMSE later
        
    history_1.append(obs_1) 
MSE_1 = np.square(np.subtract(test_1,predictions_1)).mean() 
RMSE_1 = math.sqrt(MSE_1)
print("\nRMSE Value ----->",RMSE_1)
test_MAPE_1= mean_absolute_error(test_1,predictions_1)
print("\nMAPE Value ------>",test_MAPE_1)



