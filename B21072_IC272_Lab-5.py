''' Unnat Maaheshwari (B21072)'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Part-A

print("\n Part-A")
print("_________________________________________________________________________")
#import csv file
df = pd.read_csv('SteelPlateFaults-2class.csv')

#column list
col_list=list(df.columns)

#split list in train and test
[df1_train, df1_test] = train_test_split(df,test_size=0.3, random_state=42, shuffle=True)

#copy of test and train data
traingmm=df1_train.copy(deep=True)
testgmm=df1_test.copy(deep=True)

#dataframe of test Class
testclass=testgmm['Class']

#drop bad columns
traingmm.drop(columns=['TypeOfSteel_A300', 'TypeOfSteel_A400','X_Minimum','Y_Minimum'], inplace=True)
testgmm.drop(columns=['TypeOfSteel_A300', 'TypeOfSteel_A400','X_Minimum','Y_Minimum','Class'], inplace=True)

#divide train data
traingmm0 = traingmm[traingmm["Class"] == 0]
traingmm1 = traingmm[traingmm["Class"] == 1]

#drop Class column in both divided dataframes
traingmm0.drop(columns=['Class'], inplace=True)
traingmm1.drop(columns=['Class'], inplace=True)

#list of accuracies
accs=[]

# -------------------------------------------------------------

# Q-1.

print("\n Q-1.")

for i in (2,4,8,16):
    
    #fit train data with class 0 in GMM0 with i clusters
    GMM0 = GaussianMixture(n_components=i, covariance_type='full',reg_covar=1e-5)
    GMM0.fit(traingmm0.values)
    
    #fit train data with class 1 in GMM1 with i clusters
    GMM1 = GaussianMixture(n_components=i, covariance_type='full',reg_covar=1e-5)
    GMM1.fit(traingmm1.values)
    
    #find score for test data with both class 0 and 1
    t0=GMM0.score_samples(testgmm.values)
    t1=GMM1.score_samples(testgmm.values)
    
    #list of predicted class
    p=[]      #a ppend class in list based on score sample of class 0 and 1    

    for j in range(len(t0)):
        if t0[j]>=t1[j]:
            p.append(0)
        elif t0[j]<=t1[j]:
            p.append(1)  
            
    #find accuracy and confusion matrix for each case      
    mat=confusion_matrix(testclass,p)
    acc=accuracy_score(testclass,p)
    
    #append accuracy in list
    accs.append(acc)
    
    #print result    
    print("\n Confusions matrix with Q=",i,":")
    print(mat)
    print("Accuracy with Q=",i,":",acc)

print("\n -----------------------------------------------------------------------")

# -------------------------------------------------------------

# Q-2.

#print max accuracy for each model
print("\n Q-2.")
print("\nMax accuracy for KNN",0.896)
print("Max accuracy for KNN Normalized",1.0)
print("Accuracy of Bayes Classifier",0.9375)
print("Max Accuracy in GMM",round(max(accs),3))


#___________________________________________________________________________________

# Part B

print("\n __________________________________________________________________________")
print("\n __________________________________________________________________________")

print("\n Part-B")
print("_________________________________________________________________________")

file=pd.read_csv("abalone.csv")

x=file[file.columns[:-1]]
pd.DataFrame(x)

y=file[file.columns[-1]]
pd.DataFrame(y)

# -------------------------------------------------------------

# Q-1. 

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,shuffle=True,random_state=42)

x_train.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)
training_data= pd.concat([x_train,y_train],axis=1)
training_data.to_csv("abalone-train.csv")
pd.DataFrame(training_data)

x_test.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)
testing_data= pd.concat([x_test,y_test],axis=1)
testing_data.to_csv("abalone-test.csv")
pd.DataFrame(testing_data)

# Finding attribute with the highest pearson correlation coefficient 
pcc_dict={}
max_pcc=-1000
max_pcc_col="null"
for col in x.columns:
    pcc,_=pearsonr(x[col],y)
    pcc_dict[col]=pcc
    if(max_pcc<pcc):
        max_pcc=pcc
        max_pcc_col=col

print("\n Q-1.")
print("\nAttribute with the highest pearson correlation coefficient:",max_pcc_col)
print("Value of Pearson Correlation Coefficient for that attribute:",max_pcc)

df=pd.DataFrame(pcc_dict.values(),pcc_dict.keys())
df.columns=["PCC"]
pd.DataFrame(df)

# Linear Regression of attribute with highest PCC with Rings
reg=LinearRegression()
pcc_train=np.array(x_train[max_pcc_col])
pcc_train=pcc_train.reshape(1,-1)
pcc_train=pcc_train.transpose()
pcc_test=np.array(x_test[max_pcc_col])
pcc_test=pcc_test.reshape(1,-1)
pcc_test=pcc_test.transpose()
reg.fit(pcc_train,y_train)

prediction_training=reg.predict(pcc_train)
prediction_testing=reg.predict(pcc_test)

plt.scatter(pcc_train,y_train,color="red")
m,c=np.polyfit(pcc_train[:,0],y_train, 1)
plt.plot(pcc_train[:,0], m*pcc_train[:,0] + c,color="black") 

print("\n -------------------------------------------------------")
print("\n (a)")
plt.xlabel(max_pcc_col)
plt.ylabel("Rings")
plt.title("Best Fit Line")
plt.grid()
plt.show()

rmse_training=metrics.mean_squared_error(y_train,prediction_training,squared=False)
rmse_testing=metrics.mean_squared_error(y_test,prediction_testing,squared=False)

print("\n -------------------------------------------------------")
print("\n (b)")
print("RMSE in training:",rmse_training)

print("\n -------------------------------------------------------")
print("\n (c)")
print("RMSE in testing:", rmse_testing)

print("\n -------------------------------------------------------")
print("\n (d)")
plt.scatter(y_test,prediction_testing)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.title("Predicted V/S Actual")
plt.show()

# -------------------------------------------------------------

# Q-2.

# Multivariate Linear Regression
reg.fit(x_train,y_train)

prediction_training=reg.predict(x_train)
prediction_testing=reg.predict(x_test)

rmse_training=metrics.mean_squared_error(y_train,prediction_training,squared=False)
rmse_testing=metrics.mean_squared_error(y_test,prediction_testing,squared=False)

print("\n -----------------------------------------------------------------------")
print("\n Q-2.")

print("\n -------------------------------------------------------")
print("\n (a)")
print("\n RMSE in training:",rmse_training)

print("\n -------------------------------------------------------")
print("\n (b)")
print("\n RMSE in testing:", rmse_testing)

print("\n -------------------------------------------------------")
print("\n (c)")
plt.scatter(y_test,prediction_testing)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.title("Predicted V/S Actual")
plt.show()

# -------------------------------------------------------------

# Q-3.

# Single Variate Non-Linear Regression
reg=LinearRegression()
poly_features2=PolynomialFeatures(2)
poly_features3=PolynomialFeatures(3)
poly_features4=PolynomialFeatures(4)
poly_features5=PolynomialFeatures(5)

poly_features=[poly_features2,poly_features3,poly_features4,poly_features5]
p=["Degree 2","Degree 3","Degree 4","Degree 5"]
poly_dict={}

for features in range(len(poly_features)):
    x_poly_training=poly_features[features].fit_transform(pcc_train)
    x_poly_testing=poly_features[features].fit_transform(pcc_test)

    reg.fit(x_poly_training,y_train)
    
    prediction_training=reg.predict(x_poly_training)
    prediction_testing=reg.predict(x_poly_testing)
    
    rmse_training=metrics.mean_squared_error(y_train,prediction_training,squared=False)
    rmse_testing=metrics.mean_squared_error(y_test,prediction_testing,squared=False)
    
    poly_dict[p[features]]=[rmse_training,rmse_testing]

df=pd.DataFrame(poly_dict.values(),poly_dict.keys())
df.columns=["RMSE-Train","RMSE-Test"]
pd.DataFrame(df)

# It can be seen Degree 4 has the least error on Test data set
rmse_list=list(poly_dict.values())
rmse_train_list=[]
rmse_test_list=[]
for lists in rmse_list:
    rmse_train_list.append(lists[0])
    rmse_test_list.append(lists[1])

print("\n -----------------------------------------------------------------------")
print("\n Q-3.")

print("\n -------------------------------------------------------")
print("\n (a)")
plt.bar(p,rmse_train_list)
plt.plot(p,rmse_train_list,"-o",color="black")
plt.xlabel("Degrees")
plt.ylabel("RMSE Errors")
plt.title("Train Data")
plt.show()

print("\n -------------------------------------------------------")
print("\n (b)")
plt.bar(p,rmse_test_list)
plt.plot(p,rmse_train_list,"o-",color="black")
plt.plot(p,rmse_test_list,"*-",color="red")
plt.title("Test Data")
plt.show()

def best_fit_curve(x,a,b,c,d,e):
    curve=a*x**4+b*x**3+c*x**2+d*x+e
    return curve

x_poly_training=poly_features4.fit_transform(pcc_train)
x_poly_testing=poly_features4.fit_transform(pcc_test)
reg.fit(x_poly_training,y_train)
prediction_training=reg.predict(x_poly_training)
prediction_testing=reg.predict(x_poly_testing)
plt.scatter(pcc_train,y_train)

parameters,_=curve_fit(best_fit_curve,pcc_train[:,0],prediction_training)
a,b,c,d,e=parameters

component_x=np.linspace(min(pcc_train),max(pcc_train),200)
component_y=best_fit_curve(component_x,a,b,c,d,e)

plt.plot(component_x,component_y,color="black")

print("\n -------------------------------------------------------")
print("\n (c)")
plt.xlabel(max_pcc_col)
plt.ylabel("Rings")
plt.title("Best Fit Curve")
plt.grid()
plt.show()

print("\n -------------------------------------------------------")
print("\n (d)")
plt.scatter(y_test,prediction_testing)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.title("Predicted V/S Actual")
plt.show()

# -------------------------------------------------------------

# Q-4.

# Multi-variate non-linear regression
for features in range(len(poly_features)):
    x_poly_training=poly_features[features].fit_transform(x_train)
    x_poly_testing=poly_features[features].fit_transform(x_test)
    reg.fit(x_poly_training,y_train)    
    prediction_training=reg.predict(x_poly_training)
    prediction_testing=reg.predict(x_poly_testing)    
    rmse_training=metrics.mean_squared_error(y_train,prediction_training,squared=False)
    rmse_testing=metrics.mean_squared_error(y_test,prediction_testing,squared=False)    
    poly_dict[p[features]]=[rmse_training,rmse_testing]

df=pd.DataFrame(poly_dict.values(),poly_dict.keys())
df.columns=["RMSE-Train","RMSE-Test"]
pd.DataFrame(df)

# Degree 2 is the best for multivariate polynomial regression
rmse_list=list(poly_dict.values())
rmse_train_list=[]
rmse_test_list=[]
for lists in rmse_list:
    rmse_train_list.append(lists[0])
    rmse_test_list.append(lists[1])

print("\n -----------------------------------------------------------------------")
print("\n Q-4.")

print("\n -------------------------------------------------------")
print("\n (a)")
plt.bar(p,rmse_train_list)
plt.plot(p,rmse_train_list,"-o",color="black")
plt.xlabel("Degrees")
plt.ylabel("RMSE Errors")
plt.title("Train Data")
plt.show()

print("\n -------------------------------------------------------")
print("\n (b)")
plt.bar(p,rmse_test_list)
plt.plot(p,rmse_train_list,"o-",color="black")
plt.plot(p,rmse_test_list,"*-",color="red")
plt.title("Test Data")
plt.show()

x_poly_training=poly_features2.fit_transform(x_train)
x_poly_testing=poly_features2.fit_transform(x_test)
reg.fit(x_poly_training,y_train)
prediction_testing=reg.predict(x_poly_testing)

print("\n -------------------------------------------------------")
print("\n (c)")
plt.scatter(y_test,prediction_testing)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.title("Predicted V/S Actual")
plt.show()