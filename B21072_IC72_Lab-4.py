''' Unnat Maaheshwari (B21072)'''

#Importing libraries to be used in the program
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB

  
data = pd.read_csv('SteelPlateFaults-2class.csv')

class_0 = data.loc[data['Class']==0]
class_1 = data.loc[data['Class']==1]

x_0 = class_0['Class']
x_1 = class_1['Class']

f0=class_0.drop(['Class'],axis=1) 
f1=class_1.drop(['Class'],axis=1)

[train_0,test_0,ctr0,cte0] = train_test_split(f0,x_0 ,test_size=0.3, random_state=42, shuffle=True)
[train_1,test_1,ctr1,cte1] = train_test_split(f1,x_1, test_size=0.3, random_state=42, shuffle=True)

#formation of training set (X)
train=[train_0,train_1]
train=pd.concat(train)
train.to_csv('SteelPlateFaults-train.csv') #saving training data

#formation of test set
test=[test_0,test_1]
test=pd.concat(test)
test.to_csv('SteelPlateFaults-test.csv') #saving test data

#Forming target set(Y)
ctr=[ctr0,ctr1]
ctr=pd.concat(ctr)

cte=[cte0,cte1]
cte=pd.concat(cte) #correct Y

#K=1
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(train,ctr)
knn_1=knn.predict(test)

#K=3
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(train,ctr)
knn_2=knn.predict(test)

#K=5
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(train,ctr)
knn_3=knn.predict(test)

#Q1.a CONFUSION MATRIX
conf_mat_1=confusion_matrix(cte,knn_1) #confusion matrix for K=1
conf_mat_2=confusion_matrix(cte,knn_2) #confusion matrix for K=3
conf_mat_3=confusion_matrix(cte,knn_3) #confusion matrix for K=5

print('\n Q-1 (i)')
print('\n Confusion Matrix for K=1 is \n' + str(conf_mat_1))
print('\n Confusion Matrix for K=3 is \n' + str(conf_mat_2))
print('\n Confusion Matrix for K=5 is \n' + str(conf_mat_3))
print('\n -----------------------------------------------------')

#Q1.b CLASSIFICATION ACCURACY
acc_1=accuracy_score(cte,knn_1)
acc_2=accuracy_score(cte,knn_2)
acc_3=accuracy_score(cte,knn_3)
KNN_score = max(acc_1,acc_2,acc_3)
print('\n Q-1 (ii)')
print('\n Maximum accuracy is: '+str(KNN_score))
print('\n -----------------------------------------------------')

# Q-2
#Normalising Training Set
train_copy=train.copy()
test_copy=test.copy()

for i in train.columns:
    rng_train=(max(train[i])-min(train[i]))
    min_train=min(train[i])
    train[i]=(train[i]-min_train)/rng_train
      
train.to_csv('SteelPlateFaults-train-normalised.csv') #Saving normalised training set

#Normalising Test Set
for i in test_copy.columns:
    rng_test=(max(train_copy[i])-min(train_copy[i]))
    min_test=min(train_copy[i])
    test[i]=(test[i]-min_test)/rng_test
    
test.to_csv('SteelPlateFaults-test-normalised.csv') #Saving normalised test set

#K=1
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(train,ctr)
knn_new_1=knn.predict(test)

#K=3
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(train,ctr)
knn_new_2=knn.predict(test)

#K=5
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(train,ctr)
knn_new_3=knn.predict(test)

#Q2.a CONFUSION MATRIX
conf_mat_new_1=confusion_matrix(cte,knn_new_1) #confusion matrix for K=1
conf_mat_new_2=confusion_matrix(cte,knn_new_2) #confusion matrix for K=3
conf_mat_new_3=confusion_matrix(cte,knn_new_3) #confusion matrix for K=5

print('\n Q-2 (i)')
print('\n Confusion Matrix for K=1 is \n' + str(conf_mat_new_1))
print('\n Confusion Matrix for K=3 is \n' + str(conf_mat_new_2))
print('\n Confusion Matrix for K=5 is \n' + str(conf_mat_new_3))
print('\n -----------------------------------------------------')

#Q2.b CLASSIFICATION ACCURACY
acc_new_1=accuracy_score(cte,knn_new_1)
acc_new_2=accuracy_score(cte,knn_new_2)
acc_new_3=accuracy_score(cte,knn_new_3)
KNN_score_normalised = max(acc_new_1,acc_new_2,acc_new_3)
print('\n Q-2 (ii)')
print('\n Maximum accuracy is: '+str(KNN_score_normalised))
print('\n -----------------------------------------------------')

#Q3 
#dropping TypeOfSteel_A300 TypeOfSteel_A400 from train_0 as they are making covariance matrix non-invertible

print('\n Q-3')
train_0=train_0.drop(['TypeOfSteel_A300','TypeOfSteel_A400'],axis=1)
train_1=train_1.drop(['TypeOfSteel_A300','TypeOfSteel_A400'],axis=1)

#making mean vector for class 0
m_vec0=train_0.mean()

#making covariance matrix,determinant and inverse for class 0
cov_mat0=np.cov(train_0.T)
cov0i=np.linalg.inv(cov_mat0) #inverse
d0=np.linalg.det(cov_mat0) #determinant

#making mean vector for class 1
m_vec1=train_1.mean()

#making covariance matrix,determinant and inverse for class 1
cov_mat1=np.cov(train_1.T)
cov1i=np.linalg.inv(cov_mat1) #inverse
d1=np.linalg.det(cov_mat1) #determinant

#Just for info using gaussian naive bayes algorithm
GNB=GaussianNB()
GNB.fit(train,ctr)
y=GNB.predict(test)
print('\n Accuracy score using Naive: '+str(accuracy_score(cte,y)))
print('\n Confusion Matrix using Naive: '+'\n'+str(confusion_matrix(cte,y)))

test=test.drop(['TypeOfSteel_A300','TypeOfSteel_A400'],axis=1)
z=[]
for i in test.itertuples():
    j=np.array(i)
    j=np.delete(j,0)
    likely0=(1/((2*(np.pi))**(len(j)/2)))*((1/d0)**0.5)*np.exp((-1*1/2)*np.dot(np.dot(((j-m_vec0).T),cov0i),(j-m_vec0)))
    likely1=(1/((2*(np.pi))**(len(j)/2)))*((1/d1)**0.5)*np.exp((-1*1/2)*np.dot(np.dot(((j-m_vec1).T),cov1i),(j-m_vec1)))
    if likely0>=likely1:
        z.append(0)
        
    elif likely1>likely0:
        z.append(1)
    
Bayes_score=accuracy_score(cte,z)
print('\n Accuracy score using Bayes: '+str(Bayes_score))
print('\n Confusion Matrix using Bayes: '+'\n'+str(confusion_matrix(cte,z)))
index_cte = cte.index[cte == 1]
print('\n -----------------------------------------------------')

#Q4
#comparison of different classifiers
print('\n Q-4')
scores = {'Accuracy of KNN classifier':KNN_score, 'Accuracy of Normalised KNN classifier': KNN_score_normalised, 'Accuracy using Bayes': Bayes_score}
print('\n' + str(scores))