# Importing modules required for the program
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
from numpy.linalg import eig

# Reading the CSV file using pandas
data = pd.read_csv('pima-indians-diabetes.csv')

#Q-1
test_dict={'Pregs':data["pregs"],'Plas':data["plas"],'Pres':data["pres"],'Skin':data["skin"],'Test':data["test"],'BMI':data["BMI"],'Pedi':data["pedi"],'Age':data["Age"]}

Q1=[]
Q3=[]
IQR=[]

for i in test_dict:      
    Q1.append(np.percentile(test_dict[i],25))
    Q3.append(np.percentile(test_dict[i],75))
    IQR.append(np.percentile(test_dict[i],75)-np.percentile(test_dict[i],25))

Q1_dict={'Pregs':Q1[0],'Plas':Q1[1],'Pres':Q1[2],'Skin':Q1[3],'Test':Q1[4],'BMI':Q1[5],'Pedi':Q1[6],'Age':Q1[7]}  
Q3_dict={'Pregs':Q3[0],'Plas':Q3[1],'Pres':Q3[2],'Skin':Q3[3],'Test':Q3[4],'BMI':Q3[5],'Pedi':Q3[6],'Age':Q3[7]}     
IQR_dict={'Pregs':IQR[0],'Plas':IQR[1],'Pres':IQR[2],'Skin':IQR[3],'Test':IQR[4],'BMI':IQR[5],'Pedi':IQR[6],'Age':IQR[7]}  

dict_arrays = {'Pregs':np.array((test_dict['Pregs'])),'Plas':np.array((test_dict['Plas'])),'Pres':np.array((test_dict['Pres'])),'Skin':np.array((test_dict['Skin'])),'Test':np.array((test_dict['Test'])),'BMI':np.array((test_dict['BMI'])),'Pedi':np.array((test_dict['Pedi'])),'Age':np.array((test_dict['Age']))}

out=[]
for i in dict_arrays:
    out_i=[]
    for j in dict_arrays[i]:
        if((Q1_dict[i] - (1.5 * IQR_dict[i]) < j < Q3_dict[i] + (1.5 * IQR_dict[i]))):
            continue
        else:
           out_i.append(j)
    out.append(out_i) 
        
med_dict = {'Pregs':np.median(dict_arrays['Pregs']),'Plas':np.median(dict_arrays['Plas']),'Pres':np.median(dict_arrays['Pres']),'Skin':np.median(dict_arrays['Skin']),'Test':np.median(dict_arrays['Test']),'BMI':np.median(dict_arrays['BMI']),'Pedi':np.median(dict_arrays['Pedi']),'Age':np.median(dict_arrays['Age'])}

dict_arrays_new = {'Pregs':np.array((test_dict['Pregs'])),'Plas':np.array((test_dict['Plas'])),'Pres':np.array((test_dict['Pres'])),'Skin':np.array((test_dict['Skin'])),'Test':np.array((test_dict['Test'])),'BMI':np.array((test_dict['BMI'])),'Pedi':np.array((test_dict['Pedi'])),'Age':np.array((test_dict['Age']))}

for i in dict_arrays_new:
    index = 0 
    for j in dict_arrays_new[i]:
        if((Q1_dict[i] - (1.5 * IQR_dict[i]) < j < Q3_dict[i] + (1.5 * IQR_dict[i]))):
            index+=1
            continue
        else:
            dict_arrays_new[i][index] = med_dict[i]
            index+=1

bn_min = []
bn_max = []
for i in dict_arrays_new:
    m = min(dict_arrays_new[i])
    M = max(dict_arrays_new[i])
    bn_min.append(m)
    bn_max.append(M)

min_dict={'Pregs':bn_min[0],'Plas':bn_min[1],'Pres':bn_min[2],'Skin':bn_min[3],'Test':bn_min[4],'BMI':bn_min[5],'Pedi':bn_min[6],'Age':bn_min[7]}
max_dict={'Pregs':bn_max[0],'Plas':bn_max[1],'Pres':bn_max[2],'Skin':bn_max[3],'Test':bn_max[4],'BMI':bn_max[5],'Pedi':bn_max[6],'Age':bn_max[7]}

print(' ------ Min. before Normalisation: ------')
print(min_dict)
print(' \n ------ Max. before Normalisation: ------ \n')
print(max_dict)

x = []
for i in dict_arrays_new:
    y = []
    for j in dict_arrays_new[i]:
        norm = 5 + (((j - min_dict[i]) / (max_dict[i] - min_dict[i]))*7)
        y.append(norm)
    x.append(y) 
    
norm_dict = {'Pregs':x[0],'Plas':x[1],'Pres':x[2],'Skin':x[3],'Test':x[4],'BMI':x[5],'Pedi':x[6],'Age':x[7]}

an_min = []
an_max = []
for i in norm_dict:
    m = min(norm_dict[i])
    M = max(norm_dict[i])
    an_min.append(m)
    an_max.append(M)

min_dict_an = {'Pregs':an_min[0],'Plas':an_min[1],'Pres':an_min[2],'Skin':an_min[3],'Test':an_min[4],'BMI':an_min[5],'Pedi':an_min[6],'Age':an_min[7]}
max_dict_an = {'Pregs':an_max[0],'Plas':an_max[1],'Pres':an_max[2],'Skin':an_max[3],'Test':an_max[4],'BMI':an_max[5],'Pedi':an_max[6],'Age':an_max[7]}

print('------ Min. after Normalisation: ------')
print(min_dict_an)
print(' \n ------ Max. after Normalisation: ------ \n')
print(max_dict_an)
   
mean_list = []
for i in dict_arrays_new:
    mean = stat.mean(dict_arrays_new[i])
    mean_list.append(mean) 

std_list = []
for i in dict_arrays_new:
    std = stat.stdev(dict_arrays_new[i])
    std_list.append(std)
    
mean_dict = {'Pregs':mean_list[0],'Plas':mean_list[1],'Pres':mean_list[2],'Skin':mean_list[3],'Test':mean_list[4],'BMI':mean_list[5],'Pedi':mean_list[6],'Age':mean_list[7]}
std_dict = {'Pregs':std_list[0],'Plas':std_list[1],'Pres':std_list[2],'Skin':std_list[3],'Test':std_list[4],'BMI':std_list[5],'Pedi':std_list[6],'Age':std_list[7]}

print(' \n ------ Mean of each attribute: ------')
print(mean_dict)
print('\n ------ Std Dev of each attribute: ------ \n')
print(std_dict)

a = []
for i in dict_arrays_new:
    b = []
    for j in dict_arrays_new[i]:      
       k = (j - mean_dict[i])/std_dict[i]
       b.append(k)
    a.append(b)


stand_dict= {'Pregs':a[0],'Plas':a[1],'Pres':a[2],'Skin':a[3],'Test':a[4],'BMI':a[5],'Pedi':a[6],'Age':a[7]}

mean_stand = []
for i in stand_dict:
    mean1 = stat.mean(stand_dict[i])
    mean_stand.append(mean1) 

std_stand = []
for i in stand_dict:
    std1 = stat.stdev(stand_dict[i])
    std_stand.append(std1)
    

mean_stand_dict = {'Pregs':mean_stand[0],'Plas':mean_stand[1],'Pres':mean_stand[2],'Skin':mean_stand[3],'Test':mean_stand[4],'BMI':mean_stand[5],'Pedi':mean_stand[6],'Age':mean_stand[7]}
std_stand_dict = {'Pregs':std_stand[0],'Plas':std_stand[1],'Pres':std_stand[2],'Skin':std_stand[3],'Test':std_stand[4],'BMI':std_stand[5],'Pedi':std_stand[6],'Age':std_stand[7]}

print(' \n ------ Mean of each attribute (after standardisation): ------')
print(mean_stand_dict)
print('\n ------ Std Dev of each attribute (after standardisation): ------ \n')
print(std_stand_dict)


#Q2

mean = [0, 0]
cov = [[13, -3], [-3, 5]]

X1, X2 = np.random.multivariate_normal(mean, cov, 1000).T

#a
plt.scatter(X1, X2)
plt.axis('equal')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Plot of 2-D data')
plt.show()

#b
eig_vals,eig_vecs=eig(cov)

print("\n----Eigen Values---\n")
print(eig_vals)
print("\n----Eigen Vectors---\n")
print(eig_vecs)

origin = [0, 0]

eig_vec1 = eig_vecs[:,0]
eig_vec2 = eig_vecs[:,1]

plt.scatter(X1, X2)
plt.quiver(*origin, *eig_vec1, color=['r'], scale=5)
plt.quiver(*origin, *eig_vec2, color=['b'], scale=5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Plot of 2-D data along with eigen direction')
plt.show()

#c
#i
X_1 = np.resize(X1, (1000,1))
X_2 = np.resize(X2, (1000,1))
eig_vec1_1 = np.resize(eig_vec1, (1,2))

X1_1 = X_1.dot(eig_vec1_1)
X2_1 = X_2.dot(eig_vec1_1)

plt.scatter(X1_1, X2_1)
plt.quiver(0,0, color=['r'], scale=5)
plt.quiver(*origin, *eig_vec2, color=['b'], scale=5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Plot of 2-D data projected along vector 1')
plt.show()

#ii
X_1 = np.resize(X1, (1000,1))
X_2 = np.resize(X2, (1000,1))
eig_vec2_1 = np.resize(eig_vec2, (1,2))

X1_2 = X_1.dot(eig_vec2_1)
X2_2 = X_2.dot(eig_vec2_1)

plt.scatter(X1_2, X2_2)
plt.quiver(*origin, *eig_vec1, color=['r'], scale=5)
plt.quiver(*origin, *eig_vec2, color=['b'], scale=5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Plot of 2-D data projected along vector 2')
plt.show()

#d
X = np.column_stack((X1,X2))

eig_vec1 = eig_vecs[:,0]
eig_vec2 = eig_vecs[:,1]

eig_v = np.column_stack((eig_vec1,eig_vec2))

A_n = X.dot(eig_v)
X_new = A_n.dot(np.resize(eig_vecs.T,(2,2)))

Recon_error = ( (X - X_new)**2 ) **0.5
print("\n----Reconstruction Error---\n")
print((Recon_error.mean()))

# 3

df=pd.read_csv("pima-indians-diabetes.csv")

Q3 = np.quantile(df['pregs'], 0.75)
Q1 = np.quantile(df['pregs'], 0.25)
IQR = Q3 - Q1

Outlier1 = Q1 - (1.5*IQR)
Outlier2 = Q3 + (1.5*IQR)

median = df['pregs'].median()
df['pregs'] = np.where(df['pregs'] < Outlier1, median,df['pregs'])
df['pregs'] = np.where(df['pregs'] > Outlier2, median,df['pregs'])

Q3 = np.quantile(df['plas'], 0.75)
Q1 = np.quantile(df['plas'], 0.25)
IQR = Q3 - Q1

Outlier1 = Q1 - (1.5*IQR)
Outlier2 = Q3 + (1.5*IQR)

median = df['plas'].median()
df['plas'] = np.where(df['plas'] < Outlier1, median,df['plas'])
df['plas'] = np.where(df['plas'] > Outlier2, median,df['plas'])

Q3 = np.quantile(df['pres'], 0.75)
Q1 = np.quantile(df['pres'], 0.25)
IQR = Q3 - Q1

Outlier1 = Q1 - (1.5*IQR)
Outlier2 = Q3 + (1.5*IQR)

median = df['pres'].median()
df['pres'] = np.where(df['pres'] < Outlier1, median,df['pres'])
df['pres'] = np.where(df['pres'] > Outlier2, median,df['pres'])

Q3 = np.quantile(df['skin'], 0.75)
Q1 = np.quantile(df['skin'], 0.25)
IQR = Q3 - Q1

Outlier1 = Q1 - (1.5*IQR)
Outlier2 = Q3 + (1.5*IQR)

median = df['skin'].median()
df['skin'] = np.where(df['skin'] < Outlier1, median,df['skin'])
df['skin'] = np.where(df['skin'] > Outlier2, median,df['skin'])

Q3 = np.quantile(df['test'], 0.75)
Q1 = np.quantile(df['test'], 0.25)
IQR = Q3 - Q1

Outlier1 = Q1 - (1.5*IQR)
Outlier2 = Q3 + (1.5*IQR)

median = df['test'].median()
df['test'] = np.where(df['test'] < Outlier1, median,df['test'])
df['test'] = np.where(df['test'] > Outlier2, median,df['test'])

Q3 = np.quantile(df['BMI'], 0.75)
Q1 = np.quantile(df['BMI'], 0.25)
IQR = Q3 - Q1

Outlier1 = Q1 - (1.5*IQR)
Outlier2 = Q3 + (1.5*IQR)

median = df['BMI'].median()
df['BMI'] = np.where(df['BMI'] < Outlier1, median,df['BMI'])
df['BMI'] = np.where(df['BMI'] > Outlier2, median,df['BMI'])

Q3 = np.quantile(df['pedi'], 0.75)
Q1 = np.quantile(df['pedi'], 0.25)
IQR = Q3 - Q1

Outlier1 = Q1 - (1.5*IQR)
Outlier2 = Q3 + (1.5*IQR)

median = df['pedi'].median()
df['pedi'] = np.where(df['pedi'] < Outlier1, median,df['pedi'])
df['pedi'] = np.where(df['pedi'] > Outlier2, median,df['pedi'])

Q3 = np.quantile(df['Age'], 0.75)
Q1 = np.quantile(df['Age'], 0.25)
IQR = Q3 - Q1

Outlier1 = Q1 - (1.5*IQR)
Outlier2 = Q3 + (1.5*IQR)

median = df['Age'].median()
df['Age'] = np.where(df['Age'] < Outlier1, median,df['Age'])
df['Age'] = np.where(df['Age'] > Outlier2, median,df['Age'])

data1 = pd.DataFrame(df,columns=['pregs','plas','pres','skin','test','BMI','pedi','Age'])

df_min = data1.min()
df_max = data1.max()
norm_df = 5 + (((data1 - df_min) / (df_max - df_min)) * 7)
norm_df_min = norm_df.min()
norm_df_max = norm_df.max()
stand_df = ((data1 - data1.mean()) / (data1.std()))

# a
cov_m = stand_df.cov()
print("\n----Original Covariance Matrix---\n")
print(round(cov_m,3))

eig_vals,eig_vecs=np.linalg.eig(cov_m)

print("\n----Eigen Values---\n")
print(eig_vals)

eigs_des = np.sort(eig_vals)[::-1]

eig_vec1 = eig_vecs[:,0]
eig_vec2 = eig_vecs[:,1]

eig_v = np.column_stack((eig_vec1,eig_vec2))

red_df = stand_df.dot(eig_v)

print("\n----Variance---\n")
print(red_df.var())

plt.scatter(red_df[0], red_df[1])
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

#b
plt.plot(eigs_des)
plt.xlabel('Eigen Value (index)')
plt.ylabel('Magnitude')
plt.title('Line plot for eigen values')

#c
idx = eig_vals.argsort()[::-1]   
eig_val = eig_vals[idx]
eig_vec = eig_vecs[:,idx]

print(eig_val,eig_vec)

eig_vec1 = eig_vecs[:,0]

eig_v = np.resize(eig_vec1, (8,1))
red_df = stand_df.dot(eig_v)

n_df = red_df.dot(np.resize(eig_v, (1,8)))

print("\n----Covariance Matrix---\n")
print(round(red_df.cov(),3))

#l = 2
eig_vec1 = eig_vecs[:,0]
eig_vec2 = eig_vecs[:,1]

eig_v = np.column_stack((eig_vec1,eig_vec2))
red_df = stand_df.dot(eig_v)

n_df = red_df.dot(np.resize(eig_v, (2,8)))

print("\n----Covariance Matrix---\n")
print(round(red_df.cov(),3))

#l = 3
eig_vec1 = eig_vecs[:,0]
eig_vec2 = eig_vecs[:,1]
eig_vec3 = eig_vecs[:,2]

eig_v = np.column_stack((eig_vec1,eig_vec2,eig_vec3))
red_df = stand_df.dot(eig_v)

n_df = red_df.dot(np.resize(eig_v, (3,8)))

print("\n----Covariance Matrix---\n")
print(round(red_df.cov(),3))

#l = 4
eig_vec1 = eig_vecs[:,0]
eig_vec2 = eig_vecs[:,1]
eig_vec3 = eig_vecs[:,2]
eig_vec4 = eig_vecs[:,3]

eig_v = np.column_stack((eig_vec1,eig_vec2,eig_vec3,eig_vec4))
red_df = stand_df.dot(eig_v)

n_df = red_df.dot(np.resize(eig_v, (4,8)))

print("\n----Covariance Matrix---\n")
print(round(red_df.cov(),3))

#l = 5
eig_vec1 = eig_vecs[:,0]
eig_vec2 = eig_vecs[:,1]
eig_vec3 = eig_vecs[:,2]
eig_vec4 = eig_vecs[:,3]
eig_vec5 = eig_vecs[:,4]

eig_v = np.column_stack((eig_vec1,eig_vec2,eig_vec3,eig_vec4,eig_vec5))
red_df = stand_df.dot(eig_v)

n_df = red_df.dot(np.resize(eig_v, (5,8)))

print("\n----Covariance Matrix---\n")
print(round(red_df.cov(),3))

#l = 6
eig_vec1 = eig_vecs[:,0]
eig_vec2 = eig_vecs[:,1]
eig_vec3 = eig_vecs[:,2]
eig_vec4 = eig_vecs[:,3]
eig_vec5 = eig_vecs[:,4]
eig_vec6 = eig_vecs[:,5]

eig_v = np.column_stack((eig_vec1,eig_vec2,eig_vec3,eig_vec4,eig_vec5,eig_vec6))
red_df = stand_df.dot(eig_v)

n_df = red_df.dot(np.resize(eig_v, (6,8)))

print("\n----Covariance Matrix---\n")
print(round(red_df.cov(),3))

#l = 7
eig_vec1 = eig_vecs[:,0]
eig_vec2 = eig_vecs[:,1]
eig_vec3 = eig_vecs[:,2]
eig_vec4 = eig_vecs[:,3]
eig_vec5 = eig_vecs[:,4]
eig_vec6 = eig_vecs[:,5]
eig_vec7 = eig_vecs[:,6]

eig_v = np.column_stack((eig_vec1,eig_vec2,eig_vec3,eig_vec4,eig_vec5,eig_vec6,eig_vec7))
red_df = stand_df.dot(eig_v)

n_df = red_df.dot(np.resize(eig_v, (7,8)))

print("\n----Covariance Matrix---\n")
print(round(red_df.cov(),3))

#l = 8
eig_vec1 = eig_vecs[:,0]
eig_vec2 = eig_vecs[:,1]
eig_vec3 = eig_vecs[:,2]
eig_vec4 = eig_vecs[:,3]
eig_vec5 = eig_vecs[:,4]
eig_vec6 = eig_vecs[:,5]
eig_vec7 = eig_vecs[:,6]
eig_vec8 = eig_vecs[:,7]

eig_v = np.column_stack((eig_vec1,eig_vec2,eig_vec3,eig_vec4,eig_vec5,eig_vec6,eig_vec7,eig_vec8))
red_df = stand_df.dot(eig_v)

n_df = red_df.dot(np.resize(eig_v, (8,8)))

print("\n----Covariance Matrix---\n")
print(round(red_df.cov(),3))