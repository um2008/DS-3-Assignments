'''Unnat Maaheshwari (B@1072) '''  
#                       Lab - 7

# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#_________________________________________________________________________________________________

                                    #Code for 1
#Question 1
df = pd.read_csv("Iris.csv")
print("\nQ1)")
# Separating out the features

df1=df.copy()
del df1["Species"]
x=df1.values

# Separating out the target

y=df["Species"].values


pca=PCA(n_components=2)
principalComponents=pca.fit_transform(x)
principalDf=pd.DataFrame(data = principalComponents,columns=['principal component 1','principal component 2'])
finalDf=pd.concat([principalDf, df[['Species']]], axis=1)
print(finalDf)

#_________________________________________________________________________________________________

                                    #Code for 2   
#Question 2
print("\nQ2)")
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
import numpy as np
df=finalDf
kmeans = KMeans(n_clusters=3, random_state=0)
df['cluster'] = kmeans.fit_predict(df[['principal component 1', 'principal component 2']])
# get centroids
centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids] 
cen_y = [i[1] for i in centroids]
## add to df
df['cen_x'] = df.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
df['cen_y'] = df.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})
# define and map colors
colors = ['#DF2020', '#81DF20', '#2095DF']
df['c'] = df.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})
plt.scatter(df["principal component 1"], df["principal component 2"], c=df.c)
plt.scatter(df['cen_x'],df['cen_y'],color="black",marker="*")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
dis=0
for i in range(len(df)):
    dis_x=0
    dis_y=0
    dis_x=df["principal component 1"][i]-df["cen_x"][i]
    dis_y=df["principal component 2"][i]-df["cen_y"][i]
    dis+=(dis_x**2+dis_y**2)**(1/2)
print(dis)

from sklearn import metrics
from scipy.optimize import linear_sum_assignment
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
    #print(contingency_matrix)
    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    #return cluster accuracy
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)
df = df.assign(Species_numeral='NAN')
for i in range(len(df)):
    if(df["Species"][i]=="Iris-setosa"):
        df["Species_numeral"][i]=0
    if(df["Species"][i]=="Iris-versicolor"):
        df["Species_numeral"][i]=1
    if(df["Species"][i]=="Iris-virginica"):
        df["Species_numeral"][i]=2
print(purity_score(df["cluster"],df["Species_numeral"]))

#_________________________________________________________________________________________________

                                    #Code for 3
#Question 3
print("\nQ3)")
def distortion(k):
    kmeans = KMeans(n_clusters=k, random_state=0)
    df['cluster'] = kmeans.fit_predict(df[['principal component 1', 'principal component 2']])
    centroids = kmeans.cluster_centers_
    cen_x = [i[0] for i in centroids] 
    cen_y = [i[1] for i in centroids]
    df['cen_x'] = df.cluster.map({i:cen_x[i] for i in range(k)})
    df['cen_y'] = df.cluster.map({i:cen_y[i] for i in range(k)})
    dis=0
    for i in range(len(df)):
        dis_x=0
        dis_y=0
        dis_x=df["principal component 1"][i]-df["cen_x"][i]
        dis_y=df["principal component 2"][i]-df["cen_y"][i]
        dis+=(dis_x**2+dis_y**2)**(1/2)
    return (dis)
l=[2,3,4,5,6,7]
distortion_values=[]
for i in l:
    distortion_values.append(distortion(i))
    print("The distortion measure for number of clusters",i,"is",distortion(i))
plt.plot(l,distortion_values,marker="*",color="red")
plt.xlabel("Number of clusters")
plt.ylabel("Distortion measure")
plt.show()
# The curve looks like an elbow. 
# In the above plot, the elbow is at k=3. 
# Sum of squared distances falls suddenly indicating the optimal k for this dataset is 3.
for i in l:
    distortion(i)
    print("The purity score for K value as",i,"is",purity_score(df["cluster"],df["Species_numeral"]))
del df["cluster"]
del df["cen_x"]
del df["cen_y"]
del df["c"]
del df["Species_numeral"]
print(df)

#_________________________________________________________________________________________________

                                    #Code for 4
#Question 4
print("\nQ4)")
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components = 3)
gmm.fit(df[['principal component 1', 'principal component 2']])
df["GMM_cluster"] = gmm.predict(df[['principal component 1', 'principal component 2']])
# print(df["GMM_cluster"])
centroids = gmm.means_
cen_x = [i[0] for i in centroids] 
cen_y = [i[1] for i in centroids]
## add to df
df['cen_x'] = df.GMM_cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
df['cen_y'] = df.GMM_cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})
colors = ['#DF2020', '#81DF20', '#2095DF']
df['c'] = df.GMM_cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})
plt.scatter(df["principal component 1"], df["principal component 2"], c=df.c)
plt.scatter(df['cen_x'],df['cen_y'],color="black",marker="*")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
df = df.assign(Species_numeral='NAN')
for i in range(len(df)):
    if(df["Species"][i]=="Iris-setosa"):
        df["Species_numeral"][i]=df["GMM_cluster"][0:50].mode().values[0]
    if(df["Species"][i]=="Iris-versicolor"):
        df["Species_numeral"][i]=df["GMM_cluster"][50:100].mode().values[0]
    if(df["Species"][i]=="Iris-virginica"):
        df["Species_numeral"][i]=df["GMM_cluster"][100:].mode().values[0]
k=gmm.score(df[['principal component 1', 'principal component 2']]).sum()
print("The value of  total data log likelihood at the last iteration of the GMM as distortion measure is",k)
print("The Purity score is",purity_score(df["GMM_cluster"],df["Species_numeral"]))

#_________________________________________________________________________________________________

                                    #Code for 5
#Question 5 
print("\nQ5)")
def GMMdistortion(k):
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components = k)
    gm.fit(df[['principal component 1', 'principal component 2']])
    df["GMM_cluster"] = gm.predict(df[['principal component 1', 'principal component 2']])
    for i in range(len(df)):
        if(k>2):
            if(df["Species"][i]=="Iris-setosa"):
                df["Species_numeral"][i]=df["GMM_cluster"][0:50].mode().values[0]
            if(df["Species"][i]=="Iris-versicolor"):
                df["Species_numeral"][i]=df["GMM_cluster"][50:100].mode().values[0]
            if(df["Species"][i]=="Iris-virginica"):
                df["Species_numeral"][i]=df["GMM_cluster"][100:].mode().values[0]
        k=gm.score(df[['principal component 1', 'principal component 2']]).sum()
    return k
l=[2,3,4,5,6,7]
distortion_values=[]
for i in l:
    distortion_values.append(-GMMdistortion(i))
    print("The distortion measure for number of clusters",i,"is",GMMdistortion(i))
plt.plot(l,distortion_values,marker="*",color="red")
plt.xlabel("Number of clusters")
plt.ylabel("Modulus Distortion measure")
plt.show()
for i in l:
    GMMdistortion(i)
    print("The purity score for K value as",i,"is",purity_score(df["GMM_cluster"],df["Species_numeral"]))
del df["GMM_cluster"]
del df["cen_x"]
del df["cen_y"]
del df["c"]
del df["Species_numeral"]
print(df)

#_________________________________________________________________________________________________

                                    #Code for 6
#Question 6
print("\nQ6)")
from sklearn.cluster import DBSCAN
df = df.assign(Species_numeral='NAN')
for i in range(len(df)):
    if(df["Species"][i]=="Iris-setosa"):
        df["Species_numeral"][i]=0
    if(df["Species"][i]=="Iris-versicolor"):
        df["Species_numeral"][i]=1
    if(df["Species"][i]=="Iris-virginica"):
        df["Species_numeral"][i]=2
epr_meansample=[[1,4],[1,10],[5,4],[5,10]]
for i in range(len(epr_meansample)):
    dbscan_model=DBSCAN(eps=epr_meansample[i][0], min_samples=epr_meansample[i][1]).fit(df[['principal component 1', 'principal component 2']])
    df["DBSCAN_predictions"] = dbscan_model.labels_
    k=df["DBSCAN_predictions"].unique()
    colors = ['#DF2020', '#81DF20', '#2095DF']
    df['c'] = df.DBSCAN_predictions.map({i:colors[i] for i in range(len(k))})
    plt.scatter(df["principal component 1"], df["principal component 2"], c=df.c)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()
    print("The purity score for K value as",i,"is",purity_score(df["DBSCAN_predictions"],df["Species_numeral"]))