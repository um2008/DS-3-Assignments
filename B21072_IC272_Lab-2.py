# Importing modules required for the program
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the CSV file using pandas
data_miss = pd.read_csv('landslide_data3_miss.csv')
data_org =  pd.read_csv('landslide_data3_original.csv')
data_miss2 = pd.read_csv('landslide_data3_miss.csv')
data_miss3 = pd.read_csv('landslide_data3_miss.csv')
data_miss4 = pd.read_csv('landslide_data3_miss.csv')
data_miss5 = pd.read_csv('landslide_data3_miss.csv')

# Data frames defining
df_org = pd.DataFrame(data_org)
df_miss = pd.DataFrame(data_miss)
df_miss2 = pd.DataFrame(data_miss2)
df_miss3 = pd.DataFrame(data_miss3)
df_miss4 = pd.DataFrame(data_miss4)
df = pd.DataFrame(data_miss5)

cols = ['dates','stationid','temperature','humidity','pressure','rain','lightavgw/o0','lightmax','moisture']
dict_miss = {'Station Id' : data_miss["stationid"], 'Temperature' : data_miss["temperature"], 'Humidity' : data_miss["humidity"], 'Pressure': data_miss["pressure"], 'Rain': data_miss["rain"], 'Light Avg.' : data_miss["lightavgw/o0"], 'Light Max.' : data_miss["lightmax"], 'Moisture' : data_miss["moisture"]} 
dict_org = {'Station Id' : data_org["stationid"], 'Temperature' : data_org["temperature"], 'Humidity' : data_org["humidity"], 'Pressure': data_org["pressure"], 'Rain': data_org["rain"], 'Light Avg.' : data_org["lightavgw/o0"], 'Light Max.' : data_org["lightmax"], 'Moisture' : data_org["moisture"]}

# 1
print('Question-1')
print()

x = []
y = []
for i in dict_miss:
    null_i=(dict_miss[i]).isnull().sum()
    y.append(null_i)
    x.append(i)

plt.bar(x,y)
plt.xlabel("Attributes")
plt.ylabel("Missing Values")
plt.title("No. of Missing Values")
plt.xticks(rotation=90)
plt.show()
print()
print('_______________________________________________')
print()

# 2
print('Question-2')
print()

# a
print('Part-A')
org_len=len(data_miss)
data_miss.dropna(subset=["stationid"],inplace=True)
new_len1=len(data_miss)
print("Number of tuples deleted are: " + str((org_len)-(new_len1)))
print()

# b
print('Part-B')
data_drop=data_miss.dropna(thresh=7)
new_len2 = len(data_drop)
print("Number of tuples deleted which have equal to or more than one third of attributes with missing values are :",new_len1-new_len2)
print()
print('_______________________________________________')
print()


# 3
print('Question-3')
print()
print(data_drop.isnull().sum().sum())
print()
print('_______________________________________________')
print()


# 4
print('Question-4')
print()
# (i)(a)
print('Part-i (a)')
for i in cols[2:]:
    df_miss[i]=df_miss[i].fillna(df_miss[i].mean())
    
def parameters(df_para):
    return [df_para.mean(),df_para.median(),df_para.mode()[0],df_para.std()] 

para_miss =[]
para_org =[]
for i in cols[2:]:
    para_miss.append(parameters(df_miss[i]))
    para_org.append(parameters(df_org[i]))

print('For Filled Dataframe: ')
index=0
for i in cols[2:]:
    print(index+1,')')
    print('Mean for',i,'=',para_miss[index][0])
    print('Median for',i,'=',para_miss[index][1])
    print('Mode for',i,'=',para_miss[index][2])
    print('Standard Deviation for',i,'=',para_miss[index][3])
    print()
    index+=1
print()
print('-------------------------------------------------------')
print()

print('For Original Dataframe: ')
index=0
for i in cols[2:]:
    print(index+1,')')
    print('Mean for',i,'=',para_org[index][0])
    print('Median for',i,'=',para_org[index][1])
    print('Mode for',i,'=',para_org[index][2])
    print('Standard Deviation for',i,'=',para_org[index][3])
    print()
    index+=1
print()

# (i)(b)
print('Part-i (b)')

count_na={}
for attribute in cols[2:]:
    count_na[attribute] = (df_miss2[attribute].isna())

for attribute in cols[2:]:
    df_miss2[attribute]=df_miss2[attribute].fillna(df_miss2[attribute].mean())

def RMSE(df_miss2,df_org,attribute,count_na):
    count=0
    Rmse=0
    for index in range(len(count_na)):
        if(count_na[index] == True):
            Rmse += (df_org[index] - df_miss2[index])**2
            count+=1
    Rmse=(Rmse/count)**(1/2)
    return(Rmse)
    
print('\nRMSE is computed between the replaced value by respective mean and its corresponding original value -:\n')
RMSE_values=[]
for attribute in cols[2:]:
    RMSE_values.append(RMSE(df_miss2[attribute],df_org[attribute],attribute,count_na[attribute]))

index=0
for attribute in cols[2:]:
    print('RMSE for',attribute,'=',RMSE_values[index])
    index+=1
plt.scatter(cols[2:],RMSE_values)
plt.title('RMSE vs Attributes')
plt.xlabel('Attributes')
plt.ylabel('RMSE Values')
plt.xticks(rotation=90)
plt.grid()
plt.show()

# (ii)(a)
print('Part-ii (a)')

df_miss3=df_miss3.interpolate(method = 'linear', limit_direction = 'both')

def para(dfp):
    return [dfp.mean(),dfp.median(),dfp.mode()[0],dfp.std()] 

para_filled1 =[]
para_org1 =[]
for attribute in cols[2:]:
    para_filled1.append(para(df_miss3[attribute]))
    para_org1.append(para(df_org[attribute]))

print('For Filled Dataframe by Linear Interpolation-:\n')
index=0
for attribute in cols[2:]:
    print(index+1,')')
    print('Mean for',attribute,'=',para_filled1[index][0])
    print('Median for',attribute,'=',para_filled1[index][1])
    print('Mode for',attribute,'=',para_filled1[index][2])
    print('Standard Deviation for',attribute,'=',para_filled1[index][3],'\n')
    index+=1
print()
print('-------------------------------------------------------')
print()
   
print('\n\nFor Original Dataframe -:\n')
index=0
for attribute in cols[2:]:
    print(index+1,')')
    print('Mean for',attribute,'=',para_org1[index][0])
    print('Median for',attribute,'=',para_org1[index][1])
    print('Mode for',attribute,'=',para_org1[index][2])
    print('Standard Deviation for',attribute,'=',para_org1[index][3],'\n')
    index+=1
print()

# (ii)(b)
print('Part-ii (b)')

count_na1={}
for attribute in cols[2:]:
    count_na1[attribute] = (df_miss4[attribute].isna())
    
df_miss4=df_miss4.interpolate(method = 'linear', limit_direction = 'both')

def RMSE(df_miss4,df_org,attribute,count_na1):
    count1=0
    Rmse1=0
    for index in range(len(count_na1)):
        if(count_na1[index] == True):
            Rmse1 += (df_org[index] - df_miss4[index])**2
            count1+=1
    Rmse1=(Rmse1/count1)**(1/2)
    return(Rmse1)    

print('\nRMSE is computed between the replaced value by respective mean and its corresponding original value -:\n')
RMSE_values1=[]
for attribute in cols[2:]:
    RMSE_values1.append(RMSE(df_miss4[attribute],df_org[attribute],attribute,count_na1[attribute]))

index=0
for attribute in cols[2:]:
    print('RMSE for',attribute,'=',RMSE_values1[index])
    index+=1
plt.scatter(cols[2:],RMSE_values1)
plt.title('RMSE vs Attributes')
plt.xlabel('Attributes')
plt.ylabel('RMSE Values')
plt.xticks(rotation=90)
plt.grid()
plt.show()
print()
print('_______________________________________________')
print()


# 5
print('Question-5')
print()
# a 
print('Part-A')
col = ['temperature','rain']
df=df.interpolate(method = 'linear', limit_direction = 'both')

Q1=[]
Q3=[]

for attribute in col:
    Q1.append(np.percentile(df[attribute],25))
    Q3.append(np.percentile(df[attribute],75))

IQR = [Q3[0]-Q1[0] , Q3[1]-Q1[1]]

out_temp =[]
temperature = np.array(df[col[0]])
for temp_values in temperature:
    if((Q1[0] - (1.5 * IQR[0]) < temp_values < Q3[0] + (1.5 * IQR[0]))):
        continue
    else:
        out_temp.append(temp_values)

out_rain =[]
rain = np.array(df[col[1]])
for rain_values in rain:
    if((Q1[1] - (1.5 * IQR[1]) < rain_values < Q3[1] + (1.5 * IQR[1]))):
        continue
    else:
        out_rain.append(rain_values)
    
print('Outliers for temperature -:\n')
print(out_temp)
print('\nOutliers for rain -:\n')
print(out_rain)

plt.boxplot(temperature)
plt.xticks([1],['temperature'])
plt.title('For temperature after Linear Interpolation')
plt.grid()
plt.show()

plt.boxplot(rain)
plt.xticks([1],['rain'])
plt.title('For rain after Linear Interpolation')
plt.grid()
plt.show()
print()

# b
print('Part-B')
median = [ df[col[0]].median(), df[col[1]].median() ]
temperature1 = np.array(df[col[0]])
temp_median=temperature1.copy()

index=0
for temp_values in temperature1:
    if((Q1[0] - (1.5 * IQR[0]) < temp_values < Q3[0] + (1.5 * IQR[0]))):
        index+=1
        continue
    else:
        temp_median[index] = median[0]
    index+=1
    
rain1 = np.array(df[col[1]])
rain_median=rain1.copy()
index=0
for rain_values in rain1:
    if((Q1[1] - (1.5 * IQR[1]) < rain_values < Q3[1] + (1.5 * IQR[1]))):
        index+=1
        continue
    else:
        rain_median[index] = median[1]
    index+=1

plt.boxplot(temp_median)
plt.xticks([1],['temperature'])
plt.title('For temperature after replacing with median and Linear Interpolation')
plt.grid()
plt.show()

plt.boxplot(rain_median)
plt.xticks([1],['rain'])
plt.title('For rain after replacing with median and Linear Interpolation')
plt.grid()
plt.show()
    