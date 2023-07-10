# Importing modules required for the program
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat

# Reading the CSV file using pandas
data = pd.read_csv('pima-indians-diabetes.csv')

a=data["pregs"]
b=data["plas"]
c=data["pres"]
d=data["skin"]
e=data["test"]
f=data["BMI"]
g=data["pedi"]
h=data["Age"]

test_dict={'Pregs': a,'Plas':b,'Pres':c,'Skin':d,'Test':e,'BMI':f,'Pedi':g,'Age':h}
dict_a={'Pregs': a,'Plas':b,'Pres':c,'Skin':d,'Test':e,'BMI':f,'Pedi':g}
dict_b={'Pregs': a, 'Plas':b,'Pres':c,'Skin':d,'Test':e,'Pedi':g, 'Age':h}

#_________________________________________________________________________________
# 1st

for i in test_dict:
    mean_i=stat.mean(test_dict[i])
    median_i=stat.median(test_dict[i])
    mode_i=stat.mode(test_dict[i])
    max_i=max(test_dict[i])
    min_i=min(test_dict[i])
    std_i=stat.stdev(test_dict[i])
    print('Data for ' + i + ': ' )
    print('Mean: ' + str(mean_i))
    print('Median: ' + str(median_i))
    print('Mode: ' + str(mode_i))
    print('Maximum: ' + str(max_i))
    print('Minimum: ' + str(min_i))
    print('Standard Deviation: ' + str(std_i))
    print()

#_________________________________________________________________________________
# 2nd 

for i in dict_a:     
    plt.scatter(h,dict_a[i])
    plt.title("Age Vs " + i)
    plt.show()
    print()
    print()
    
print()
print('-----------------------------------------------------')
print()


for i in dict_b:    
    plt.scatter(f,dict_b[i],color='green')
    plt.title("BMI Vs " + i)
    plt.show()
    print()
    print()

#_________________________________________________________________________________
# 3rd

for i in dict_a:   
    print('Correlation of Age with ' + i + ': ')    
    corr_ai= np.corrcoef(h,dict_a[i])
    print(corr_ai[0,1])
    print()

print()
print('-----------------------------------------------------')
print()

for i in dict_b:   
    print('Correlation of BMI with ' + i + ': ')    
    corr_bi= np.corrcoef(f,dict_b[i])
    print(corr_bi[0,1])
    print()

#_________________________________________________________________________________
# 4th

plt.hist(a, bins = 10, color='orange')
plt.title('Pregs')
plt.show()
                      
plt.hist(d, bins = 10, color='red')
plt.title('Skin')
plt.show()

#_________________________________________________________________________________
# 5th

grp_by = data.groupby('class')
grp_1 = grp_by.get_group(0)
grp_2 = grp_by.get_group(1)

a1 = grp_1["pregs"]
plt.hist(a1, bins = 10, color='violet')
plt.title('Preg for Class = 0')
plt.show()

a2 = grp_2["pregs"]
plt.hist(a2, bins = 10, color='brown')
plt.title('Preg for Class = 1')
plt.show()

#_________________________________________________________________________________
# 6th

for i in test_dict:
    plt.boxplot(test_dict[i]);
    plt.title('Boxplot of ' + i )
    plt.show()






