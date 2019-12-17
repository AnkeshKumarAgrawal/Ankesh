# Ankesh
## -*- coding: utf-8 -*-
#"""
#Created on Thu Nov 28 11:18:43 2019
#
#@author: Ank
#"""
#
## -*- coding: utf-8 -*-
#"""
#Created on Sun Nov 24 10:02:48 2019
#
#@author: Ank
#"""
#
# To work with data frames
import pandas as pd
#
# To perform numerical operations
import numpy as np

# To partition the data
from sklearn.model_selection import train_test_split

# Importing lirary for logistic regression
from sklearn.linear_model import LogisticRegression

# Importing library for Linear Regression
from sklearn.linear_model import LinearRegression

# Importing scaler
from sklearn.preprocessing import StandardScaler


# Redaing data
data_income=pd.read_csv('Input Train v2.csv')

# Creating copy of the original data
data=data_income.copy()

# To get column labels from data
columns_list=list(data.columns)
print(columns_list) 

# To see unique elements of column
print(np.unique(data['pat_iden']))
print(np.unique(data['time_from_anchor']))
print(np.unique(data['Event_name']))
print(np.unique(data['lab_result_numeric']))


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
## Exploratory data analysis
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#1. Getting to know the data
#2. Data preprocessing (Missing values)
#3. Cross tables and data visualization

# Checking variables data types
print(data.info())

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
## Data pre-processing 
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# To check any missing values
print(data.isnull().sum())

# There are 1733 missing values for Event_desc variable
# As Event_name and Event_desc are similar variable thus remove Event_desc variable that has missing values

# Removing all the missing values
data1=data.copy()
data1.dropna(axis=1,inplace=True)

# Removing Duplicate records
data1.drop_duplicates(keep='first',inplace=True)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
## Working with data1 values
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Seperating diseased and healthy person
data1_disease=data1[data1['y_flag']==1]
data1_healthy=data1[data1['y_flag']==0]

# Diseased and healthy data without y_flag
data1_disease.drop(['y_flag'],axis=1,inplace=True)
data1_healthy.drop(['y_flag'],axis=1,inplace=True)

# Total number of patient
patient=list(np.unique(data1['pat_iden']))
print(patient)

# Finding number of unique diseased patient 
diseased_patient=list(np.unique(data1_disease['pat_iden']))
print(diseased_patient)

# Finding number of unique healthy patient 
healthy_patient=list(np.unique(data1_healthy['pat_iden']))
print(healthy_patient)

# Number of unique test for patient
test=list(np.unique(data1['Event_name']))
#
## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
## Working with diseased patient
## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
## Empty cell
final_diseased=np.empty((len(diseased_patient),2*len(test)))

## Fitting Linear Regression curve for Diseased patient
# Some test are being performed at multiple time instant
# For each patient linear regression model is predicted
# with time as independent and test value as dependent variable
# to find the mean and slope

for j in range(len(diseased_patient)):
    pat_diseased=data1_disease[data1_disease['pat_iden']==diseased_patient[j]]
    test_pat_diseased=list(pat_diseased['Event_name'])
    data_new_diseased=np.empty((len(pat_diseased),2))
    for k in range(len(test)):
        for i in range(len(test_pat_diseased)):
            if test_pat_diseased[i]==test[k]:
                data_new_diseased[i,0],data_new_diseased[i,1]=pat_diseased.iloc[i,3],pat_diseased.iloc[i,1]
            else:
                data_new_diseased[i,0],data_new_diseased[i,1]=np.nan,np.nan
        data_new_diseased1=pd.DataFrame(data_new_diseased) 
        data_new_diseased2=data_new_diseased1.dropna(axis=0)
        if  data_new_diseased2.size==0:
            final_diseased[j,(2*k)],final_diseased[j,(2*k+1)]=np.nan,np.nan
        else:
            x=np.array(data_new_diseased2.iloc[:,1])
            x1=x.reshape(-1,1)
            y=np.array(data_new_diseased2.iloc[:,0])
            y1=y.reshape(-1,1)
            reg=LinearRegression().fit(x1,y1)
            b1=reg.coef_
            b0=reg.intercept_
            final_diseased[j,(2*k)],final_diseased[j,(2*k+1)]=b0,b1
 

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Working with healthy patient
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Empty cell
final_healthy=np.empty((len(healthy_patient),2*len(test)))

# Fitting Linear Regression curve for Healthy patient

for j in range(len(healthy_patient)):
    pat_healthy=data1_healthy[data1_healthy['pat_iden']==healthy_patient[j]]
    test_pat_healthy=list(pat_healthy['Event_name'])
    data_new_healthy=np.empty((len(pat_healthy),2))
    for k in range(len(test)):
        for i in range(len(test_pat_healthy)):
            if test_pat_healthy[i]==test[k]:
                data_new_healthy[i,0],data_new_healthy[i,1]=pat_healthy.iloc[i,3],pat_healthy.iloc[i,1]
            else:
                data_new_healthy[i,0],data_new_healthy[i,1]=np.nan,np.nan
        data_new_healthy1=pd.DataFrame(data_new_healthy) 
        data_new_healthy2=data_new_healthy1.dropna(axis=0)
        if  data_new_healthy2.size==0:
            final_healthy[j,(2*k)],final_healthy[j,(2*k+1)]=np.nan,np.nan
        else:
            x=np.array(data_new_healthy2.iloc[:,1])
            x1=x.reshape(-1,1)
            y=np.array(data_new_healthy2.iloc[:,0])
            y1=y.reshape(-1,1)
            reg=LinearRegression().fit(x1,y1)
            b1=reg.coef_
            b0=reg.intercept_
            final_healthy[j,(2*k)],final_healthy[j,(2*k+1)]=b0,b1


# Replacing 0 with NaN in diseased data
final_diseased=pd.DataFrame(final_diseased)
final_diseased.replace(0,np.nan,inplace=True)

# Replacing 0 with NaN in healthy data
final_healthy=pd.DataFrame(final_healthy)
final_healthy.replace(0,np.nan,inplace=True)


# Filling NaN values of diseased patient with median value
# for columns having more than 90% as null values are remained as it
# and for remaining columns null valus are filled with
# corresponding median value 


m=0
while (m<1202):
    if final_diseased[m].isnull().sum()>=1100:  
       m=m+1
    else:
       a=pd.notnull(final_diseased[m])
       a1=pd.DataFrame(final_diseased[a])
       a2=list(a1[m])
       a2.sort()
       a3=pd.DataFrame(a2)
       a4=list(a3.median())
       final_diseased[m].fillna(a4[0],inplace=True)
       m=m+1


# Filling NaN values of healthy patient with mode value
m=0
while (m<1202):
    if final_healthy[m].isnull().sum()>=1250:  
       m=m+1
    else:
       a=pd.notnull(final_healthy[m])
       a1=pd.DataFrame(final_healthy[a])
       a2=list(a1[m])
       a2.sort()
       a3=pd.DataFrame(a2)
       a4=list(a3.median())
       final_healthy[m].fillna(a4[0],inplace=True)
       m=m+1



## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Adding y_flag in Diseased data            
## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
y_flag=pd.DataFrame(np.zeros(len(final_diseased)))
diseased=np.concatenate((final_diseased,y_flag),axis=1)


## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Adding y_flag in Diseased data            
## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
y_flag=pd.DataFrame(np.ones(len(final_healthy)))
healthy=np.concatenate((final_healthy,y_flag),axis=1)


## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Concatenating diseased and healthy patient            
## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
patient_total=pd.DataFrame(np.concatenate((diseased,healthy),axis=0))


## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Removing columns having null values           
## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
m=0
col_nan=[]
while (m<len(patient_total.columns)):
    if patient_total.iloc[:,m].isnull().sum()>=1:  
       col_nan.append(m)
       m=m+1
    else:
       m=m+1

patient_total.drop(col_nan,axis=1,inplace=True) 


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Seperating input and output
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Columns of patient_total1
col_p=list(patient_total.columns)
col_p1=(col_p.pop())

# Output data
y=pd.DataFrame(patient_total[col_p1].values)    

# Feature selection
features=list(set(col_p)-set([col_p1]))

# Input data
x=pd.DataFrame(patient_total[features].values)

# Standarizing input data x
scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)
x=pd.DataFrame(x)

# Concatenating input and output
patient_final=pd.DataFrame(np.concatenate((x,y),axis=1))

# Removing outlier
for i in range(173):
    patient_final=patient_final[patient_final[i]<1]
    patient_final=patient_final[patient_final[i]>-1]


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Seperating input and output
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@s
    
# Columns of patient_total1
col_p=list(patient_final.columns)
col_p1=(col_p.pop())

# Output data
y=pd.DataFrame(patient_final[col_p1].values)    
#y=y.astype(int)


# Feature selection
features=list(set(col_p)-set([col_p1]))

# Input data
x=pd.DataFrame(patient_final[features].values)
           

#### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
## Splitting the data into train and test
#### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
#
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)  


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Logistic Regression
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 


# Make an instance of the model
logistic=LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x,train_y)

# to get the coefficient of logitic model
logistic.coef_

# to get intercept
logistic.intercept_

# Prediction from test data
prediction=pd.DataFrame(logistic.predict(test_x)) 

## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
## Performance of model
## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
# Confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_y,prediction))

# Calculating the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(test_y,prediction))



# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Desining KNN model
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier

# Storing the K nearest neighbours classifier
KNN_classifier=KNeighborsClassifier(n_neighbors=10)

# Fitting the values for x and y
KNN_classifier.fit(train_x,train_y)

# Predicting the test values with model
prediction=pd.DataFrame(KNN_classifier.predict(test_x))

# Confusion matrix
print(confusion_matrix(test_y,prediction))

# Calculating the accuracy
print(accuracy_score(test_y,prediction))

