import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
from patsy import dmatrices
import sklearn
import seaborn as sns
dataframe=pd.read_csv("IBM Attrition Data.csv")
dataframe.head()
names = dataframe.columns.values
print(names)

# histogram for age\n",
plt.figure(figsize=(10,8))
dataframe['Age'].hist(bins=70)
plt.title("Age distribution of Employees")
plt.xlabel("Age")
plt.ylabel("# of Employees")
plt.show()

# explore data for Attrition by Age\n",
plt.figure(figsize=(14,10))
plt.scatter(dataframe.Attrition,dataframe.Age, alpha=.55)
plt.title("Attrition by Age ")
plt.ylabel("Age")
plt.grid(b=True, which='major',axis='y')
plt.show()

# explore data for Left employees breakdown\n",
plt.figure(figsize=(8,6))
dataframe.Attrition.value_counts().plot(kind='barh',color='blue',alpha=.65)
plt.title("Attrition breakdown ")
plt.show()

# explore data for Education Field distribution\n",
plt.figure(figsize=(10,8))
dataframe.EducationField.value_counts().plot(kind='barh',color='g',alpha=.65)
plt.title("Education Field Distribution")
plt.show()

# explore data for Marital Status\n",
plt.figure(figsize=(8,6))
dataframe.MaritalStatus.value_counts().plot(kind='bar',alpha=.5)
plt.show()

dataframe.describe()
dataframe.info()
dataframe.columns
dataframe.std()
dataframe['Attrition'].value_counts()
dataframe['Attrition'].dtypes
dataframe['Attrition'].replace('Yes',1, inplace=True)
dataframe['Attrition'].replace('No',0, inplace=True)
dataframe.head(10)

# building up a logistic regression model\n",
X = dataframe.drop(['Attrition'],axis=1)
X.head()
Y = dataframe['Attrition']
Y.head()

dataframe['EducationField'].replace('Life Sciences',1, inplace=True)
dataframe['EducationField'].replace('Medical',2, inplace=True)
dataframe['EducationField'].replace('Marketing', 3, inplace=True)
dataframe['EducationField'].replace('Other',4, inplace=True)
dataframe['EducationField'].replace('Technical Degree',5, inplace=True)
dataframe['EducationField'].replace('Human Resources', 6, inplace=True)
dataframe['EducationField'].value_counts()

dataframe['Department'].value_counts()
dataframe['Department'].replace('Research & Development',1, inplace=True)
dataframe['Department'].replace('Sales',2, inplace=True)
dataframe['Department'].replace('Human Resources', 3, inplace=True)
dataframe['Department'].value_counts()

dataframe['MaritalStatus'].value_counts()
dataframe['MaritalStatus'].replace('Married',1, inplace=True)
dataframe['MaritalStatus'].replace('Single',2, inplace=True)
dataframe['MaritalStatus'].replace('Divorced',3, inplace=True)
dataframe['MaritalStatus'].value_counts()

x=dataframe.select_dtypes(include=['int64'])
x.dtypes
x.columns
y=dataframe['Attrition']
y.head()
y, x = dmatrices('Attrition ~ Age + Department + DistanceFromHome + Education + EducationField + YearsAtCompany',dataframe, return_type="dataframe")
print (x.columns)
y = np.ravel(y)

#from sklearn.linear_model import LogisticRegression   (remove this)

model = LogisticRegression()
model = model.fit(x, y)
# check the accuracy on the training set
model.score(x, y)
y.mean()

X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y, test_size=0.3, random_state=0)
model2=LogisticRegression()
model2.fit(X_train, y_train)

predicted= model2.predict(X_test)
print(predicted)

probs = model2.predict_proba(X_test)
print(probs)

from sklearn import metrics
print(metrics.accuracy_score(y_test, predicted))
print(metrics.roc_auc_score(y_test, probs[:, 1]))
print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))

print(X_train)

#add random values to KK according to the parameters mentioned above to check the proabily of attrition of the employee\n",
kk=[[1.0, 23.0, 1.0, 500.0, 3.0, 24.0, 1.0]]
print(model.predict_proba(kk))
