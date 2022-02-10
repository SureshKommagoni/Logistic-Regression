import pandas as pd
import numpy as np

affair = pd.read_csv("file:///D:/DATA SCIENCE/ExcelR/Assignments/Logistic Regression/affairs.csv")

affair

affair['affair_yn'] = 0

affair.loc[affair.affairs != 0, "affair_yn" ] = 1

affair.head(10)
affair.columns


affair['gender'].replace({'female':0, 'male':1}, inplace = True)
affair['children'].replace({'no':0, 'yes':1}, inplace = True)

affair.shape

#for visulization and plotting
import seaborn as sb
import matplotlib.pyplot as plt
#to apply logistic regresison
from sklearn.linear_model import LogisticRegression
#to split the dataset into train and test
from sklearn.model_selection import train_test_split # trian and test
#to create the confusion matrix
from sklearn import metrics
from sklearn import preprocessing 
from sklearn.metrics import classification_report

affair.columns

sb.countplot(x="affair_yn",data= affair, palette="hls")
pd.crosstab(affair.affair_yn, affair.gender).plot(kind = 'bar')
pd.crosstab(affair.affair_yn, affair.children).plot(kind = 'bar')

sb.countplot(x = "gender", data = affair, palette = "hls")

# Data Distribution - Boxplot of continuous variables wrt to each category of categorical columns


sb.boxplot(x="affair_yn",y="gender",data=affair,palette="hls")
sb.boxplot(x="affair_yn",y="children",data=affair,palette="hls")

affair.drop(["affairs"], inplace = True, axis = 1)
affair.columns

from sklearn.linear_model import LogisticRegression

x = affair.iloc[:,[0,1,2,3,4,5,6,7]]
y = affair.iloc[:,8]

classifier = LogisticRegression()
classifier.fit(x,y)

classifier.coef_ # coeffient of features

classifier.predict_proba(x) # probability values

y_pred = classifier.predict(x)

affair["y_pred"]= y_pred

y_prob = pd.DataFrame(classifier.predict_proba(x.iloc[:,:]))

new_df = pd.concat([affair, y_prob], axis =1)

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y, y_pred)
print(confusion_matrix)

type(y_pred)

accuracy = sum(y == y_pred)/affair.shape[0]
print(accuracy)
pd.crosstab(y_pred,y)















