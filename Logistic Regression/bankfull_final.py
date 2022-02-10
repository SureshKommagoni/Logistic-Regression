#for data import and basic oprtaion
import pandas as pd
import numpy as np

bankfull = pd.read_csv("file:///D:/DATA SCIENCE/ExcelR/Assignments/Logistic Regression/bankfull.csv")

bankfull.columns
bankfull.dropna()

# creating dummy columns for the categorical columns 
bank_dummies = pd.get_dummies(bankfull[[ "job", "marital", "education", "default","housing", "loan", "contact", "month","poutcome" ]])

# Dropping the columns for which we have created dummies
bankfull.drop([ "job", "marital", "education", "default","housing", "loan", "contact", "month","poutcome" ], inplace = True, axis = 1)

# adding the columns to the bankfull data frame 
bankfull = pd.concat([bankfull, bank_dummies],axis =1)
bankfull.columns
bankfull.drop(bankfull.columns[[19,26,35,51]], inplace = True, axis = 1)
bankfull.shape

bankfull["tdeposit"] = 0
bankfull.loc[bankfull.y=="yes","tdeposit"] = 1 
bankfull.drop(["y"], inplace = True, axis = 1)
bankfull.y.value_counts()
bankfull.tdeposit.value_counts()

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

bankfull.head(10)

# getting the barplot for the catergorical columns

sb.countplot(x = 'tdeposit', data = bankfull, palette = "hls")
pd.crosstab(bankfull.tdeposit , bankfull.age).plot(kind="bar")


#
#

###################
# model building 

# from sklean.linear_model import LogisticRegression

bankfull.shape

X = bankfull.iloc[:,:47]
Y = bankfull.iloc[:,47:]

classifier = LogisticRegression()
classifier.fit(X,Y)

classifier.coef_ # coefficients of features

classifier.predict_proba (X) # probability values

y_pred = classifier.predict(X)

bankfull["y_pred"] = y_pred

y_prob = pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))

new_df = pd.concat([bankfull, y_prob], axis =1)

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(Y, y_pred)
print(confusion_matrix)
type(y_pred)
accuracy = sum(Y==y_pred)/bankfull.shape[0]
pd.crosstab(y_pred,Y)

















