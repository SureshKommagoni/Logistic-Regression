#for data import and basic oprtaion
import pandas as pd
import numpy as np

bankfull = pd.read_csv("file:///D:/ExcelR/Assignments/Logistic Regression/bankfull.csv")

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
pd.crosstab(bankfull.y , bankfull.age).plot(kind="bar")

sb.countplot(x = )
