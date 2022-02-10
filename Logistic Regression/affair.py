import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

affairs1 = pd.read_csv("file:///D:/DATA SCIENCE/ExcelR/Assignments/Logistic Regression/affairs.csv")
affairs1

affairs1["affairs_yn"] = 0
affairs1.loc[affairs1.affairs != 0, "affairs_yn"] = 1

affairs1.columns

affairs1.isnull().sum()

# creating dummy columns for the categorical columns 

affairs1_dummies = pd.get_dummies(affairs1[["gender", "children"]])

# Dropping the columns for which we have created dummies

affairs1.drop(["gender", "children"], inplace = True, axis = 1)

# adding the columns to the affairs1 data frame

affairs1 = pd.concat([affairs1, affairs1_dummies], axis = 1)
affairs1.columns
affairs1.shape

affairs1.affairs.value_counts()
affairs1.affairs_yn.value_counts()

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

affairs1.head(10)

# getting the barplot for the catergorical columns

sb.countplot(x="affairs_yn",data= affairs1, palette="hls")
pd.crosstab(affairs1.affairs_yn,affairs1.gender_female).plot(kind="bar")
pd.crosstab(affairs1.affairs_yn,affairs1.gender_male).plot(kind="bar")

sb.countplot(x="gender_female",data= affairs1,palette="hls")
pd.crosstab(affairs1.gender_female,affairs1.children_no).plot(kind="bar")
pd.crosstab(affairs1.gender_female,affairs1.children_yes).plot(kind="bar")

sb.countplot(x="gender_male",data= affairs1,palette="hls")
pd.crosstab(affairs1.gender_male,affairs1.children_no).plot(kind="bar")
pd.crosstab(affairs1.gender_male,affairs1.children_yes).plot(kind="bar")

sb.countplot(x="children_no",data= affairs1, palette="hls")
pd.crosstab(affairs1.children_no,affairs1.gender_female).plot(kind="bar")

sb.countplot(x="children_yes",data= affairs1, palette="hls")
pd.crosstab(affairs1.children_yes,affairs1.gender_male).plot(kind="bar")

# Data Distribution - Boxplot of continuous variables wrt to each category of categorical columns

sb.boxplot(x="affairs_yn",y="gender_female",data=affairs1,palette="hls")
sb.boxplot(x="affairs_yn",y="gender_male",data=affairs1,palette="hls")

sb.boxplot(x="affairs_yn",y="children_no",data=affairs1,palette="hls")
sb.boxplot(x="affairs_yn",y="childer_yes",data=affairs1,palette="hls")

sb.boxplot(x="gender_female",y="children_no",data= affairs1,palette="hls")
sb.boxplot(x="gender_male",y="children_yes",data= affairs1,palette="hls")

affairs1.shape

# model building 

from sklearn.linear_model import LogisticRegression

X = affairs1.iloc[:,[1,2,3,4,5,6,8,9,10,11]]
Y = affairs1.iloc[:,7]

classifier = LogisticRegression()
classifier.fit(X,Y)

classifier.coef_ # coefficients of features
classifier.predict_proba (X) # probability values

y_pred = classifier.predict(X)

affairs1["y_pred"] = y_pred
y_prob = pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))

new_df = pd.concat([affairs1, y_prob], axis =1)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y, y_pred)
print(confusion_matrix)
type(y_pred)
accuracy = sum(Y==y_pred)/affairs1.shape[0]
print(accuracy)
pd.crosstab(y_pred,Y)
































