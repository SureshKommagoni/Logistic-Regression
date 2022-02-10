import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report

# loading the dataset

affair_yn = pd.read_csv("file:///D:/DATA SCIENCE/ExcelR/Assignments/Logistic Regression/affairs.csv")
affair_yn.head(10)

affair_yn.describe()

affair_yn.columns

type(affair_yn)

affair_yn.loc[(affair_yn["affairs"] > 0, "affairs")] = 1
affair_yn.loc[(affair_yn["affairs"] == 0, "affairs")] = 0

affair_yn["gender"] = affair_yn["gender"].replace({"female":0, "male":1})
affair_yn["children"] = affair_yn["children"].replace({"yes":1, "no":0})

affair_yn.head()

sb.countplot( x= "gender", data = affair_yn, palette = "hls")
pd.crosstab(affair_yn.affairs, affair_yn.gender).plot(kind = "bar")

sb.countplot(x="children", data = affair_yn, palette = "hls")
pd.crosstab(affair_yn.affairs, affair_yn.children).plot(kind = "bar")

sb.boxplot(x = "affairs", y = "gender", data = affair_yn, palette = "hls")

sb.boxplot(x = "affairs", y = "children", data = affair_yn, palette = "hls")


affair_yn.isnull().sum()

affair_yn.count()

affair_yn.shape


x = affair_yn.iloc[:,1:]
y = affair_yn.iloc[:,:1]

classifier = LogisticRegression()
classifier.fit(x,y)

classifier.coef_

classifier.predict_proba(x)

y_pred = classifier.predict_proba(x)

affair_yn["y_pred"] = y_pred

