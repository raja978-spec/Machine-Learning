import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
import os

data = pd.read_csv(os.path.join('data.csv'))
#print(data.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 270 entries, 0 to 269
Data columns (total 14 columns):
 #   Column                   Non-Null Count  Dtype
---  ------                   --------------  -----
 0   Age                      270 non-null    int64
 1   Sex                      270 non-null    int64
 2   Chest pain type          270 non-null    int64
 3   BP                       270 non-null    int64
 4   Cholesterol              270 non-null    int64
 5   FBS over 120             270 non-null    int64
 6   EKG results              270 non-null    int64
 7   Max HR                   270 non-null    int64
 8   Exercise angina          270 non-null    int64
 9   ST depression            270 non-null    float64
 10  Slope of ST              270 non-null    int64
 11  Number of vessels fluro  270 non-null    int64
 12  Thallium                 270 non-null    int64
 13  Heart Disease            270 non-null    object
dtypes: float64(1), int64(12), object(1)
memory usage: 29.7+ KB
None
'''
# print(data.isna().sum()) 0 
# print(data.isnull().sum()) 0

#sns.pairplot(data.iloc[:,2:], kind='scatter')
#plt.show()

features = data.iloc[:,2:12]

trans = LabelEncoder()
labels = trans.fit_transform(data[['Heart Disease']])
labels = labels.ravel()
print(labels.shape)

train_features, test_feature, train_labels, test_labels = train_test_split(features,labels,
                                             test_size=0.2)
model = LogisticRegression()
model.fit(train_features, train_labels)

predicted_lables = model.predict(test_feature)

score = accuracy_score(test_labels, predicted_lables)
print('{:.2f}'.format(score))

con = confusion_matrix(test_labels, predicted_lables)

sns.heatmap(con, annot=True, cbar=True, cmap='Blues')
plt.show()




















