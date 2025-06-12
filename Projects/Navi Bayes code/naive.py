import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('IRIS.csv')

features = data.iloc[:,:4]

labels = data.iloc[:,-1]

transformer = LabelEncoder()
labels = transformer.fit_transform(labels)

train_features, test_features,train_labels, test_lables = train_test_split(features,
                                                                            labels, 
                                                                            test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(train_features, train_labels)

predicted_lables = model.predict(test_features)

report = classification_report(test_lables, predicted_lables)
print(report)