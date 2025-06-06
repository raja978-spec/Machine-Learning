import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')

features = df.drop('Outcome',axis=1)
labels  = df['Outcome']

train_feature, test_feature, train_labels, test_labels = train_test_split(features,labels, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(train_feature, train_labels)

model_train_prediction = model.predict(train_feature)
val_score = accuracy_score(train_labels, model_train_prediction)

model_prediction = model.predict(test_feature)
test_score = accuracy_score(test_labels, model_prediction)

plt.bar(x=['validation_score','test_score'],height=[val_score,test_score])
plt.show()


