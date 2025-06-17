from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd

data = pd.read_csv('fetal_health.csv')

features = data.iloc[:,:-1]

labels = data.iloc[:,-1:]

min_max_transformers = MinMaxScaler()
standard_scaler = StandardScaler()

features = min_max_transformers.fit_transform(features)
features = standard_scaler.fit_transform(features)

train_feature, test_feature, train_labels, test_labels = train_test_split(features, labels,
                                                                          test_size=0.2)
model = LinearSVC(loss='hinge')
model.fit(train_feature, train_labels)

train_predicted_labels = model.predict(train_feature)
test_predicted_labels = model.predict(test_feature)

val_accuracy = accuracy_score(train_labels, train_predicted_labels)
test_accuracy = accuracy_score(test_labels, test_predicted_labels)

print(val_accuracy, test_accuracy)
#0.908235294117647 0.892018779342723
# Overfitting

                        # Model 2 with polynomial kernel
model2= svm.SVC(kernel='poly', degree=3, coef0=1, C=5)
model2.fit(train_feature, train_labels)

train_predicted_labels = model2.predict(train_feature)
test_predicted_labels = model2.predict(test_feature)

val_accuracy = accuracy_score(train_labels, train_predicted_labels)
test_accuracy = accuracy_score(test_labels, test_predicted_labels)

print(val_accuracy, test_accuracy)
#0.9658823529411765 0.9295774647887324

                         #MODEL 3 WITH RADIAL KERNEL 
model3 = svm.SVC(kernel='rbf', C=9)
model3.fit(train_feature, train_labels)

train_predicted_labels = model3.predict(train_feature)
test_predicted_labels = model3.predict(test_feature)

val_accuracy = accuracy_score(train_labels, train_predicted_labels)
test_accuracy = accuracy_score(test_labels, test_predicted_labels)

print(val_accuracy, test_accuracy)
#0.9694117647058823 0.9084507042253521








