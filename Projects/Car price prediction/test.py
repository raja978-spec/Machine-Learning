import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

import os
import seaborn as sns
import matplotlib.pyplot as plt

path_to_dataset =os.path.join('cardekho_dataset.csv')
data=pd.read_csv(path_to_dataset)

columns = data.columns.to_list()
non_numeric_columns = []
for i in range(len(columns)):
    if data[columns[i]].dtype == 'object':
        non_numeric_columns.append(columns[i])

numeric_data = data.drop(columns=non_numeric_columns)
plt.title('Correlation of numeric features')
sns.heatmap(data=numeric_data.corr(), annot=True)
plt.tight_layout(pad=2.0)
#plt.show()


train_feature, test_feature, train_labels, test_labels = train_test_split(numeric_data.iloc[:,1:-1],numeric_data.iloc[:,-1:],
                                                                          test_size=0.2)
model = LinearRegression()
model.fit(train_feature,train_labels)

predicted_labels = model.predict(test_feature)
error = mean_squared_error(test_labels, predicted_labels)
print('Errors: {:.2f}'.format(error))

r2 = r2_score(test_labels, predicted_labels)
print("R² Score:", r2)

'''
OUTPUT:

Errors: 217770954177.99
R² Score: 0.7022954897380547
'''