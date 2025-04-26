import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

path = os.path.join('dataset','data.csv')
data=pd.read_csv(path)
sns.lineplot(data,x='x',y='y')

poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(np.array(data['x']).reshape(-1, 1))

X = pd.DataFrame(poly_features, columns=[f'x^{i}' for i in range(poly_features.shape[1])])
y = data['y'].values.reshape(-1, 1)

data['y'] = np.array(data['y']).reshape(-1,1)


model = LinearRegression()
model.fit(data[['x']], data[['y']])

test_features =  np.linspace(data[['x']].min(), data[['y']].max(), 300).reshape(-1,1)
predict_lables = model.predict(test_features)

from sklearn.metrics import mean_absolute_error

errors = mean_absolute_error(test_features, predict_lables)

plt.plot(test_features, predict_lables)
plt.show()
print('{:.2f}'.format(errors))